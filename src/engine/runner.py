"""High-throughput concurrent evaluation runner with QPS throttling and resilient retries (Zero dependencies)."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import time
from typing import Callable, List, Optional

from src.engine.cache import RunCache
from src.engine.rate_limiter import TokenBucketRateLimiter
from src.models.base import BaseVisionModel
from src.schemas import EvalTask, ModelPrediction


class ConcurrentEvaluationRunner:
    """Executes multimodal evaluation tasks concurrently with strict rate-limiting, retries, and caching."""

    def __init__(
        self,
        model: Optional[BaseVisionModel] = None,
        concurrency: int = 4,
        max_concurrency: Optional[int] = None,
        qps_limit: float = 5.0,
        max_retries: int = 3,
        retry_limit: Optional[int] = None,
        cache: Optional[RunCache] = None,
        bypass_cache: bool = False,
    ):
        self.model = model
        actual_concurrency = max_concurrency if max_concurrency is not None else concurrency
        self.concurrency = max(1, int(actual_concurrency))
        self.qps_limit = float(qps_limit)
        self.max_retries = int(retry_limit if retry_limit is not None else max_retries)
        self.rate_limiter = TokenBucketRateLimiter(rate=self.qps_limit, capacity=max(self.qps_limit, 2.0))
        self.cache = cache or RunCache(bypass_cache=bypass_cache)
        if bypass_cache:
            self.cache.bypass_cache = True

    def _execute_single_task_with_retry(
        self,
        model: BaseVisionModel,
        task: EvalTask,
    ) -> ModelPrediction:
        """Executes a single task with cache lookup, rate limiting, and exponential retry backoff."""
        # 1. Check cache first
        cached_pred = self.cache.get(model.model_name, task)
        if cached_pred is not None:
            return cached_pred

        # 2. Execute with rate-limiter and retry loop
        last_error = None
        for attempt in range(self.max_retries):
            # Throttle via token bucket before firing API call
            self.rate_limiter.acquire(1.0)

            pred = model.predict(task)

            # If success, save to cache and return
            if not pred.error:
                self.cache.set(model.model_name, task, pred)
                return pred

            # Check if error is retryable (e.g. HTTP 429 Rate Limit, HTTP 500/503)
            err_lower = pred.error.lower()
            is_retryable = any(code in err_lower for code in ("429", "500", "502", "503", "504", "rate limit", "quota"))

            if not is_retryable or attempt == self.max_retries - 1:
                return pred

            # Exponential backoff with jitter: 2^attempt + jitter
            backoff_s = (2.0 ** attempt) + random.uniform(0.1, 0.5)
            time.sleep(backoff_s)
            last_error = pred.error

        return ModelPrediction(
            task_id=task.task_id,
            model_name=model.model_name,
            raw_response="[ERROR: Max retries exceeded]",
            latency_ms=0.0,
            error=f"Max retries exceeded. Last error: {last_error}",
        )

    def run_batch(
        self,
        model_or_tasks: any,
        tasks: Optional[List[EvalTask]] = None,
        on_progress: Optional[Callable[[int, int, ModelPrediction], None]] = None,
    ) -> List[ModelPrediction]:
        """Runs a list of tasks concurrently across worker threads while preserving original task order."""
        if isinstance(model_or_tasks, BaseVisionModel):
            model = model_or_tasks
            actual_tasks = tasks or []
        else:
            if self.model is None:
                raise ValueError("Model must be provided in runner __init__ or run_batch.")
            model = self.model
            actual_tasks = model_or_tasks
            if callable(tasks) and on_progress is None:
                on_progress = tasks

        total_tasks = len(actual_tasks)
        predictions_map = {}
        completed_count = 0
        start_time = time.perf_counter()

        print(f"\n⚡ Concurrently evaluating {total_tasks} tasks on [{model.model_name}] (concurrency={self.concurrency}, QPS={self.qps_limit})...")

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {
                executor.submit(self._execute_single_task_with_retry, model, task): (i, task)
                for i, task in enumerate(actual_tasks)
            }

            for future in as_completed(future_to_idx):
                idx, task = future_to_idx[future]
                completed_count += 1
                try:
                    pred = future.result()
                except Exception as e:
                    pred = ModelPrediction(
                        task_id=task.task_id,
                        model_name=model.model_name,
                        raw_response="[CRITICAL UNCAUGHT EXCEPTION]",
                        latency_ms=0.0,
                        error=str(e),
                    )

                predictions_map[idx] = pred

                if on_progress:
                    on_progress(completed_count, total_tasks, pred)
                else:
                    cache_tag = " (⚡ cached)" if pred.is_cached else ""
                    status = "❌ ERR" if pred.error else f"✅ {pred.latency_ms:.0f}ms{cache_tag}"
                    print(f"  [{completed_count:>2}/{total_tasks}] Task {task.task_id:<20} | {status}")

        elapsed_total = time.perf_counter() - start_time
        effective_qps = (total_tasks / elapsed_total) if elapsed_total > 0 else 0.0
        cached_count = sum(1 for p in predictions_map.values() if p.is_cached)
        print(f"✨ Batch completed in {elapsed_total:.2f}s (Effective QPS: {effective_qps:.1f} | Cached: {cached_count}/{total_tasks})")

        # Return predictions sorted in original input task order
        return [predictions_map[i] for i in range(total_tasks)]

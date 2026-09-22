"""Persistent response cache and checkpointing for evaluation runs (Zero dependencies)."""

import hashlib
import json
import os
import threading
from typing import Dict, Optional

from src.schemas import EvalTask, ModelPrediction


class RunCache:
    """Persistent response cache to prevent redundant API calls and enable fast crash recovery.
    
    Includes explicit bypass support for non-deterministic sampling (e.g. pass@k or temperature > 0).
    """

    def __init__(self, cache_file: str = ".eval_cache.jsonl", bypass_cache: bool = False, cache_path: Optional[str] = None):
        self.cache_file = cache_path or cache_file
        self.bypass_cache = bypass_cache
        self._memory_cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._load_cache()

    def _compute_key(self, model_name: str, task: EvalTask) -> str:
        """Computes a deterministic hash key for a task + model configuration."""
        raw_key = f"{model_name}:{task.task_id}:{task.prompt}:{task.image_path_or_url}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def _load_cache(self) -> None:
        """Loads existing entries from the JSONL cache file."""
        if not os.path.exists(self.cache_file):
            return

        with self._lock:
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            k = entry.get("cache_key")
                            if k:
                                self._memory_cache[k] = entry
            except Exception as e:
                print(f"⚠️ Warning: Could not read cache file {self.cache_file}: {e}")

    def get(self, model_name: str, task: EvalTask) -> Optional[ModelPrediction]:
        """Retrieves a cached prediction if available and caching is not bypassed."""
        if self.bypass_cache:
            return None

        key = self._compute_key(model_name, task)
        with self._lock:
            entry = self._memory_cache.get(key)

        if not entry:
            return None

        pred_dict = entry.get("prediction", {})
        return ModelPrediction(
            task_id=pred_dict.get("task_id", task.task_id),
            model_name=pred_dict.get("model_name", model_name),
            raw_response=pred_dict.get("raw_response", ""),
            latency_ms=pred_dict.get("latency_ms", 0.0),
            input_tokens=pred_dict.get("input_tokens"),
            output_tokens=pred_dict.get("output_tokens"),
            cost_usd=pred_dict.get("cost_usd"),
            is_cached=True,
            error=pred_dict.get("error"),
        )

    def set(self, model_name: str, task: EvalTask, prediction: ModelPrediction) -> None:
        """Saves a prediction to memory and appends to the JSONL cache file."""
        # Never cache failed predictions or if caching is bypassed
        if self.bypass_cache or prediction.error:
            return

        key = self._compute_key(model_name, task)
        entry = {
            "cache_key": key,
            "task_id": task.task_id,
            "model_name": model_name,
            "prediction": {
                "task_id": prediction.task_id,
                "model_name": prediction.model_name,
                "raw_response": prediction.raw_response,
                "latency_ms": prediction.latency_ms,
                "input_tokens": prediction.input_tokens,
                "output_tokens": prediction.output_tokens,
                "cost_usd": prediction.cost_usd,
            },
        }

        with self._lock:
            self._memory_cache[key] = entry
            try:
                with open(self.cache_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                print(f"⚠️ Warning: Could not write to cache file {self.cache_file}: {e}")

    def clear(self) -> None:
        """Clears both memory and disk cache."""
        with self._lock:
            self._memory_cache.clear()
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)

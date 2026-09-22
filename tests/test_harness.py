"""Unit tests for Multimodal Evaluation Harness (Zero external dependencies)."""

import os
import shutil
import sys
import tempfile
import time
import unittest

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schemas import EvalTask, ModelPrediction, TaskCategory, PerturbationType
from src.engine.rate_limiter import TokenBucketRateLimiter
from src.engine.cache import RunCache
from src.engine.runner import ConcurrentEvaluationRunner
from src.models.base import BaseVisionModel
from src.models.mock_model import MockVisionModel
from src.models.registry import get_model, list_supported_providers
from src.models.utils import calculate_token_cost


class TestRateLimiter(unittest.TestCase):
    """Verifies token bucket rate limiter math and timing."""

    def test_rate_limiter_immediate_acquire(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=10.0)
        start = time.perf_counter()
        limiter.acquire(1.0)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.05)

    def test_rate_limiter_throttles(self):
        limiter = TokenBucketRateLimiter(rate=5.0, capacity=1.0)
        limiter.acquire(1.0)  # Consumes full capacity
        start = time.perf_counter()
        limiter.acquire(1.0)  # Must wait ~0.2s
        elapsed = time.perf_counter() - start
        self.assertGreaterEqual(elapsed, 0.15)


class TestRunCache(unittest.TestCase):
    """Verifies persistent JSONL caching and bypass semantics."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self.temp_dir, "test_cache.jsonl")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_cache_miss_then_hit(self):
        cache = RunCache(cache_file=self.cache_file)
        task = EvalTask(
            task_id="test_task_1",
            category=TaskCategory.CHART_UNDERSTANDING,
            prompt="What is the revenue?",
            image_path_or_url="dummy.png",
            ground_truth="100",
        )

        # Cache Miss
        self.assertIsNone(cache.get("test-model", task))

        # Save to cache
        pred = ModelPrediction(
            task_id=task.task_id,
            model_name="test-model",
            raw_response="100",
            latency_ms=12.5,
            input_tokens=50,
            output_tokens=5,
            cost_usd=0.0001,
        )
        cache.set("test-model", task, pred)

        # Cache Hit
        cached_pred = cache.get("test-model", task)
        self.assertIsNotNone(cached_pred)
        self.assertEqual(cached_pred.raw_response, "100")
        self.assertTrue(cached_pred.is_cached)
        self.assertEqual(cached_pred.cost_usd, 0.0001)

    def test_bypass_cache_flag(self):
        cache = RunCache(cache_file=self.cache_file, bypass_cache=True)
        task = EvalTask(
            task_id="test_task_2",
            category=TaskCategory.DOCUMENT_OCR,
            prompt="Read total",
            image_path_or_url="dummy.png",
            ground_truth="$50",
        )
        pred = ModelPrediction(
            task_id=task.task_id,
            model_name="test-model",
            raw_response="$50",
            latency_ms=10.0,
        )
        cache.set("test-model", task, pred)
        # Bypassed cache must return None even if entry was set
        self.assertIsNone(cache.get("test-model", task))


class TestModelRegistry(unittest.TestCase):
    """Verifies factory instantiation and provider dispatch."""

    def test_get_mock_model(self):
        m = get_model("mock")
        self.assertIsInstance(m, MockVisionModel)
        self.assertEqual(m.model_name, "mock")

    def test_get_explicit_provider(self):
        from src.models.gemini_model import GeminiVisionModel
        from src.models.openai_model import OpenAIVisionModel
        from src.models.anthropic_model import AnthropicVisionModel

        m_gem = get_model("gemini:gemini-1.5-flash", api_key="dummy")
        self.assertIsInstance(m_gem, GeminiVisionModel)

        m_oai = get_model("openai:gpt-4o", api_key="dummy")
        self.assertIsInstance(m_oai, OpenAIVisionModel)

        m_ant = get_model("anthropic:claude-3-5-sonnet-20241022", api_key="dummy")
        self.assertIsInstance(m_ant, AnthropicVisionModel)

    def test_list_supported_providers(self):
        catalog = list_supported_providers()
        self.assertIn("gemini", catalog)
        self.assertIn("openai", catalog)
        self.assertIn("anthropic", catalog)


class TestTokenCostCalculation(unittest.TestCase):
    """Verifies token pricing engine across VLM families."""

    def test_gemini_pricing(self):
        cost = calculate_token_cost("gemini-1.5-flash", input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(cost, 0.075 + 0.30, places=4)

    def test_gpt4o_pricing(self):
        cost = calculate_token_cost("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(cost, 2.50 + 10.00, places=4)

    def test_claude_pricing(self):
        cost = calculate_token_cost("claude-3-5-sonnet", input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(cost, 3.00 + 15.00, places=4)


class TestConcurrentRunner(unittest.TestCase):
    """Verifies multi-threaded batch execution and ordering."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self.temp_dir, "runner_cache.jsonl")
        self.cache = RunCache(cache_file=self.cache_file)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_batch_execution_preserves_order(self):
        model = MockVisionModel(model_name="FastMock", base_latency_ms=10.0)
        tasks = [
            EvalTask(
                task_id=f"task_{i}",
                category=TaskCategory.FINE_GRAINED_PERCEPTION,
                prompt=f"Count items {i}",
                image_path_or_url="dummy.png",
                ground_truth=str(i),
            )
            for i in range(10)
        ]

        runner = ConcurrentEvaluationRunner(
            model=model,
            concurrency=4,
            qps_limit=100.0,
            cache=self.cache,
        )

        predictions = runner.run_batch(tasks)
        self.assertEqual(len(predictions), 10)
        for i, pred in enumerate(predictions):
            self.assertEqual(pred.task_id, f"task_{i}")
            self.assertFalse(pred.is_cached)

        # Second run should be 100% cached
        cached_preds = runner.run_batch(tasks)
        self.assertEqual(len(cached_preds), 10)
        for pred in cached_preds:
            self.assertTrue(pred.is_cached)


class TestVisualCorruptions(unittest.TestCase):
    """Verifies pure standard library image corruptions and PNG codecs."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_png = os.path.join(self.temp_dir, "sample.png")
        # Create a simple 20x20 test image (RGB)
        pixels = bytearray([100, 150, 200] * (20 * 20))
        from src.perturbations.image_corruptions import write_png
        write_png(self.test_png, 20, 20, 3, pixels)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_png_codec_roundtrip(self):
        from src.perturbations.image_corruptions import read_png, write_png
        w, h, c, pixels = read_png(self.test_png)
        self.assertEqual((w, h, c), (20, 20, 3))
        self.assertEqual(len(pixels), 20 * 20 * 3)

        out_path = os.path.join(self.temp_dir, "roundtrip.png")
        write_png(out_path, w, h, c, pixels)
        w2, h2, c2, px2 = read_png(out_path)
        self.assertEqual((w2, h2, c2), (20, 20, 3))
        self.assertEqual(pixels, px2)

    def test_pixel_noise(self):
        from src.perturbations.image_corruptions import apply_pixel_noise, read_png
        w, h, c, pixels = read_png(self.test_png)
        noisy = apply_pixel_noise(pixels, w, h, c, sigma=30, seed=42)
        self.assertEqual(len(noisy), len(pixels))
        self.assertNotEqual(noisy, pixels)

    def test_contrast_shift(self):
        from src.perturbations.image_corruptions import apply_contrast_shift, read_png
        w, h, c, pixels = read_png(self.test_png)
        low_contrast = apply_contrast_shift(pixels, w, h, c, factor=0.2)
        # Value 200 compressed towards 128: 128 + 0.2*(200-128) = 142
        self.assertEqual(low_contrast[2], 142)

    def test_occlusion(self):
        from src.perturbations.image_corruptions import apply_occlusion, read_png
        w, h, c, pixels = read_png(self.test_png)
        occluded = apply_occlusion(pixels, w, h, c, box_rel=(0.0, 0.0, 0.5, 0.5), fill_color=(0, 0, 0))
        # Top-left pixel should be black (0, 0, 0)
        self.assertEqual((occluded[0], occluded[1], occluded[2]), (0, 0, 0))
        # Bottom-right pixel (19, 19) should remain uncorrupted (100, 150, 200)
        br_idx = (19 * 20 + 19) * 3
        self.assertEqual((occluded[br_idx], occluded[br_idx+1], occluded[br_idx+2]), (100, 150, 200))

    def test_engine_creates_visual_perturbation(self):
        from src.perturbations.engine import PerturbationEngine
        engine = PerturbationEngine(seed=42)
        task = EvalTask(
            task_id="test_vis_task",
            category=TaskCategory.CHART_UNDERSTANDING,
            prompt="What is the value?",
            image_path_or_url=self.test_png,
            ground_truth="320",
        )
        pert_task = engine.create_perturbed_task(task, PerturbationType.IMAGE_OCCLUSION)
        self.assertTrue(pert_task.is_adversarial)
        self.assertTrue(os.path.exists(pert_task.image_path_or_url))
        self.assertIn("image_occlusion", pert_task.image_path_or_url)


if __name__ == "__main__":
    unittest.main()

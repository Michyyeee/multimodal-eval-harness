"""Mock Vision Model for rapid local testing, CI, and baseline simulation."""

import random
import time
from src.models.base import BaseVisionModel
from src.schemas import EvalTask, ModelPrediction, TaskCategory


class MockVisionModel(BaseVisionModel):
    """Simulates a VLM (e.g. Gemini-Flash or Gemma-VLM) with configurable accuracy & latency profiles."""

    def __init__(
        self,
        model_name: str = "mock-gemini-flash",
        accuracy_rate: float = 0.85,
        base_latency_ms: float = 120.0,
        simulate_adversarial: bool = False,
        **kwargs: any,
    ):
        super().__init__(model_name=model_name)
        self.accuracy_rate = 0.45 if simulate_adversarial else accuracy_rate
        self.base_latency_ms = base_latency_ms
        self.simulate_adversarial = simulate_adversarial

    def predict(self, task: EvalTask) -> ModelPrediction:
        start_time = time.perf_counter()
        
        # Simulate realistic network / inference delay jitter
        simulated_delay = (self.base_latency_ms + random.uniform(-20, 45)) / 1000.0
        time.sleep(max(0.01, simulated_delay))
        
        # Determine whether model answers accurately or hallucinates
        is_correct = random.random() < self.accuracy_rate
        
        if is_correct:
            response_text = task.ground_truth
        else:
            if task.category == TaskCategory.HALLUCINATION_PROBING:
                # Hallucinate that the nonexistent object actually exists
                response_text = "Yes, there is a red delivery truck clearly visible in the foreground."
            elif task.category == TaskCategory.CHART_UNDERSTANDING:
                response_text = "The value in Q3 reached approximately 450 units."
            else:
                response_text = "Based on the image, the answer appears to be inconclusive."

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ModelPrediction(
            task_id=task.task_id,
            model_name=self.model_name,
            raw_response=response_text,
            latency_ms=round(elapsed_ms, 2),
            input_tokens=256,
            output_tokens=len(response_text.split()),
        )

"""Base interface for multimodal / vision-language models."""

from abc import ABC, abstractmethod
from src.schemas import EvalTask, ModelPrediction


class BaseVisionModel(ABC):
    """Abstract interface for all Vision-Language Models evaluated by the harness."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def predict(self, task: EvalTask) -> ModelPrediction:
        """Run multimodal inference on a single task and return prediction + telemetry."""
        pass

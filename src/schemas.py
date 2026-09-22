"""Core data schemas for the Multimodal Evaluation Harness (Zero external dependencies)."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TaskCategory(str, Enum):
    """Categories of visual evaluation tasks."""
    CHART_UNDERSTANDING = "chart_understanding"
    DOCUMENT_OCR = "document_ocr"
    SPATIAL_REASONING = "spatial_reasoning"
    HALLUCINATION_PROBING = "hallucination_probing"
    FINE_GRAINED_PERCEPTION = "fine_grained_perception"


class PerturbationType(str, Enum):
    """Types of adversarial perturbations applied to test robustness."""
    NONE = "none"
    DISTRACTING_CONTEXT = "distracting_context"
    TYPO_NOISE = "typo_noise"
    PROMPT_PARAPHRASE = "prompt_paraphrase"
    IMAGE_CONTRAST_SHIFT = "image_contrast_shift"
    IMAGE_PIXEL_NOISE = "image_pixel_noise"
    IMAGE_OCCLUSION = "image_occlusion"


@dataclass
class EvalTask:
    """A single multimodal evaluation test case."""
    task_id: str
    category: TaskCategory
    prompt: str
    image_path_or_url: str
    ground_truth: str
    acceptable_alternatives: List[str] = field(default_factory=list)
    is_adversarial: bool = False
    perturbation_type: PerturbationType = PerturbationType.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.category, str):
            self.category = TaskCategory(self.category)
        if isinstance(self.perturbation_type, str):
            self.perturbation_type = PerturbationType(self.perturbation_type)


@dataclass
class ModelPrediction:
    """The raw response and telemetry from a model under test."""
    task_id: str
    model_name: str
    raw_response: str
    latency_ms: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    is_cached: bool = False
    error: Optional[str] = None


@dataclass
class JudgeVerdict:
    """Verdict output by an evaluation mechanism (pointwise evaluation)."""
    task_id: str
    model_name: str
    evaluator_name: str
    is_correct: bool
    score: float  # Normalized 0.0 to 1.0
    reasoning: str
    rubric_scores: Dict[str, float] = field(default_factory=dict)
    bias_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairwiseVerdict:
    """Side-by-side comparative verdict between two competing models with position bias detection."""
    task_id: str
    model_a_name: str
    model_b_name: str
    winner: str  # "model_a", "model_b", "tie", or "position_bias_inconclusive"
    has_position_bias: bool
    forward_winner: str
    reverse_winner: str
    reasoning: str


@dataclass
class RobustnessMetrics:
    """Measures model degradation under adversarial perturbations."""
    clean_accuracy: float
    perturbed_accuracy: float
    retention_rate: float  # (perturbed / clean) * 100
    regression_count: int  # Number of tasks where model passed clean but failed perturbed
    regression_rate: float  # (regression_count / total_clean_passed) * 100


@dataclass
class BenchmarkSummary:
    """Aggregated scorecard for a model across the benchmark with statistical confidence intervals."""
    model_name: str
    total_tasks: int
    passed_tasks: int
    accuracy: float
    ci_95_accuracy: Tuple[float, float] = (0.0, 0.0)  # Bootstrap 95% Confidence Interval (lower, upper)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    category_breakdown: Dict[str, float] = field(default_factory=dict)
    hallucination_rate: float = 0.0
    robustness: Optional[RobustnessMetrics] = None

"""Deterministic and heuristic evaluators for multimodal tasks."""

import re
from typing import Optional
from src.schemas import EvalTask, JudgeVerdict, ModelPrediction, TaskCategory


def extract_first_number(text: str) -> Optional[float]:
    """Extracts the first floating-point or integer number from text."""
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return float(match.group()) if match else None


class DeterministicEvaluator:
    """Evaluates model predictions against ground truth using deterministic logic and fuzzy rules."""

    def __init__(self, name: str = "deterministic_v1", numerical_tolerance: float = 0.05):
        self.name = name
        self.numerical_tolerance = numerical_tolerance

    def evaluate(self, task: EvalTask, prediction: ModelPrediction) -> JudgeVerdict:
        norm_pred = prediction.raw_response.strip().lower()
        norm_gt = task.ground_truth.strip().lower()
        norm_alts = [alt.strip().lower() for alt in task.acceptable_alternatives]

        # 1. Check Hallucination Probing
        if task.category == TaskCategory.HALLUCINATION_PROBING:
            # For hallucination probing, ground truth is typically "no / none / not present"
            negative_indicators = ["no", "not present", "none", "cannot see", "does not contain", "not visible", "isn't any"]
            is_refuting = any(ind in norm_pred for ind in negative_indicators)
            
            if is_refuting:
                return JudgeVerdict(
                    task_id=task.task_id,
                    model_name=prediction.model_name,
                    evaluator_name=self.name,
                    is_correct=True,
                    score=1.0,
                    reasoning="Model correctly identified that the queried entity does not exist in the visual context.",
                )
            else:
                return JudgeVerdict(
                    task_id=task.task_id,
                    model_name=prediction.model_name,
                    evaluator_name=self.name,
                    is_correct=False,
                    score=0.0,
                    reasoning=f"Model visually hallucinated the entity. Prediction: '{prediction.raw_response}'",
                    bias_metadata={"hallucination_detected": True}
                )

        # 2. Check Numerical Tolerance (Common in Chart Understanding)
        gt_num = extract_first_number(norm_gt)
        pred_num = extract_first_number(norm_pred)

        if gt_num is not None and pred_num is not None and task.category == TaskCategory.CHART_UNDERSTANDING:
            diff = abs(pred_num - gt_num)
            allowed_error = max(0.01, abs(gt_num) * self.numerical_tolerance)
            if diff <= allowed_error:
                return JudgeVerdict(
                    task_id=task.task_id,
                    model_name=prediction.model_name,
                    evaluator_name=self.name,
                    is_correct=True,
                    score=1.0,
                    reasoning=f"Predicted numerical value {pred_num} is within {self.numerical_tolerance*100}% of ground truth {gt_num}.",
                )

        # 3. String Match / Semantic Substring
        if norm_gt in norm_pred or any(alt in norm_pred for alt in norm_alts):
            return JudgeVerdict(
                task_id=task.task_id,
                model_name=prediction.model_name,
                evaluator_name=self.name,
                is_correct=True,
                score=1.0,
                reasoning=f"Prediction matches ground truth '{task.ground_truth}' or acceptable variants.",
            )

        # 4. Incorrect
        return JudgeVerdict(
            task_id=task.task_id,
            model_name=prediction.model_name,
            evaluator_name=self.name,
            is_correct=False,
            score=0.0,
            reasoning=f"Prediction '{prediction.raw_response}' failed to match ground truth '{task.ground_truth}'.",
        )

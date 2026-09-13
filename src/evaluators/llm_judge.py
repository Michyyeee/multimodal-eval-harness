"""LLM-as-a-Judge module with structured rubrics and pairwise position-bias mitigation."""

import json
import re
from typing import Any, Dict, Optional, Tuple

from src.models.base import BaseVisionModel
from src.schemas import EvalTask, JudgeVerdict, ModelPrediction, PairwiseVerdict, TaskCategory


def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Strips markdown code fences and extracts valid JSON dictionary."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text.strip())
    except Exception:
        # Fallback regex extraction if model added conversational preamble
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Could not parse JSON from judge response: {raw_text[:100]}...")


class MockJudgeModel(BaseVisionModel):
    """Simulated Judge for fast offline testing and CI verification without requiring an API key."""

    def __init__(self, name: str = "mock-gemini-pro-judge"):
        super().__init__(model_name=name)

    def predict(self, task: EvalTask) -> ModelPrediction:
        # Not used directly; judge uses evaluate_prompt
        return ModelPrediction(task_id=task.task_id, model_name=self.model_name, raw_response="", latency_ms=0.0)

    def generate(self, prompt: str) -> str:
        """Simulates judge responses for pointwise and pairwise evaluations."""
        if "Which response is better?" in prompt:
            # Check for simulated answers in pairwise
            if "delivery truck clearly visible" in prompt:
                # Penalize hallucination
                return '{"winner": "A", "reasoning": "Option A correctly refutes the false premise, while Option B hallucinated."}'
            return '{"winner": "A", "reasoning": "Option A is more direct and matches the reference ground truth."}'
        
        # Pointwise rubric evaluation
        if "delivery truck clearly visible" in prompt:
            return json.dumps({
                "visual_grounding": 1.0,
                "factual_accuracy": 1.0,
                "conciseness": 4.0,
                "is_correct": False,
                "reasoning": "Model hallucinated the non-existent red truck in the scene."
            })
        
        return json.dumps({
            "visual_grounding": 5.0,
            "factual_accuracy": 5.0,
            "conciseness": 5.0,
            "is_correct": True,
            "reasoning": "Prediction directly aligns with the reference ground truth."
        })


class LLMJudgeEvaluator:
    """Evaluates multimodal model outputs using an LLM Judge with bias mitigation mechanisms."""

    def __init__(self, judge_model: Any, name: str = "llm_judge_v1"):
        self.judge_model = judge_model
        self.name = name

    def _call_judge(self, prompt: str) -> str:
        """Sends a text-only prompt to the judge model (live API or mock)."""
        if hasattr(self.judge_model, "generate"):
            return self.judge_model.generate(prompt)
        elif hasattr(self.judge_model, "predict"):
            # Use predict with a dummy text-only task
            dummy_task = EvalTask(
                task_id="judge_eval",
                category=TaskCategory.CHART_UNDERSTANDING,
                prompt=prompt,
                image_path_or_url="",
                ground_truth="",
            )
            pred = self.judge_model.predict(dummy_task)
            return pred.raw_response
        raise NotImplementedError("Judge model must support generate() or predict()")

    def evaluate_pointwise(self, task: EvalTask, prediction: ModelPrediction) -> JudgeVerdict:
        """Grades a single prediction on a structured 1-5 rubric using the LLM judge."""
        eval_prompt = f"""You are an expert multimodal evaluation judge.
Evaluate the model prediction against the ground truth reference for the given visual question.

[Visual Task Category]: {task.category.value}
[Question Prompt]: {task.prompt}
[Ground Truth Reference]: {task.ground_truth}
[Acceptable Variations]: {', '.join(task.acceptable_alternatives) if task.acceptable_alternatives else 'None'}
[Model Prediction Under Test]: {prediction.raw_response}

Score the prediction on a 1-5 scale across three criteria:
1. visual_grounding: 5 = perfectly faithful to visual ground truth, 1 = severe visual hallucination.
2. factual_accuracy: 5 = fully accurate, 1 = completely incorrect.
3. conciseness: 5 = concise and informative, 1 = overly verbose or repetitive.

Output strictly valid JSON with this exact schema:
{{
  "visual_grounding": <float 1.0-5.0>,
  "factual_accuracy": <float 1.0-5.0>,
  "conciseness": <float 1.0-5.0>,
  "is_correct": <boolean>,
  "reasoning": "<concise explanation of score>"
}}
"""
        try:
            judge_raw = self._call_judge(eval_prompt)
            parsed = clean_json_response(judge_raw)

            vg = float(parsed.get("visual_grounding", 3.0))
            fa = float(parsed.get("factual_accuracy", 3.0))
            conc = float(parsed.get("conciseness", 3.0))
            is_correct = bool(parsed.get("is_correct", fa >= 4.0))

            # Composite normalized score (0.0 to 1.0)
            composite_score = round(((vg * 0.45) + (fa * 0.45) + (conc * 0.10)) / 5.0, 2)

            return JudgeVerdict(
                task_id=task.task_id,
                model_name=prediction.model_name,
                evaluator_name=self.name,
                is_correct=is_correct,
                score=composite_score,
                reasoning=parsed.get("reasoning", "No reasoning provided."),
                rubric_scores={
                    "visual_grounding": vg,
                    "factual_accuracy": fa,
                    "conciseness": conc,
                },
                bias_metadata={"judge_model": getattr(self.judge_model, "model_name", "unknown")},
            )
        except Exception as e:
            return JudgeVerdict(
                task_id=task.task_id,
                model_name=prediction.model_name,
                evaluator_name=self.name,
                is_correct=False,
                score=0.0,
                reasoning=f"Judge evaluation failed: {str(e)}",
            )

    def evaluate_pairwise(
        self,
        task: EvalTask,
        pred_a: ModelPrediction,
        pred_b: ModelPrediction
    ) -> PairwiseVerdict:
        """Compares two model responses side-by-side with bidirectional order swapping to detect position bias."""

        def build_comparison_prompt(resp_1: str, resp_2: str) -> str:
            return f"""You are an unbiased AI evaluation judge comparing two model responses to a visual reasoning query.

[Question Prompt]: {task.prompt}
[Ground Truth Reference]: {task.ground_truth}

[Response A]:
{resp_1}

[Response B]:
{resp_2}

Which response is better? Consider factual accuracy, visual grounding (absence of hallucination), and directness.
Output strictly JSON:
{{
  "winner": "A" or "B" or "Tie",
  "reasoning": "<1-2 sentence justification>"
}}
"""
        # Call 1: Forward orientation (A = pred_a, B = pred_b)
        raw_1 = self._call_judge(build_comparison_prompt(pred_a.raw_response, pred_b.raw_response))
        parsed_1 = clean_json_response(raw_1)
        w1 = parsed_1.get("winner", "Tie").upper()

        # Call 2: Reverse orientation (A = pred_b, B = pred_a) -> Position Bias Check
        raw_2 = self._call_judge(build_comparison_prompt(pred_b.raw_response, pred_a.raw_response))
        parsed_2 = clean_json_response(raw_2)
        w2 = parsed_2.get("winner", "Tie").upper()

        # Resolve winners
        # In Call 1: 'A' means Model A won, 'B' means Model B won
        winner_call_1 = "model_a" if w1 == "A" else ("model_b" if w1 == "B" else "tie")
        # In Call 2: 'A' means Model B won (since B was first), 'B' means Model A won
        winner_call_2 = "model_b" if w2 == "A" else ("model_a" if w2 == "B" else "tie")

        if winner_call_1 == winner_call_2:
            # Consistent agreement regardless of presentation order
            final_winner = winner_call_1
            has_position_bias = False
            reasoning = parsed_1.get("reasoning", "")
        else:
            # Position bias detected: the model favored a specific position (e.g. always picking option A)
            final_winner = "position_bias_inconclusive"
            has_position_bias = True
            reasoning = f"Position bias detected: Judge chose '{winner_call_1}' when presented first, but chose '{winner_call_2}' when presented second."

        return PairwiseVerdict(
            task_id=task.task_id,
            model_a_name=pred_a.model_name,
            model_b_name=pred_b.model_name,
            winner=final_winner,
            has_position_bias=has_position_bias,
            forward_winner=winner_call_1,
            reverse_winner=winner_call_2,
            reasoning=reasoning,
        )

"""Adversarial Perturbation Engine for testing model robustness and regression."""

import copy
import os
import random
import re
from typing import List, Tuple

from src.schemas import EvalTask, PerturbationType, RobustnessMetrics


class PerturbationEngine:
    """Generates controlled adversarial perturbations (text distraction, typos, and visual transformations)."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def inject_distracting_context(self, prompt: str) -> str:
        """Prepends realistic but irrelevant distractor context before the visual query."""
        distractors = [
            "FYI: The following image was captured during a routine system diagnostic in warehouse sector 4G. Please review all telemetry data. Question: ",
            "Context note: Multiple analysts have examined this document and noted unusual font variances across sections. Regarding the visual: ",
            "Important instruction update: Ensure all values conform to ISO-9001 standard reporting protocols. Question: ",
            "Background: This visualization was created using automated quarterly dashboard sync tools. Query: ",
        ]
        prefix = self.rng.choice(distractors)
        return prefix + prompt

    def inject_typo_noise(self, prompt: str) -> str:
        """Injects subtle keyboard neighbor typos or transposition errors."""
        words = prompt.split()
        if not words:
            return prompt

        # Pick 1-2 words to introduce typos
        idx_to_alter = self.rng.sample(range(len(words)), min(2, len(words)))
        for idx in idx_to_alter:
            w = words[idx]
            if len(w) > 4:
                # Transpose two adjacent letters
                t_idx = self.rng.randint(1, len(w) - 2)
                words[idx] = w[:t_idx] + w[t_idx + 1] + w[t_idx] + w[t_idx + 2:]
            elif len(w) > 2:
                # Common typo substitution
                substitutions = {"a": "s", "e": "r", "i": "o", "o": "p", "t": "r", "h": "g"}
                for char, sub in substitutions.items():
                    if char in w.lower():
                        words[idx] = w.lower().replace(char, sub, 1)
                        break

        return " ".join(words)

    def create_perturbed_task(self, task: EvalTask, perturbation_type: PerturbationType) -> EvalTask:
        """Creates an adversarial variant of an existing task."""
        perturbed = copy.deepcopy(task)
        perturbed.task_id = f"{task.task_id}__pert_{perturbation_type.value}"
        perturbed.is_adversarial = True
        perturbed.perturbation_type = perturbation_type

        if perturbation_type == PerturbationType.DISTRACTING_CONTEXT:
            perturbed.prompt = self.inject_distracting_context(task.prompt)
        elif perturbation_type == PerturbationType.TYPO_NOISE:
            perturbed.prompt = self.inject_typo_noise(task.prompt)
        elif perturbation_type == PerturbationType.IMAGE_CONTRAST_SHIFT:
            # Metadata tag indicating visual transformation
            perturbed.metadata["visual_perturbation"] = "contrast_shift_20pct"
        
        return perturbed

    def generate_stress_test_suite(self, clean_tasks: List[EvalTask]) -> List[Tuple[EvalTask, EvalTask]]:
        """Pairs every clean task with a perturbed counterpart to measure regression."""
        pairs = []
        for i, task in enumerate(clean_tasks):
            # Alternate perturbation types
            if i % 2 == 0:
                p_type = PerturbationType.DISTRACTING_CONTEXT
            else:
                p_type = PerturbationType.TYPO_NOISE
            
            pert_task = self.create_perturbed_task(task, p_type)
            pairs.append((task, pert_task))
        return pairs

    @staticmethod
    def compute_robustness_metrics(clean_verdicts: List[bool], perturbed_verdicts: List[bool]) -> RobustnessMetrics:
        """Calculates regression rate and performance retention under noise."""
        total = len(clean_verdicts)
        if total == 0:
            return RobustnessMetrics(0.0, 0.0, 0.0, 0, 0.0)

        clean_passed = sum(1 for v in clean_verdicts if v)
        pert_passed = sum(1 for v in perturbed_verdicts if v)

        clean_acc = (clean_passed / total) * 100.0
        pert_acc = (pert_passed / total) * 100.0
        retention_rate = (pert_acc / clean_acc * 100.0) if clean_acc > 0 else 0.0

        # Regressions: Clean was correct, but Perturbation broke it!
        regressions = sum(1 for c, p in zip(clean_verdicts, perturbed_verdicts) if c and not p)
        regression_rate = (regressions / clean_passed * 100.0) if clean_passed > 0 else 0.0

        return RobustnessMetrics(
            clean_accuracy=round(clean_acc, 1),
            perturbed_accuracy=round(pert_acc, 1),
            retention_rate=round(retention_rate, 1),
            regression_count=regressions,
            regression_rate=round(regression_rate, 1),
        )

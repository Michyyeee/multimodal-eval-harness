"""Statistical reporting engine: Bootstrap 95% Confidence Intervals & Hypothesis Testing (Standard Library Only)."""

import math
import random
from typing import List, Tuple


def bootstrap_confidence_interval(
    data: List[bool],
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float]:
    """Computes a non-parametric bootstrap confidence interval for accuracy/success rate."""
    if not data:
        return (0.0, 0.0)

    n = len(data)
    rng = random.Random(seed)
    bootstrap_means = []

    for _ in range(num_bootstraps):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_means.append(sum(1 for x in sample if x) / n * 100.0)

    bootstrap_means.sort()
    lower_idx = int((1.0 - confidence_level) / 2.0 * num_bootstraps)
    upper_idx = int((1.0 - (1.0 - confidence_level) / 2.0) * num_bootstraps)

    lower_bound = round(bootstrap_means[max(0, lower_idx)], 1)
    upper_bound = round(bootstrap_means[min(num_bootstraps - 1, upper_idx)], 1)

    return (lower_bound, upper_bound)


def calculate_mcnemar_p_value(results_a: List[bool], results_b: List[bool]) -> Tuple[float, bool]:
    """Performs McNemar's paired test with continuity correction to test if difference is statistically significant.
    
    Returns (p_value, is_significant_at_p05).
    """
    if len(results_a) != len(results_b) or len(results_a) == 0:
        return (1.0, False)

    # b: A passed, B failed
    # c: A failed, B passed
    b = sum(1 for a, b_val in zip(results_a, results_b) if a and not b_val)
    c = sum(1 for a, b_val in zip(results_a, results_b) if not a and b_val)

    if b + c == 0:
        return (1.0, False)

    # Chi-square with Edwards continuity correction
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)

    # Approximation of p-value for 1 degree of freedom: p = erfc(sqrt(chi2 / 2))
    z = math.sqrt(max(0.0, chi2))
    p_value = math.erfc(z / math.sqrt(2.0))
    p_value = round(max(0.0001, min(1.0, p_value)), 4)

    is_significant = p_value < 0.05
    return (p_value, is_significant)

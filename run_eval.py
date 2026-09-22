#!/usr/bin/env python3
"""Main runner for Multimodal Evaluation Harness (Standard Library Only)."""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

# Ensure src can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.schemas import (
    BenchmarkSummary,
    EvalTask,
    JudgeVerdict,
    ModelPrediction,
    PairwiseVerdict,
    RobustnessMetrics,
    TaskCategory,
)
from src.models.base import BaseVisionModel
from src.models.mock_model import MockVisionModel
from src.models.gemini_model import GeminiVisionModel
from src.models.registry import get_model, list_supported_providers
from src.engine import ConcurrentEvaluationRunner, RunCache, TokenBucketRateLimiter
from src.evaluators.deterministic import DeterministicEvaluator
from src.evaluators.llm_judge import LLMJudgeEvaluator, MockJudgeModel
from src.perturbations.engine import PerturbationEngine
from src.reporting.statistics import bootstrap_confidence_interval, calculate_mcnemar_p_value
from src.reporting.scorecard_generator import ScorecardGenerator


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculates percentile using standard library."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (percentile / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1


def load_benchmark(json_path: str) -> List[EvalTask]:
    """Loads benchmark dataset from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvalTask(**item) for item in data]


def run_pointwise_benchmark(
    model: BaseVisionModel,
    tasks: List[EvalTask],
    evaluator: any,
    concurrency: int = 4,
    qps_limit: float = 10.0,
    cache: Optional[RunCache] = None,
    bypass_cache: bool = False,
) -> Tuple[BenchmarkSummary, List[ModelPrediction], List[JudgeVerdict]]:
    """Runs all tasks through a given model concurrently with rate limiting and evaluates pointwise."""
    print(f"\n🚀 Running evaluation on model: {model.model_name} ({len(tasks)} tasks, concurrency={concurrency}, qps_limit={qps_limit})...")
    
    runner = ConcurrentEvaluationRunner(
        model=model,
        max_concurrency=concurrency,
        qps_limit=qps_limit,
        cache=cache,
        retry_limit=3,
        bypass_cache=bypass_cache,
    )
    predictions = runner.run_batch(tasks)
    verdicts: List[JudgeVerdict] = []

    for task, pred in zip(tasks, predictions):
        cache_indicator = " ⚡[cached]" if pred.is_cached else ""
        if pred.error:
            print(f"  ⚠️ [{task.category.value}] {task.task_id}: ERROR - {pred.error[:80]}")
            verdicts.append(JudgeVerdict(
                task_id=task.task_id,
                model_name=model.model_name,
                evaluator_name=evaluator.name,
                is_correct=False,
                score=0.0,
                reasoning=f"Execution error: {pred.error}",
            ))
            continue

        if hasattr(evaluator, "evaluate_pointwise"):
            verdict = evaluator.evaluate_pointwise(task, pred)
        else:
            verdict = evaluator.evaluate(task, pred)
        verdicts.append(verdict)

        status_icon = "✅" if verdict.is_correct else "❌"
        rubric_str = ""
        if verdict.rubric_scores:
            rubric_str = f" [VG:{verdict.rubric_scores.get('visual_grounding', 0):.0f}, FA:{verdict.rubric_scores.get('factual_accuracy', 0):.0f}]"
        cost_str = f", ${pred.cost_usd:.5f}" if pred.cost_usd is not None else ""
        print(f"  {status_icon} [{task.category.value}] {task.task_id}{rubric_str}: {verdict.reasoning[:60]}... ({pred.latency_ms:.1f}ms{cache_indicator}{cost_str})")

    # Aggregate metrics
    latencies = [p.latency_ms for p in predictions if not p.error]
    p50_lat = calculate_percentile(latencies, 50) if latencies else 0.0
    p95_lat = calculate_percentile(latencies, 95) if latencies else 0.0
    
    binary_outcomes = [v.is_correct for v in verdicts]
    passed_count = sum(1 for v in binary_outcomes if v)
    total_count = len(tasks)
    accuracy = (passed_count / total_count * 100.0) if total_count > 0 else 0.0

    # Non-parametric Bootstrap 95% Confidence Interval
    ci_95 = bootstrap_confidence_interval(binary_outcomes, num_bootstraps=1000)

    cat_scores = {}
    for cat in TaskCategory:
        cat_tasks = [v for t, v in zip(tasks, verdicts) if t.category == cat]
        if cat_tasks:
            cat_acc = sum(1 for v in cat_tasks if v.is_correct) / len(cat_tasks)
            cat_scores[cat.value] = round(cat_acc * 100, 1)

    hallucination_tasks = [v for t, v in zip(tasks, verdicts) if t.category == TaskCategory.HALLUCINATION_PROBING]
    hallucination_failures = sum(1 for v in hallucination_tasks if not v.is_correct)
    hallucination_rate = (hallucination_failures / len(hallucination_tasks)) * 100 if hallucination_tasks else 0.0

    total_input = sum(p.input_tokens or 0 for p in predictions)
    total_output = sum(p.output_tokens or 0 for p in predictions)
    total_cost = sum(p.cost_usd or 0.0 for p in predictions)

    summary = BenchmarkSummary(
        model_name=model.model_name,
        total_tasks=total_count,
        passed_tasks=passed_count,
        accuracy=round(accuracy, 1),
        ci_95_accuracy=ci_95,
        p50_latency_ms=round(p50_lat, 1),
        p95_latency_ms=round(p95_lat, 1),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cost_usd=round(total_cost, 5),
        category_breakdown=cat_scores,
        hallucination_rate=round(hallucination_rate, 1),
    )
    return summary, predictions, verdicts


def run_robustness_evaluation(
    model: BaseVisionModel,
    clean_tasks: List[EvalTask],
    evaluator: any,
    perturbation_engine: PerturbationEngine,
    concurrency: int = 4,
    qps_limit: float = 10.0,
    cache: Optional[RunCache] = None,
    bypass_cache: bool = False,
) -> Tuple[RobustnessMetrics, List[EvalTask]]:
    """Evaluates a model on paired clean and perturbed tasks to measure robustness and regressions."""
    print(f"\n🌪️ Running Adversarial Perturbation Stress Test on {model.model_name}...")
    task_pairs = perturbation_engine.generate_stress_test_suite(clean_tasks)

    runner = ConcurrentEvaluationRunner(
        model=model,
        max_concurrency=concurrency,
        qps_limit=qps_limit,
        cache=cache,
        retry_limit=3,
        bypass_cache=bypass_cache,
    )

    all_clean = [clean for clean, _ in task_pairs]
    all_pert = [pert for _, pert in task_pairs]

    preds_clean = runner.run_batch(all_clean)
    preds_pert = runner.run_batch(all_pert)

    clean_verdicts: List[bool] = []
    perturbed_verdicts: List[bool] = []

    for clean, pred_clean in zip(all_clean, preds_clean):
        v_clean = evaluator.evaluate(clean, pred_clean) if hasattr(evaluator, "evaluate") else evaluator.evaluate_pointwise(clean, pred_clean)
        clean_verdicts.append(v_clean.is_correct)

    for (clean, pert), pred_pert, is_clean_pass in zip(task_pairs, preds_pert, clean_verdicts):
        v_pert = evaluator.evaluate(pert, pred_pert) if hasattr(evaluator, "evaluate") else evaluator.evaluate_pointwise(pert, pred_pert)
        perturbed_verdicts.append(v_pert.is_correct)

        if is_clean_pass and not v_pert.is_correct:
            print(f"  💥 [REGRESSION] {clean.task_id} ({pert.perturbation_type.value}): Clean passed but noise caused failure!")

    metrics = perturbation_engine.compute_robustness_metrics(clean_verdicts, perturbed_verdicts)
    return metrics, all_pert


def run_pairwise_benchmark(
    tasks: List[EvalTask],
    preds_a: List[ModelPrediction],
    preds_b: List[ModelPrediction],
    judge: LLMJudgeEvaluator
) -> List[PairwiseVerdict]:
    """Runs SxS pairwise evaluation with bidirectional swap to detect position bias."""
    print(f"\n⚖️ Running Pairwise LLM-as-a-Judge with Position Bias Mitigation ({len(tasks)} comparisons)...")
    verdicts: List[PairwiseVerdict] = []
    
    for task, pa, pb in zip(tasks, preds_a, preds_b):
        verdict = judge.evaluate_pairwise(task, pa, pb)
        verdicts.append(verdict)

        if verdict.has_position_bias:
            icon = "⚠️ [BIAS]"
        elif verdict.winner == "model_a":
            icon = f"🏆 [{pa.model_name}]"
        elif verdict.winner == "model_b":
            icon = f"🏆 [{pb.model_name}]"
        else:
            icon = "🤝 [TIE]"

        print(f"  {icon} {task.task_id}: {verdict.reasoning[:70]}")

    return verdicts


def print_executive_scorecard(summaries: List[BenchmarkSummary], eval_name: str, robustness_map: Dict[str, RobustnessMetrics] = None):
    """Prints a comparison table of evaluated models."""
    print("\n" + "=" * 116)
    print(f"                                MULTIMODAL EVALUATION SCORECARD [{eval_name.upper()}]")
    print("=" * 116)
    header = f"{'Model Name':<24} | {'Acc (%)':<8} | {'95% Bootstrap CI':<18} | {'p50 Lat':<10} | {'p95 Lat':<10} | {'Tokens (In/Out)':<16} | {'Est. Cost':<10} | {'Hallucination'}"
    print(header)
    print("-" * 116)
    for s in summaries:
        ci_str = f"[{s.ci_95_accuracy[0]:.1f}% - {s.ci_95_accuracy[1]:.1f}%]"
        tok_str = f"{s.total_input_tokens}/{s.total_output_tokens}"
        cost_str = f"${s.total_cost_usd:.4f}"
        row = f"{s.model_name:<24} | {s.accuracy:<8.1f} | {ci_str:<18} | {s.p50_latency_ms:<7.1f}ms | {s.p95_latency_ms:<7.1f}ms | {tok_str:<16} | {cost_str:<10} | {s.hallucination_rate:<13.1f}%"
        print(row)
    print("=" * 116)

    if robustness_map:
        print("\n🌪️ Adversarial Robustness & Regression Testing:")
        r_header = f"{'Model Name':<24} | {'Clean Acc':<10} | {'Noise Acc':<10} | {'Retention Rate':<16} | {'Regression Rate'}"
        print(r_header)
        print("-" * 116)
        for m_name, rob in robustness_map.items():
            r_row = f"{m_name:<24} | {rob.clean_accuracy:<8.1f}% | {rob.perturbed_accuracy:<8.1f}% | {rob.retention_rate:<14.1f}% | {rob.regression_rate:.1f}% ({rob.regression_count} tasks broke)"
            print(r_row)
        print("=" * 116)


def main():
    parser = argparse.ArgumentParser(description="Run Multimodal Evaluation Harness (Standard Library Only)")
    parser.add_argument("--eval-mode", choices=["deterministic", "llm-judge", "pairwise"], default="pairwise",
                        help="Evaluation mode: deterministic, llm-judge (rubric), or pairwise (SxS with position bias mitigation)")
    parser.add_argument("--model", type=str, default=None,
                        help="Primary model to evaluate (e.g. 'gemini:gemini-1.5-flash', 'openai:gpt-4o', 'anthropic:claude-3-5-sonnet-20241022', or 'mock')")
    parser.add_argument("--model-b", type=str, default=None,
                        help="Second model for pairwise comparison (defaults to Gemma-3B-Quantized mock)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Concurrent evaluation worker threads (default: 4)")
    parser.add_argument("--qps-limit", type=float, default=10.0,
                        help="Token bucket QPS limit across worker pool (default: 10.0)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass response cache (essential for non-deterministic sampling / pass@k)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear the persistent response cache file before evaluation")
    parser.add_argument("--cache-file", type=str, default=".eval_cache.jsonl",
                        help="Path to JSONL response cache file (default: .eval_cache.jsonl)")
    parser.add_argument("--run-robustness", action="store_true", default=True,
                        help="Run adversarial perturbation stress-testing to measure regression rate")
    parser.add_argument("--generate-report", action="store_true", default=True,
                        help="Generate Markdown and HTML scorecard reports in reports/")
    parser.add_argument("--use-gemini", action="store_true",
                        help="Run live evaluation against Gemini API (alias for --model gemini:gemini-1.5-flash)")
    parser.add_argument("--gemini-model", type=str, default="gemini-1.5-flash",
                        help="Gemini model name (default: gemini-1.5-flash)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API Key for live provider (or use GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_path = os.path.join(repo_dir, "data", "sample_benchmark.json")
    tasks = load_benchmark(benchmark_path)

    # Cache setup
    cache_path = os.path.join(repo_dir, args.cache_file)
    cache = RunCache(cache_path=cache_path)
    if args.clear_cache:
        cache.clear()
        print(f"🧹 Cleared response cache at {cache_path}")

    # Model resolution
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if args.model:
        model_a = get_model(args.model, api_key=args.api_key)
    elif args.use_gemini and api_key:
        model_a = get_model(f"gemini:{args.gemini_model}", api_key=api_key)
    else:
        model_a = MockVisionModel(model_name="Gemini-Flash-Baseline", accuracy_rate=0.88, base_latency_ms=135.0)

    if args.model_b:
        model_b = get_model(args.model_b, api_key=args.api_key)
    else:
        model_b = MockVisionModel(model_name="Gemma-3B-Quantized", accuracy_rate=0.74, base_latency_ms=45.0)

    # Configure Evaluator / Judge
    if args.use_gemini and api_key:
        judge_backend = GeminiVisionModel(model_name=args.gemini_model, api_key=api_key)
        judge_evaluator = LLMJudgeEvaluator(judge_model=judge_backend, name="live_gemini_judge")
    else:
        judge_backend = MockJudgeModel()
        judge_evaluator = LLMJudgeEvaluator(judge_model=judge_backend, name="mock_llm_judge")

    deterministic_evaluator = DeterministicEvaluator()
    perturbation_engine = PerturbationEngine(seed=42)

    robustness_map = {}
    if args.run_robustness:
        rob_a, _ = run_robustness_evaluation(
            model=model_a,
            clean_tasks=tasks,
            evaluator=deterministic_evaluator,
            perturbation_engine=perturbation_engine,
            concurrency=args.concurrency,
            qps_limit=args.qps_limit,
            cache=cache,
            bypass_cache=args.no_cache,
        )
        rob_b, _ = run_robustness_evaluation(
            model=model_b,
            clean_tasks=tasks,
            evaluator=deterministic_evaluator,
            perturbation_engine=perturbation_engine,
            concurrency=args.concurrency,
            qps_limit=args.qps_limit,
            cache=cache,
            bypass_cache=args.no_cache,
        )
        robustness_map[model_a.model_name] = rob_a
        robustness_map[model_b.model_name] = rob_b

    pairwise_verdicts = None

    if args.eval_mode == "deterministic":
        summaries = [
            run_pointwise_benchmark(
                model=model_a,
                tasks=tasks,
                evaluator=deterministic_evaluator,
                concurrency=args.concurrency,
                qps_limit=args.qps_limit,
                cache=cache,
                bypass_cache=args.no_cache,
            )[0],
            run_pointwise_benchmark(
                model=model_b,
                tasks=tasks,
                evaluator=deterministic_evaluator,
                concurrency=args.concurrency,
                qps_limit=args.qps_limit,
                cache=cache,
                bypass_cache=args.no_cache,
            )[0],
        ]
        print_executive_scorecard(summaries, eval_name="Deterministic", robustness_map=robustness_map)

    elif args.eval_mode == "llm-judge":
        summaries = [
            run_pointwise_benchmark(
                model=model_a,
                tasks=tasks,
                evaluator=judge_evaluator,
                concurrency=args.concurrency,
                qps_limit=args.qps_limit,
                cache=cache,
                bypass_cache=args.no_cache,
            )[0],
            run_pointwise_benchmark(
                model=model_b,
                tasks=tasks,
                evaluator=judge_evaluator,
                concurrency=args.concurrency,
                qps_limit=args.qps_limit,
                cache=cache,
                bypass_cache=args.no_cache,
            )[0],
        ]
        print_executive_scorecard(summaries, eval_name="LLM-as-a-Judge (Rubric)", robustness_map=robustness_map)

    elif args.eval_mode == "pairwise":
        summary_a, preds_a, verdicts_a = run_pointwise_benchmark(
            model=model_a,
            tasks=tasks,
            evaluator=deterministic_evaluator,
            concurrency=args.concurrency,
            qps_limit=args.qps_limit,
            cache=cache,
            bypass_cache=args.no_cache,
        )
        summary_b, preds_b, verdicts_b = run_pointwise_benchmark(
            model=model_b,
            tasks=tasks,
            evaluator=deterministic_evaluator,
            concurrency=args.concurrency,
            qps_limit=args.qps_limit,
            cache=cache,
            bypass_cache=args.no_cache,
        )
        pairwise_verdicts = run_pairwise_benchmark(tasks, preds_a, preds_b, judge_evaluator)
        summaries = [summary_a, summary_b]

        print_executive_scorecard(summaries, eval_name="Baseline vs. Edge Quantized", robustness_map=robustness_map)

        # Calculate McNemar paired statistical significance
        p_val, is_sig = calculate_mcnemar_p_value([v.is_correct for v in verdicts_a], [v.is_correct for v in verdicts_b])
        print(f"📊 Paired Statistical Significance (McNemar Test): p-value = {p_val:.4f} ({'Statistically Significant' if is_sig else 'Not Significant at α=0.05'})")

    if args.generate_report:
        reports_dir = os.path.join(repo_dir, "reports")
        md_file = os.path.join(reports_dir, "scorecard.md")
        html_file = os.path.join(reports_dir, "scorecard.html")

        ScorecardGenerator.generate_markdown(summaries, pairwise_verdicts, robustness_map, output_path=md_file)
        ScorecardGenerator.generate_html(summaries, pairwise_verdicts, robustness_map, output_path=html_file)
        print(f"\n📑 Generated evaluation reports:")
        print(f"   • Markdown: {md_file}")
        print(f"   • HTML:     {html_file}\n")


if __name__ == "__main__":
    main()

"""Generates executive-ready Markdown and HTML evaluation scorecards."""

import os
from typing import List, Optional
from src.schemas import BenchmarkSummary, PairwiseVerdict, RobustnessMetrics


class ScorecardGenerator:
    """Renders formatted Markdown and HTML report artifacts for launch decisions."""

    @staticmethod
    def generate_markdown(
        summaries: List[BenchmarkSummary],
        pairwise_verdicts: Optional[List[PairwiseVerdict]] = None,
        robustness_map: Optional[dict] = None,
        output_path: str = "reports/scorecard.md"
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        lines = []

        lines.append("# Multimodal Model Evaluation Scorecard")
        lines.append("\nAn automated benchmark report measuring model accuracy, latency, visual hallucination, and adversarial robustness.\n")

        # Table 1: Core Performance & Confidence Intervals
        lines.append("## 1. Overall Performance & 95% Confidence Intervals")
        lines.append("| Model Name | Accuracy (%) | 95% Bootstrap CI | p50 Latency (ms) | p95 Latency (ms) | Tokens (In/Out) | Est. Cost ($) | Hallucination Rate (%) |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for s in summaries:
            ci_str = f"[{s.ci_95_accuracy[0]:.1f}% – {s.ci_95_accuracy[1]:.1f}%]"
            tok_str = f"{s.total_input_tokens} / {s.total_output_tokens}"
            cost_str = f"${s.total_cost_usd:.4f}"
            lines.append(f"| **{s.model_name}** | {s.accuracy:.1f}% | {ci_str} | {s.p50_latency_ms:.1f}ms | {s.p95_latency_ms:.1f}ms | {tok_str} | {cost_str} | {s.hallucination_rate:.1f}% |")

        # Table 2: Robustness & Regression Testing
        if robustness_map:
            lines.append("\n## 2. Adversarial Stress Testing & Robustness")
            lines.append("| Model Name | Clean Accuracy | Perturbed Accuracy | Retention Rate | Regressions (Broke) | Regression Rate |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
            for model_name, rob in robustness_map.items():
                lines.append(f"| **{model_name}** | {rob.clean_accuracy:.1f}% | {rob.perturbed_accuracy:.1f}% | {rob.retention_rate:.1f}% | {rob.regression_count} tasks | {rob.regression_rate:.1f}% |")

        # Table 3: Pairwise SxS Comparison
        if pairwise_verdicts:
            total = len(pairwise_verdicts)
            m_a = pairwise_verdicts[0].model_a_name
            m_b = pairwise_verdicts[0].model_b_name
            a_wins = sum(1 for v in pairwise_verdicts if v.winner == "model_a")
            b_wins = sum(1 for v in pairwise_verdicts if v.winner == "model_b")
            ties = sum(1 for v in pairwise_verdicts if v.winner == "tie")
            bias_inconclusive = sum(1 for v in pairwise_verdicts if v.has_position_bias)

            lines.append("\n## 3. Pairwise LLM-as-a-Judge (Side-by-Side with Position Bias Mitigation)")
            lines.append(f"* **Total Evaluations**: {total}")
            lines.append(f"* **{m_a} Wins**: {a_wins} ({a_wins/total*100:.1f}%)")
            lines.append(f"* **{m_b} Wins**: {b_wins} ({b_wins/total*100:.1f}%)")
            lines.append(f"* **Ties**: {ties} ({ties/total*100:.1f}%)")
            lines.append(f"* **Position Bias Inconclusive (Order Inconsistency)**: {bias_inconclusive} ({bias_inconclusive/total*100:.1f}%)")

        content = "\n".join(lines) + "\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        return output_path

    @staticmethod
    def generate_html(
        summaries: List[BenchmarkSummary],
        pairwise_verdicts: Optional[List[PairwiseVerdict]] = None,
        robustness_map: Optional[dict] = None,
        output_path: str = "reports/scorecard.html"
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        rows_perf = ""
        for s in summaries:
            ci = f"[{s.ci_95_accuracy[0]:.1f}% – {s.ci_95_accuracy[1]:.1f}%]"
            rows_perf += f"""
            <tr>
                <td><strong>{s.model_name}</strong></td>
                <td><span class="badge badge-primary">{s.accuracy:.1f}%</span></td>
                <td><code>{ci}</code></td>
                <td>{s.p50_latency_ms:.1f} ms</td>
                <td>{s.p95_latency_ms:.1f} ms</td>
                <td>{s.total_input_tokens} / {s.total_output_tokens}</td>
                <td>${s.total_cost_usd:.4f}</td>
                <td><span class="badge badge-warning">{s.hallucination_rate:.1f}%</span></td>
            </tr>
            """

        rows_rob = ""
        if robustness_map:
            for m_name, rob in robustness_map.items():
                rows_rob += f"""
                <tr>
                    <td><strong>{m_name}</strong></td>
                    <td>{rob.clean_accuracy:.1f}%</td>
                    <td>{rob.perturbed_accuracy:.1f}%</td>
                    <td><strong>{rob.retention_rate:.1f}%</strong></td>
                    <td>{rob.regression_count}</td>
                    <td><span class="badge badge-danger">{rob.regression_rate:.1f}%</span></td>
                </tr>
                """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Multimodal Model Evaluation Scorecard</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 40px; background: #f8fafc; color: #1e293b; }}
        .card {{ background: #ffffff; padding: 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
        h1 {{ color: #0f172a; margin-top: 0; }}
        h2 {{ color: #334155; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
        th {{ background: #f1f5f9; color: #475569; font-weight: 600; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; display: inline-block; }}
        .badge-primary {{ background: #e0f2fe; color: #0369a1; }}
        .badge-warning {{ background: #fef3c7; color: #b45309; }}
        .badge-danger {{ background: #fee2e2; color: #b91c1c; }}
        .callout {{ background: #f0fdf4; border-left: 4px solid #16a34a; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Multimodal Model Evaluation Scorecard</h1>
        <p>Automated launch verification report with bootstrap 95% confidence intervals and adversarial stress testing.</p>
        
        <h2>1. Overall Performance & 95% Confidence Intervals</h2>
        <table>
            <thead>
                <tr>
                    <th>Model Name</th>
                    <th>Accuracy (%)</th>
                    <th>95% Bootstrap CI</th>
                    <th>p50 Latency</th>
                    <th>p95 Latency</th>
                    <th>Tokens (In/Out)</th>
                    <th>Est. Cost</th>
                    <th>Hallucination Rate</th>
                </tr>
            </thead>
            <tbody>
                {rows_perf}
            </tbody>
        </table>

        {"<h2>2. Adversarial Stress Testing (Robustness)</h2><table><thead><tr><th>Model Name</th><th>Clean Acc</th><th>Perturbed Acc</th><th>Retention</th><th>Regressions</th><th>Regression Rate</th></tr></thead><tbody>" + rows_rob + "</tbody></table>" if robustness_map else ""}
    </div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path

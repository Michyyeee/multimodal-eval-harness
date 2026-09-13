# Multimodal Evaluation Harness (`multimodal-eval-harness`)

An extensible, production-grade benchmarking and evaluation framework for Vision-Language Models (VLMs). Evaluates model accuracy, latency, and reliability across visual reasoning tasks with a focus on **visual hallucination detection**, **automated LLM-as-a-Judge grading with position-bias mitigation**, and **adversarial stress testing with statistical confidence intervals**.

---

## Architecture Overview

```
multimodal_eval_harness/
├── src/
│   ├── schemas.py                 # Dataclasses: EvalTask, ModelPrediction, JudgeVerdict, RobustnessMetrics
│   ├── models/
│   │   ├── base.py                # Abstract BaseVisionModel interface
│   │   ├── mock_model.py          # Fast hermetic simulation for CI & local development
│   │   └── gemini_model.py        # Live Gemini Vision & Judge REST client (zero external dependencies)
│   ├── evaluators/
│   │   ├── deterministic.py       # Heuristic, regex, numeric-tolerance, and hallucination logic
│   │   └── llm_judge.py           # LLM-as-a-Judge: 1-5 rubrics + SxS position-bias mitigation
│   ├── perturbations/
│   │   └── engine.py              # Adversarial engine: distracting context, typo noise, regression tracking
│   └── reporting/
│       ├── statistics.py          # Bootstrap 95% Confidence Intervals & McNemar paired significance test
│       └── scorecard_generator.py # Automated executive Markdown and HTML report generation
├── data/
│   ├── sample_benchmark.json      # Curated multi-category visual reasoning test set
│   └── images/                    # Generated sample PNG benchmark images
├── reports/                       # Generated evaluation scorecards
│   ├── scorecard.md               # GitHub Markdown summary report
│   └── scorecard.html             # Standalone executive HTML report
├── scripts/
│   └── generate_sample_images.py  # Pure stdlib PNG generator for test visual assets
└── run_eval.py                    # Main evaluation engine and executive scorecard CLI
```

---

## Key Features

### 1. Multi-Modal Task Categories
* **Chart Understanding**: Extracts numerical data within configurable error bounds (e.g. $\pm 5\%$).
* **Document OCR**: Validates key-value extraction, totals, and currency.
* **Spatial Reasoning**: Tests directional and geometric scene relationships.
* **Fine-Grained Perception**: Small object counting and state verification.
* **Hallucination Probing**: Adversarial false-premise queries verifying whether the model correctly refutes non-existent visual entities.

### 2. Automated LLM-as-a-Judge with Position-Bias Mitigation
* **Pointwise Mode**: Grades predictions on a 1–5 scale across **Visual Grounding** (penalizing hallucinations), **Factual Accuracy**, and **Conciseness**.
* **Pairwise SxS Mode**: Compares two models using **bidirectional order swapping** `(A, B)` and `(B, A)` to eliminate and report position bias where presentation order influences the outcome.

### 3. Adversarial Stress-Testing & Regression Tracking
* Injects real-world **distracting context** and **keyboard typo noise**.
* Measures **Retention Rate** (performance preserved under noise) and **Regression Rate** (frequency where clean input passed but noise triggered failure).

### 4. Statistical Rigor (Non-Parametric CIs & Hypothesis Testing)
* **Bootstrap 95% Confidence Intervals**: Runs 1,000 resamples with replacement to provide empirical uncertainty bounds (`[42.9% – 100.0%]`).
* **McNemar's Paired Test**: Computes two-tailed p-values with continuity correction to verify whether performance deltas between competing models are statistically significant ($\alpha = 0.05$) or random noise.

### 5. Automated Executive Reports
* Automatically exports [`reports/scorecard.md`](reports/scorecard.md) and styled standalone [`reports/scorecard.html`](reports/scorecard.html).

---

## Quickstart & CLI Usage

### 1. Run Full Benchmark (Default)
Executes the end-to-end evaluation pipeline out of the box with zero external dependencies (fast hermetic simulation, 1,000-sample Bootstrap CIs, McNemar significance test, adversarial perturbations, and scorecard generation):
```bash
python3 run_eval.py
```

### 2. Evaluation Modes
```bash
# Deterministic Only (Exact match, regex, and numerical tolerance - fast CI check)
python3 run_eval.py --eval-mode deterministic

# Pointwise LLM-as-a-Judge (1–5 rubric scoring: Visual Grounding, Accuracy, Conciseness)
python3 run_eval.py --eval-mode llm-judge

# Pairwise SxS Mode (Bidirectional order swapping with position-bias mitigation & McNemar test)
python3 run_eval.py --eval-mode pairwise
```

### 3. Adversarial Robustness & Stress-Testing
Injects distracting context and keyboard typo noise to measure **Retention Rate** and **Regression Rate**:
```bash
# Run deterministic evaluation with adversarial robustness tracking
python3 run_eval.py --eval-mode deterministic --run-robustness

# Full pairwise SxS with adversarial stress-testing
python3 run_eval.py --eval-mode pairwise --run-robustness
```

### 4. Run Live Gemini Vision API
Evaluate against live Google Gemini models using standard REST API (no external pip dependencies required):
```bash
export GEMINI_API_KEY="your-api-key-here"
python3 run_eval.py --use-gemini --gemini-model gemini-1.5-flash
```

### 5. Inspect Executive Reports & Dashboard
View generated summary reports written to `reports/`:
```bash
# Inspect Markdown summary in terminal
cat reports/scorecard.md

# Open standalone styled executive HTML dashboard
xdg-open reports/scorecard.html  # On Linux
# open reports/scorecard.html    # On macOS
```

---

## CLI Options Reference

| Flag | Options / Default | Description |
|---|---|---|
| `--eval-mode` | `pairwise` (default), `llm-judge`, `deterministic` | Primary evaluation mode. `pairwise` runs SxS with position-bias mitigation & McNemar test. |
| `--run-robustness` | `True` (default) | Executes adversarial perturbation suite (context injection + typo noise). |
| `--generate-report` | `True` (default) | Automatically exports `reports/scorecard.md` and `reports/scorecard.html`. |
| `--use-gemini` | `False` (default) | Routes inference/judging to live Google Gemini REST API. |
| `--gemini-model` | `gemini-1.5-flash` | Target Gemini model version for live calls. |
| `--api-key` | `None` (falls back to `$GEMINI_API_KEY`) | Google Gemini API key. |

---

## Sample Scorecard Output

```text
================================================================================================
             MULTIMODAL EVALUATION SCORECARD [BASELINE VS. EDGE QUANTIZED]
================================================================================================
Model Name               | Acc (%)  | 95% Bootstrap CI   | p50 Lat    | p95 Lat    | Hallucination
------------------------------------------------------------------------------------------------
Gemini-Flash-Baseline    | 100.0    | [100.0% - 100.0%]  | 153.3  ms | 164.4  ms | 0.0          %
Gemma-3B-Quantized       | 71.4     | [42.9% - 100.0%]   | 43.2   ms | 84.6   ms | 0.0          %
================================================================================================

🌪️ Adversarial Robustness & Regression Testing:
Model Name               | Clean Acc  | Noise Acc  | Retention Rate   | Regression Rate
------------------------------------------------------------------------------------------------
Gemini-Flash-Baseline    | 100.0   % | 85.7    % | 85.7          % | 14.3% (1 task broke)
Gemma-3B-Quantized       | 57.1    % | 71.4    % | 125.0         % | 50.0% (2 tasks broke)
================================================================================================
📊 Paired Statistical Significance (McNemar Test): p-value = 0.4795 (Not Significant at α=0.05)

📑 Generated evaluation reports:
   • Markdown: reports/scorecard.md
   • HTML:     reports/scorecard.html
```


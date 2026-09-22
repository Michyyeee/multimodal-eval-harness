# Multimodal Evaluation Harness (`multimodal-eval-harness`)

An extensible, production-grade benchmarking and evaluation framework for Vision-Language Models (VLMs). Evaluates model accuracy, latency, per-token cost, and reliability across visual reasoning tasks with a focus on **multi-provider competitor benchmarking**, **high-throughput rate-limited concurrency**, **visual hallucination detection**, **automated LLM-as-a-Judge grading with position-bias mitigation**, and **adversarial stress testing with statistical confidence intervals**.

> **Design Principle**: Strict **Zero External Dependencies** — built entirely on the Python Standard Library (`urllib.request`, `concurrent.futures`, `threading`, `dataclasses`, `json`, `base64`, `hashlib`, `unittest`).

---

## Architecture Overview

```
multimodal_eval_harness/
├── src/
│   ├── schemas.py                 # Dataclasses: EvalTask, ModelPrediction, JudgeVerdict, RobustnessMetrics, BenchmarkSummary
│   ├── engine/                    # High-throughput asynchronous concurrency & caching engine
│   │   ├── rate_limiter.py        # Thread-safe Token Bucket QPS rate-limiter
│   │   ├── cache.py               # JSONL persistent response cache with pass@k bypass semantics
│   │   └── runner.py              # ConcurrentEvaluationRunner with exponential backoff & jitter
│   ├── models/                    # Multi-provider competitor adapter layer
│   │   ├── base.py                # Abstract BaseVisionModel interface
│   │   ├── registry.py            # Unified model factory (Gemini, OpenAI, Anthropic, Mock)
│   │   ├── utils.py               # Pure stdlib Base64 image loader & per-token cost calculator
│   │   ├── mock_model.py          # Fast hermetic simulation for CI & local development
│   │   ├── gemini_model.py        # Live Google Gemini REST client (stdlib urllib)
│   │   ├── openai_model.py        # Live OpenAI GPT-4o / o1 / o3 REST client (stdlib urllib)
│   │   └── anthropic_model.py     # Live Anthropic Claude 3.5 Sonnet / Haiku REST client (stdlib urllib)
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
│   └── images/                    # Benchmark visual assets (PNGs)
├── tests/
│   └── test_harness.py            # Hermetic unit test suite (11 test cases across engine, registry, cost)
├── reports/                       # Generated evaluation scorecards
│   ├── scorecard.md               # GitHub Markdown summary report
│   └── scorecard.html             # Standalone executive HTML report
├── scripts/
│   └── generate_sample_images.py  # Pure stdlib PNG generator for test visual assets
└── run_eval.py                    # Main evaluation engine and executive scorecard CLI
```

---

## Core Capabilities

### 1. Multi-Provider Competitor Adapter Layer
* **Normalized Interface**: Common `predict(task: EvalTask) -> ModelPrediction` contract across disparate vendor schemas.
* **Pure Standard Library Clients**: Native HTTP REST clients for **Google Gemini**, **OpenAI (GPT-4o, o1, o3)**, and **Anthropic (Claude 3.5 Sonnet, Haiku)** using `urllib.request` — no heavy SDKs (`google-genai`, `openai`, `anthropic`, `requests`) required.
* **Model Registry & Factory**: Dynamic resolution with provider prefixes (`"gemini:gemini-1.5-flash"`, `"openai:gpt-4o"`, `"anthropic:claude-3-5-sonnet-20241022"`, or `"mock"`).
* **Token Tracking & Cost Estimation**: Automatically extracts prompt and completion token counts from API response metadata and computes run costs in USD based on published vendor pricing tables.

### 2. High-Throughput Concurrency & Rate-Limiting Engine
* **Token-Bucket Rate Limiter**: Thread-safe leaky token bucket algorithm that enforces a global Queries-Per-Second (QPS) cap across concurrent worker threads to protect against vendor rate limits.
* **Resilient Retry Loop with Exponential Backoff & Jitter**: Automatically catches transient HTTP 429 (Too Many Requests), 500, 502, 503, and 504 errors, retrying up to $N$ times with randomized exponential backoff ($2^{\text{attempt}} + \text{jitter}$).
* **Persistent Response Cache & Checkpointing**: Thread-safe disk-backed JSONL cache (`.eval_cache.jsonl`) indexed by SHA-256 task content hashes. Enables instant crash-recovery and eliminates redundant API billing.
* **Explicit `bypass_cache` for `pass@k` Sampling**: Dedicated flag (`--no-cache`) ensures temperature-sampled / non-deterministic rollouts are always evaluated independently without polluting or pulling from cached runs.

### 3. Multi-Modal Task Categories
* **Chart Understanding**: Extracts numerical data within configurable error bounds (e.g. $\pm 5\%$).
* **Document OCR**: Validates key-value extraction, totals, and currency.
* **Spatial Reasoning**: Tests directional and geometric scene relationships.
* **Fine-Grained Perception**: Small object counting and state verification.
* **Hallucination Probing**: Adversarial false-premise queries verifying whether the model correctly refutes non-existent visual entities.

### 4. Automated LLM-as-a-Judge with Position-Bias Mitigation
* **Pointwise Mode**: Grades predictions on a 1–5 scale across **Visual Grounding** (penalizing hallucinations), **Factual Accuracy**, and **Conciseness**.
* **Pairwise SxS Mode**: Compares two models using **bidirectional order swapping** `(A, B)` and `(B, A)` to detect and flag position bias where presentation order influences the judge's verdict.

### 5. Adversarial Stress-Testing & Regression Tracking
* Injects real-world **distracting context** and **keyboard typo noise**.
* Measures **Retention Rate** (performance preserved under noise) and **Regression Rate** (frequency where clean input passed but noise triggered failure).

### 6. Statistical Rigor (Non-Parametric CIs & Hypothesis Testing)
* **Bootstrap 95% Confidence Intervals**: Runs 1,000 resamples with replacement to provide empirical uncertainty bounds (`[57.1% – 100.0%]`).
* **McNemar's Paired Test**: Computes two-tailed p-values with continuity correction to verify whether performance deltas between competing models are statistically significant ($\alpha = 0.05$) or random noise.

---

## Quickstart & CLI Usage

### 1. Run Hermetic Benchmark (Default)
Executes the end-to-end evaluation pipeline out of the box with zero external dependencies (fast simulation, multi-threaded execution, rate-limiting, 1,000-sample Bootstrap CIs, McNemar significance test, adversarial perturbations, and scorecard generation):
```bash
python3 run_eval.py
```

### 2. High-Throughput Concurrent Execution & Rate Limiting
```bash
# Run with 8 worker threads capped at 20 QPS
python3 run_eval.py --concurrency 8 --qps-limit 20.0

# Clear cache and run fresh evaluation
python3 run_eval.py --clear-cache --concurrency 4

# Bypass cache completely (mandatory for non-deterministic pass@k evaluations)
python3 run_eval.py --no-cache --concurrency 4
```

### 3. Multi-Provider Competitor Evaluations
Evaluate different models using the unified model registry:
```bash
# 1. Google Gemini
export GEMINI_API_KEY="your-gemini-key"
python3 run_eval.py --model "gemini:gemini-1.5-flash" --eval-mode deterministic

# 2. OpenAI GPT-4o
export OPENAI_API_KEY="your-openai-key"
python3 run_eval.py --model "openai:gpt-4o" --concurrency 4 --qps-limit 5.0

# 3. Anthropic Claude 3.5 Sonnet
export ANTHROPIC_API_KEY="your-anthropic-key"
python3 run_eval.py --model "anthropic:claude-3-5-sonnet-20241022" --concurrency 2

# 4. Pairwise Competitor Head-to-Head (Gemini vs. GPT-4o)
python3 run_eval.py \
  --model "gemini:gemini-1.5-pro" \
  --model-b "openai:gpt-4o" \
  --eval-mode pairwise \
  --concurrency 4
```

### 4. Evaluation Modes
```bash
# Deterministic Only (Exact match, regex, and numerical tolerance - fast CI check)
python3 run_eval.py --eval-mode deterministic

# Pointwise LLM-as-a-Judge (1–5 rubric scoring: Visual Grounding, Accuracy, Conciseness)
python3 run_eval.py --eval-mode llm-judge

# Pairwise SxS Mode (Bidirectional order swapping with position-bias mitigation & McNemar test)
python3 run_eval.py --eval-mode pairwise
```

### 5. Running the Unit Test Suite
Verify all components hermetically with Python's built-in `unittest`:
```bash
python3 -m unittest discover -s tests
```

---

## CLI Options Reference

| Flag | Default | Description |
|---|---|---|
| `--model` | `None` (mock baseline) | Primary model identifier (`gemini:...`, `openai:...`, `anthropic:...`, or `mock`). |
| `--model-b` | `None` (mock edge) | Competitor model for pairwise SxS comparison. |
| `--eval-mode` | `pairwise` | Evaluation mode: `deterministic`, `llm-judge`, or `pairwise`. |
| `--concurrency` | `4` | Number of parallel worker threads in the execution pool. |
| `--qps-limit` | `10.0` | Token bucket queries-per-second limit across workers. |
| `--no-cache` | `False` | Bypasses the response cache (essential for `pass@k` sampling). |
| `--clear-cache` | `False` | Deletes `.eval_cache.jsonl` before starting the benchmark. |
| `--cache-file` | `.eval_cache.jsonl` | Path to the persistent JSONL response cache file. |
| `--run-robustness` | `True` | Executes adversarial perturbation suite (context injection + typo noise). |
| `--generate-report`| `True` | Automatically exports `reports/scorecard.md` and `reports/scorecard.html`. |
| `--api-key` | `None` | API key override (defaults to environment variables). |

---

## Sample Scorecard Output

```text
====================================================================================================================
                                MULTIMODAL EVALUATION SCORECARD [BASELINE VS. EDGE QUANTIZED]
====================================================================================================================
Model Name               | Acc (%)  | 95% Bootstrap CI   | p50 Lat    | p95 Lat    | Tokens (In/Out)  | Est. Cost  | Hallucination
--------------------------------------------------------------------------------------------------------------------
Gemini-Flash-Baseline    | 85.7     | [57.1% - 100.0%]   | 149.3  ms | 169.4  ms | 1792/27          | $0.0000    | 50.0         %
Gemma-3B-Quantized       | 57.1     | [14.3% - 85.7%]    | 63.0   ms | 88.2   ms | 1792/44          | $0.0000    | 0.0          %
====================================================================================================================

🌪️ Adversarial Robustness & Regression Testing:
Model Name               | Clean Acc  | Noise Acc  | Retention Rate   | Regression Rate
--------------------------------------------------------------------------------------------------------------------
Gemini-Flash-Baseline    | 85.7    % | 100.0   % | 116.7         % | 0.0% (0 tasks broke)
Gemma-3B-Quantized       | 57.1    % | 57.1    % | 100.0         % | 25.0% (1 tasks broke)
====================================================================================================================
📊 Paired Statistical Significance (McNemar Test): p-value = 0.6171 (Not Significant at α=0.05)

📑 Generated evaluation reports:
   • Markdown: reports/scorecard.md
   • HTML:     reports/scorecard.html
```

# Multimodal Model Evaluation Scorecard

An automated benchmark report measuring model accuracy, latency, visual hallucination, and adversarial robustness.

## 1. Overall Performance & 95% Confidence Intervals
| Model Name | Accuracy (%) | 95% Bootstrap CI | p50 Latency (ms) | p95 Latency (ms) | Tokens (In/Out) | Est. Cost ($) | Hallucination Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemini-Flash-Baseline** | 85.7% | [57.1% – 100.0%] | 128.7ms | 175.5ms | 1792 / 29 | $0.0000 | 0.0% |
| **Gemma-3B-Quantized** | 85.7% | [57.1% – 100.0%] | 50.3ms | 76.7ms | 1792 / 29 | $0.0000 | 0.0% |

## 2. Adversarial Stress Testing & Robustness
| Model Name | Clean Accuracy | Perturbed Accuracy | Retention Rate | Regressions (Broke) | Regression Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini-Flash-Baseline** | 85.7% | 100.0% | 116.7% | 0 tasks | 0.0% |
| **Gemma-3B-Quantized** | 85.7% | 85.7% | 100.0% | 1 tasks | 16.7% |

## 3. Pairwise LLM-as-a-Judge (Side-by-Side with Position Bias Mitigation)
* **Total Evaluations**: 7
* **Gemini-Flash-Baseline Wins**: 0 (0.0%)
* **Gemma-3B-Quantized Wins**: 0 (0.0%)
* **Ties**: 0 (0.0%)
* **Position Bias Inconclusive (Order Inconsistency)**: 7 (100.0%)

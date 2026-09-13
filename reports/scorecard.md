# Multimodal Model Evaluation Scorecard

An automated benchmark report measuring model accuracy, latency, visual hallucination, and adversarial robustness.

## 1. Overall Performance & 95% Confidence Intervals
| Model Name | Accuracy (%) | 95% Bootstrap CI | p50 Latency (ms) | p95 Latency (ms) | Hallucination Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini-Flash-Baseline** | 85.7% | [57.1% – 100.0%] | 163.2ms | 175.4ms | 0.0% |
| **Gemma-3B-Quantized** | 57.1% | [14.3% – 85.7%] | 59.6ms | 74.4ms | 0.0% |

## 2. Adversarial Stress Testing & Robustness
| Model Name | Clean Accuracy | Perturbed Accuracy | Retention Rate | Regressions (Broke) | Regression Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini-Flash-Baseline** | 100.0% | 85.7% | 85.7% | 1 tasks | 14.3% |
| **Gemma-3B-Quantized** | 85.7% | 57.1% | 66.7% | 2 tasks | 33.3% |

## 3. Pairwise LLM-as-a-Judge (Side-by-Side with Position Bias Mitigation)
* **Total Evaluations**: 7
* **Gemini-Flash-Baseline Wins**: 0 (0.0%)
* **Gemma-3B-Quantized Wins**: 0 (0.0%)
* **Ties**: 0 (0.0%)
* **Position Bias Inconclusive (Order Inconsistency)**: 7 (100.0%)

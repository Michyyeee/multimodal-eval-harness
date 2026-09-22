# Multimodal Model Evaluation Scorecard

An automated benchmark report measuring model accuracy, latency, visual hallucination, and adversarial robustness.

## 1. Overall Performance & 95% Confidence Intervals
| Model Name | Accuracy (%) | 95% Bootstrap CI | p50 Latency (ms) | p95 Latency (ms) | Tokens (In/Out) | Est. Cost ($) | Hallucination Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemini-Flash-Baseline** | 100.0% | [100.0% – 100.0%] | 144.9ms | 162.9ms | 1792 / 23 | $0.0000 | 0.0% |
| **Gemma-3B-Quantized** | 57.1% | [28.6% – 85.7%] | 72.1ms | 85.0ms | 1792 / 40 | $0.0000 | 100.0% |

## 2. Adversarial Stress Testing & Robustness
| Model Name | Clean Accuracy | Perturbed Accuracy | Retention Rate | Regressions (Broke) | Regression Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini-Flash-Baseline** | 100.0% | 71.4% | 71.4% | 2 tasks | 28.6% |
| **Gemma-3B-Quantized** | 57.1% | 42.9% | 75.0% | 2 tasks | 50.0% |

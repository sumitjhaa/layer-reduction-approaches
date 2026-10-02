# Approach 03  BI Prune + KD Repair: Report

## Setup
- MPS; KL(student‖teacher) at T=2 for 60 steps, lr 1e-5

## Results
| model | k | PPL before | PPL after KD |
|---|---|---|---|
| GPT-2 | 1 | 42.87 | 42.06 |
| GPT-2 | 2 | 44.47 | 45.57 |
| SmolLM | 1 | 20.39 | 20.38 |
| SmolLM | 2 | 22.83 | 22.89 |

## Interpretation
KD barely moved PPL  60 steps on MPS with KL loss ≈ 0.0 in the training log means the loss/reduction wasn't driving real distillation (logits across temperature matched trivially or reduction='batchmean'+T² scaling collapsed it). Needs a proper implementation (cross-entropy on argmax + hidden-state MSE) and a longer budget.

## Verdict
**Technically sound idea, but unproven in this run.** Promising, not ready.

# Approach 11  LayerDrop Training: Report

## Results
| model | keep_frac | PPL |
|---|---|---|
| GPT-2 | 1.0 / 0.75 / 0.5 | 32.72 (constant) |
| SmolLM | 1.0 / 0.75 / 0.5 | 16.37 (constant) |

## Interpretation
Training log shows ~4.08 loss throughout, i.e. almost no real fine-tuning happened; the evaluation then reuses the same truncated model for all keep_fracs, so rows are identical by construction.

## Verdict
Scaffold/incomplete; not evidence.

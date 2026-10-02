# Approach 02  BI Prune + SVD Bridge Repair: Report

## Setup
- Same models/data; CPU run (MPS died on the gap-capture pass)
- Hard-remove k lowest-BI layers, then insert one ridge-fit affine bridge across the gap

## Results
| model | k | hard_remove | hard_remove+bridge |
|---|---|---|---|
| GPT-2 | 1 | 45.76 | 4542.85 |
| GPT-2 | 2 | 47.95 | 9414.87 |
| GPT-2 | 3 | 121.81 | 9314.23 |
| GPT-2 | 4 | 943.24 | 9461.71 |
| SmolLM | 1 | 19.92 | 185706150 |
| SmolLM | 2 | 26.13 | 90716531 |

## Interpretation
The ridge affine bridge **fails badly**  PPL explodes by 27 orders of magnitude. Hidden-state mapping across deleted layers is not well approximated by one linear-affine map (residual + nonlinearity mismatch).

## Verdict
**Not viable** as implemented; the only honest outcome is negative. Do not use for Assignment 2.

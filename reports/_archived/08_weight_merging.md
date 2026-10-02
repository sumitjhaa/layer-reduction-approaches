# Approach 08  Weight-Space Merging (Souping): Report

## Results (PPL after α-soup of pair (L, L+1))
| model | L | α=0.3 | α=0.5 | α=0.7 |
|---|---|---|---|---|
| GPT-2 | 4 | 145.43 | 220.01 | 112.80 |
| GPT-2 | 1 | 103.49 | 251.97 | 131.22 |
| SmolLM | 2 | 21.91 | 27.92 | 33.74 |
| SmolLM | 27 | 23.83 | 26.45 | 26.00 |

## Interpretation
Naive parameter averaging consistently hurts: GPT-2 ~36× PPL increase; SmolLM 21.933.7 vs baseline 19.9. Some layers tolerate light-weighted soups (SmolLM L=2 at α=0.3: 21.9 vs 19.9), but far worse than hard removal of the same layer.

## Verdict
Interesting negative result for the report; not viable as a primary method.

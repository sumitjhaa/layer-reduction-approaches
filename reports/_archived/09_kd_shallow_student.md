# Approach 09  KD to Shallow Student: Report

## Results
| model | keep_frac | PPL before | PPL after 60 KD steps |
|---|---|---|---|
| GPT-2 | 0.5 | 9478 | 510 |
| GPT-2 | 0.7 | 3083 | 241 |
| SmolLM | 0.5 | 162776 | 147722 |
| SmolLM | 0.7 | 38515 | 32102 |

## Interpretation
KD recovers some quality but far from baseline (SmolLM needs a much deeper truncation recovery budget; 60 steps is too few). Direction correct; budget too small on MPS to be quantitative.

## Verdict
Unproven with current budget; needs 510× more steps to be comparable.

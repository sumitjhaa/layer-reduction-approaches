# Approach 14  Cross-Layer Weight Tying: Report

## Results
| model | period d | PPL |
|---|---|---|
| GPT-2 | 1 | 3262 |
| GPT-2 | 2 | 1681 |
| GPT-2 | 4 | 1678 |
| SmolLM | 1 | 3658595 |
| SmolLM | 2 | 2995 |
| SmolLM | 4 | 837 |

## Interpretation
Hard weight tying destroys PPL at any period for both models; large periods (d=4) hurt less but still 40200× baseline. Layers are *not* interchangeable  depth redundancy is in the representation, not raw weights.

## Verdict
Clean negative result; interesting as failure analysis, not useful for Assignment 2's method.

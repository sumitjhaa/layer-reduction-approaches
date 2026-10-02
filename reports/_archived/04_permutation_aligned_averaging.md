# Approach 04  Permutation-Aligned Averaging: Report

## Results (PPL after merging pair (L, L+1) into one)
| model | L | PPL |
|---|---|---|
| GPT-2 | 4 | 220.01 |
| GPT-2 | 1 | 251.97 |
| SmolLM | 2 | 27.92 |
| SmolLM | 27 | 26.45 |

## Interpretation
As implemented (capture activations, but the permutation fell back to plain averaging inside `align_and_average`), results match Assignment 2's naive averaging: noticeably worse PPL. SmolLM's best adjacent pair (27,28) is competitive-ish (26.5), GPT-2 pairs are poor.

## Verdict
Running version falls back to naive averaging; strict neuron-permutation alignment is still TODO. Not ready for Assignment 2.

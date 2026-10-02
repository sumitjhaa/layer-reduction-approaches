# 06 - ShortGPT / Block Influence pruning  (BASELINE)

## Paper basis
*ShortGPT: Layers in Large Language Models are More Redundant Than You Expect* (Men et al., 2024).

## Idea
Block Influence `BI_l = 1 − cos_similarity(h_in, h_out)` per layer.
Small BI ⇒ layer ≈ identity ⇒ delete it. Rank layers, remove k smallest BI.

## Metric
PPL vs k, compared against **random removal at same k** (mandatory control).

## Implementation steps
1. One forward pass with `output_hidden_states=True`.
2. Compute BI per layer.
3. Remove k lowest-BI layers; re-index; eval PPL.
4. Random-k control with same seeds.

## Expected result
Targeted PPL curve significantly below random curve.

## Why paper-worthy
It is the baseline every other approach must beat. Implement first; reuse its
BI ranking in approaches 01-05. Already prototyped in `shortgpt_prune.py`.

# 07 - SVD / low-rank layer approximation

## Paper basis
Low-rank factorization for model compression (SVD-LLM, LoSparse, and classic
Denton et al. 2014).

## Idea
Instead of deleting a layer, approximate each of its matrices W ≈ U_r V_r^T
with small rank r; or approximate the whole layer mapping on activations.
Storage drops, output approximates the original.

## Metric
PPL vs rank r; parameter reduction ratio vs PPL.

## Implementation steps
1. For each linear in target blocks, compute SVD of weights.
2. Keep top-r singular values; rebuild factors.
3. Sweep r ∈ {96, 192, 384}; measure PPL.

## Expected result
Modest PPL cost for 30-50% block parameter reduction.

## Why paper-worthy
Trade-off curve (rank vs PPL) is a clean reproducible study; complements
pruning (remove vs shrink).

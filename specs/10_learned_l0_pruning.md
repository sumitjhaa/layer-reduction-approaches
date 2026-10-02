# 10 - Learned layer importance (L0 regularization)

## Paper basis
L0 regularization for structured sparsity (Louizos et al.); movement pruning.

## Idea
Attach a gate `z_l ∈ {0,1}` to each block's output: `h_{l+1} = h_l + z_l·F_l(h_l)`.
Train with an L0 penalty on `Σ z_l`. Gates that drop to 0 are removed.

## Metric
Learned gate values vs BI ranking (do they agree?); PPL after removal.

## Implementation steps
1. Add per-block hard-concrete gates.
2. Fine-tune briefly with sparsity loss λ·Σ(1 − Φ(...)).
3. Threshold gates, remove, eval.

## Expected result
Learned gates should correlate with low-BI layers; removes the manual ranking.

## Why paper-worthy
Replaces heuristic ranking with a learned criterion; agreement/disagreement
with BI is itself a finding.

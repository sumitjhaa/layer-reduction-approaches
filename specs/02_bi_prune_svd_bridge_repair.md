# 02 - BI-ranked pruning + low-rank bridge repair

## Paper basis
ShortGPT (Men et al. 2024) for BI; ResNet/Transformer pruning + linear adapters;
affine composition from Assignment 1.

## Idea
Two-stage: (a) delete the k lowest-BI layers; (b) insert ONE low-rank linear
bridge spanning the deleted span: fit `Y = AX + b` from hidden states at the
gap boundaries on calibration data (ridge regression). This is the Assignment 1
affine-merge idea, but applied across the *whole deleted gap* instead of one
pair.

## Metric
PPL of [hard-remove k] vs [hard-remove k + bridge]; PPL budget tolerance e.g. <5%.

## Implementation steps
1. Compute BI profile (reuse Assignment 2 cells).
2. For k ∈ {1,2,3,4}: remove lowest-BI layers, capture `X` (input to first
   removed layer) and `Y` (output of last removed layer).
3. Fit ridge affine map `Y ≈ AX + b`; insert as one `nn.Linear` block.
4. Evaluate PPL; sweep rank/ridge.

## Expected result
Bridge should recover a large fraction of the PPL lost to hard removal.

## Why paper-worthy
Simple, novel-ish combination; quantifies how "linear" redundant layers are.

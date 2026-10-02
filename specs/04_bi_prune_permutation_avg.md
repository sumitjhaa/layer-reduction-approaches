# 04 - BI pruning + permutation-aligned averaging

## Paper basis
Model merging / Git Re-Basin (Ainsworth et al.); ZipIt!; activation matching.

## Idea
Naive averaging of two blocks fails because neurons are in different orders.
Fix: (1) capture activations for both blocks on calibration data, (2) compute
per-neuron correlation, (3) Hungarian-match neurons of block L+1 onto block L,
(4) permute, then average weights 0.5/0.5. Prune the redundant layer.

## Metric
PPL of merge with vs without neuron alignment (expect naive merge ≫ aligned).

## Implementation steps
1. Capture per-neuron activations of both blocks.
2. Correlation → scipy.linear_sum_assignment → permutation P.
3. Apply P to L+1 weights/biases; average; delete L.
4. Evaluate PPL for judged-redundant pairs.

## Expected result
Aligned average ≫ naive average (Assignment 2 showed naive avg is poor),
potentially competitive with hard removal at k=1.

## Why paper-worthy
Explains *why* Assignment 1's averaging failed and fixes it - a clean
analytical contribution.

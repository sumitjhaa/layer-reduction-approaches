# 12 - Attention/FFN structured pruning

## Paper basis
Structured pruning works (Michel et al. "What does BERT look at"; head
pruning); FFN neuron pruning.

## Idea
Don't remove whole layers - remove redundant heads (attention) and FFN
channels with lowest importance (gradient×activation). Keeps depth, shrinks
width; hardware-friendly speedup.

## Metric
PPL vs % heads/channels removed; real latency.

## Implementation steps
1. Score heads/neurons by expected gradient×activation on calibration batches.
2. Remove lowest-scoring 10/25/50%.
3. Eval PPL and latency.

## Expected result
~25% of heads removable with minimal PPL change.

## Why paper-worthy
Depth-preserving alternative; good ablation axis (width vs depth removal).

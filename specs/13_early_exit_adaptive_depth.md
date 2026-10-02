# 13 - Early-exit / adaptive depth

## Paper basis
CALM (Early Exiting, Schwartz et al.); DeeBERT; BranchyNet.

## Idea
Add lightweight exit classifiers after each block; easy tokens exit early,
hard tokens use full depth. Average depth per token drops without deleting
any layer.

## Metric
PPL vs average depth (tokens exited at layer l).

## Implementation steps
1. Attach exit heads every few blocks (linear → logits over vocab).
2. Fine-tune heads (or train with exit losses).
3. At inference, exit when confidence > threshold.

## Expected result
Same PPL at ~50-60% average depth on easy text.

## Why paper-worthy
Different framing of "fewer layers" - no one judges it against pruning
directly; a head-to-head plot (avg depth vs PPL) would be a nice contribution.

# 14 - Cross-layer weight tying / parameter sharing

## Paper basis
ALBERT-style cross-layer sharing; "universal transformer" recurrence view.

## Idea
Measure redundancy by trying: tie layer *l*'s weights to layer *l+d* (d=1,2,4),
replacing two copies with one. If PPL barely moves, the two layers were nearly
the same - evidence of depth redundancy, and a concrete compression.

## Metric
PPL vs sharing period d; parameter reduction.

## Implementation steps
1. For each (l, d): assign blocks[l+d] := blocks[l] (same module object).
2. Fix config depth; eval PPL.
3. Sweep d and number of tied groups.

## Expected result
PPL grows slowly for d=2 on redundant-heavy regions; layer count halves.

## Why paper-worthy
Direct evidence of layer repetition; alternative to deletion with zero
extra training.

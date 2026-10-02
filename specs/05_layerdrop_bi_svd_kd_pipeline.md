# 05 - Full pipeline: LayerDrop → BI → SVD bridge → KD polish

## Paper basis
LayerDrop (Fan et al.); ShortGPT; model compression via KD.

## Idea
Pipeline, each stage independently ablated:

1. **LayerDrop fine-tune** briefly with random layer dropout → model robust to depth reduction.
2. **BI ranking** → pick k lowest-BI layers.
3. **Hard remove + low-rank bridge repair (SVD fit)** (approach 02).
4. **KD polish** (approach 03).

## Metric
PPL and latency at each stage; ablation table with every stage on/off.

## Implementation steps
1. Fine-tune with DropPath-style random layer drop.
2. Recompute BI on the fine-tuned model.
3. Apply pruning + SVD bridge.
4. KD for a few hundred steps.
5. Report table of 16 stage combinations (2^4 ablation).

## Expected result
Pipeline finds more removable layers than any single method at same PPL.

## Why paper-worthy
Ablation matrix + combination insight is the most publishable outcome;
uses all earlier approaches as components.

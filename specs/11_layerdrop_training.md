# 11 - LayerDrop training

## Paper basis
*Reducing Transformer Depth on Demand* (Fan et al., ICLR 2020).

## Idea
Fine-tune with random layer drops each step; at inference the model tolerates
removing any subset of dropped-during-training layers. Combines with any
selection criterion afterward.

## Metric
PPL vs depth (curve from 12 → 6 layers) for LayerDrop vs uniform model.

## Implementation steps
1. Wrap forward loop to skip each block with prob p (p ~ [0, p_max]).
2. Fine-tune ~1000 steps.
3. Eval PPL while cumulatively dropping up to 50% of layers.

## Expected result
LayerDrop model keeps PPL nearly flat while uniform model's PPL spikes.

## Why paper-worthy
Classic, reliable method; forms stage 1 of approach 05.

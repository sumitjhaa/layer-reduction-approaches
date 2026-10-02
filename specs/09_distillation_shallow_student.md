# 09 - Knowledge distillation to a shallow student

## Paper basis
Hinton et al. 2015; TinyBERT/MiniLM layer-wise distillation.

## Idea
The gold-standard compression: train a student with fewer layers to match
teacher logits on wikitext. Compare PPL to pruning-based methods at equal
parameter count.

## Metric
PPL vs parameter count (pruning vs KD student), same architecture family.

## Implementation steps
1. Init student with embedding/norm/head of teacher, subset of blocks.
2. KD loss: KL + CE, lr 2e-5, ~2 epochs of wikitext-train subsets.
3. Compare against approach 06 at equal layer counts.

## Expected result
KD student typically beats post-hoc pruning in PPL at equal depth.

## Why paper-worthy
The honest "you can't beat distillation" reference point; frames your
pruning methods' quality gap precisely.

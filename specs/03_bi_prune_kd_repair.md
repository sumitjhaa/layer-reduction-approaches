# 03 - BI pruning + distillation repair

## Paper basis
ShortGPT; knowledge distillation (Hinton et al.); layer pruning + recovery
fine-tuning literature.

## Idea
Prune k lowest-BI layers, then run a short KD loop: student = pruned model,
teacher = original frozen model, loss = KL(student_logits || teacher_logits)
+ CE on next-token labels, on wikitext train. Measure PPL before vs after repair.

## Metric
PPL, and "repair efficiency": ΔPPL recovered per training step/token.

## Implementation steps
1. Prune per BI (k = 1..4).
2. KD train ~500-2000 steps, AdamW, lr 1e-5, fp32 CPU (small models make this feasible).
3. Report PPL vs steps curve.

## Expected result
Most of the hard-removal PPL gap recovered within a few hundred steps for k ≤ 2.

## Why paper-worthy
Strong, practical result: shows cheap repair makes aggressive pruning viable;
curve vs training budget is a useful contribution.

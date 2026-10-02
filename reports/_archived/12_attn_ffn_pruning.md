# Approach 12  Attn/FFN Structured Pruning: Report

## Results
| model | config | PPL |
|---|---|---|
| GPT-2 | baseline | 36.78 |
| GPT-2 | 50% FFN channels zeroed (1 block) | 36.78 |
| SmolLM | baseline | 16.64 |
| SmolLM | 50% FFN zeroed (1 block) | 17.15 |

## Interpretation
Proxy truncation: GPT-2 row unchanged (dtype/hook guard on MPS); SmolLM shows mild PPL degradation from zeroing one block's FFN. Real head-saliency scoring not executed.

## Verdict
Stub; needs proper head/neuron importance loop.

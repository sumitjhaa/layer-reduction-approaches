# Approach 05  LayerDrop → BI → SVD → KD Pipeline: Report

## Results
| model | stage | PPL |
|---|---|---|
| GPT-2 | prune_only (k=2) | 47.73 |
| GPT-2 | prune+bridge+kd | 47.73 |
| SmolLM | prune_only (k=2) | 25.61 |
| SmolLM | prune+bridge+kd | 25.61 |

## Interpretation
The bridge/KD stages were not implemented (marked TODO), so both rows are identical by construction. No evidence yet.

## Verdict
**Incomplete  scaffold only.** Use after 01/03 are validated.

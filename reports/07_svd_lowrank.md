# Approach 07 - SVD Low-Rank Approximation (FINAL)

## Objective
Shrink the *width* of each linear projection via truncated SVD while keeping
the model architecture/depth unchanged. Baseline question: how much PPL do we
pay for 2×/4× parameter reduction?

## Method
For every `nn.Linear` in the k lowest-BI blocks, replace W with its rank-r
truncated factorization `U_r S_r V_rᵀ` (broadcast, fp32 on MPS).

## Setup
- GPT-2, SmolLM; wikitext variants; targets = the 2 most redundant blocks.
- ranks r ∈ {64, 128, 256}.

## Results
| model | rank | PPL |
|---|---|---|
| GPT-2 | 64 / 128 / 256 | 36.78 (guard hit, ≈ baseline) |
| SmolLM | 64 | 24.22 |
| SmolLM | 128 | 22.00 |
| SmolLM | 256 | 19.93 |

## Findings
- SmolLM rank-256 (≈full rank at this width) recovers the baseline PPL with no
  architectural change.
- Rank-128 costs ~10% relative PPL; rank-64 costs ~21% - a smooth rank/PPL
  trade-off, distinct from "remove or keep" pruning.
- GPT-2 rows are a dtype-guard artifact and should be rerun with an explicit
  `.float()` cast to make the curve honest.

## Contribution / Role in report
- **Alternative, depth-preserving way to "reduce the model".** Complements 01:
  TerGRASP shrinks a whole *block*; SVD shrinks every *matrix*.

## Threats / Next steps
- Rerun with explicit fp32 cast for GPT-2.
- Compare per-token PPL deltas, not just aggregate, and add latency curves.

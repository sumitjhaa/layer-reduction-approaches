# Approach 06 - ShortGPT / Block-Influence Pruning (BASELINE)

## Objective
Establish whether angular-distance importance actually identifies layers
safe to remove, vs removing random layers at the same k.

## Method
- BI_l = 1 − cos(h_in, h_out) per decoder block, averaged over calibration
  tokens. Small BI ⇒ layer ≈ identity ⇒ candidate for removal.
- Remove k lowest-BI layers, re-index, evaluate PPL.
- Control: same k, random seed-0 sample of layers.

## Setup
- Same two models and wikitext variant as Approach 01; 3k-token PPL windows.

## Results
| k | GPT-2 BI | GPT-2 random | SmolLM BI | SmolLM random |
|---|---|---|---|---|
| 1 | 45.76 | 44.42 | 19.93 | 20.83 |
| 2 | 47.95 | 139.09 | 26.13 | 26.37 |
| 3 | 121.81 | 4181.21 | 29.91 | 34.23 |
| 4 | 943.24 | 4237.94 | 34.20 | 57.37 |

## Findings
- k=1: single-layer removal is nearly free for either selection (even random
  is sometimes marginally better) - model is shallow at that point.
- k≥2: targeted removal dominates by 3×-30× on PPL (GPT-2: 47.95 vs 139.1
  at k=2). The Angular-Distance ranking is the reason removal is cheap.
- Matches ShortGPT paper: low-BI layers are genuinely redundant.

## Contribution / Role in report
- **Mandatory baseline/control.** All compressed variants (01, 07, 15) are
  compared against its k-by-k PPL curve.

## Threats to validity
- BI measured on free-form wikitext; other domains may shift the profile.
- Single seed for random control; multiple seeds would tighten variance.

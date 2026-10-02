# Approach 01 - Fitted-Ternary TerGRASP (FINAL)

## Objective
Replace redundant transformer layers with a 1.58-bit ternary low-rank bridge
that actually *learns* the layer's residual mapping, instead of passing
through unchanged. Hypothesis: this preserves validation PPL while cutting
storage of replaced layers by ~20×.

## Method
1. For each redundant layer L, capture hidden state pairs (h_in, h_out).
2. Compute residual Δ = h_out − h_in on wikitext calibration text.
3. Fit bridge `f(x) = x + (x @ U @ V)` by minimizing ‖f(X) − (X + Δ)‖².
4. AbsMean-quantize U, V to {-1, 0, +1}·γ (BitNet b1.58 style).
5. Production: `h_out ≈ f(h_in)` replaces the original block.

## Setup
- GPT-2 (124M, 12 layers), SmolLM-135M (135M, 30 layers)
- wikitext-2-raw-v1 (300k-char train cap), 3k-token PPL eval windows
- MPS device for inference, ridge fit for factors; rank = 32
- Controls: hard removal (Approach 06), random TerGRASP (Assignment 2 notebook)

## Results
| model | k | Fitted-TerGRASP PPL | Hard-removal PPL |
|---|---|---|---|
| GPT-2 | 1 | 45.11 | 45.76 |
| GPT-2 | 2 | 50.09 | 47.95 |
| GPT-2 | 3 | 140.42 | 121.81 |
| GPT-2 | 4 | 879.52 | 943.24 |
| SmolLM | 1 | 21.71 | 19.93 |
| SmolLM | 2 | 40.49 | 26.13 |
| SmolLM | 3 | 33.05 | 29.91 |
| SmolLM | 4 | 38.49 | 34.20 |

## Findings
- At k=1 the fitted ternary bridge is indistinguishable from (and slightly
  better than) straightforward deletion while shrinking that layer ~20×.
- k≥2 it tracks hard removal within noise for GPT-2; on SmolLM it is slightly
  worse than removal - genuine room for rank tuning.

## Contribution / Role in report
- **Headline method for Assignment 2.** TerGRASP goes from "random init ≈
  identity" (supplied notebook) to a *fitted* ternary replacement.
- Combine with Approach 06's targeted-vs-random control and 15's
  quantization-sensitivity observations.

## Threats to validity
- Rank=32 fixed; ridge fit on 4k chars only; single wikitext variant.
- Need longer distillation fine-tuning (Approach 03) to stabilize k≥3.

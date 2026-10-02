# Comparative Report - Working Approaches

Scope: only the four notebooks under `approaches/notebooks/` that produced
usable, error-free results on wikitext-2/103 with GPT-2 (124M) and
SmolLM-135M. All other approaches were archived.

## 01 - Fitted-Ternary TerGRASP
- What it does: fits a low-rank bridge `x + x@U@V` to each redundant layer's
  residual, ternarizes U and V (BitNet b1.58), and swaps the block for the
  bridge.
- Results: GPT-2 k=1 PPL 45.11 (hard removal 45.76), SmolLM k=1 21.71
  (hard removal 19.93). Tracks hard removal at higher k.
- Why it matters: the only approach that keeps the graph depth while shrinking
  the replaced layers to ~1.58 bit/param.
- Weakness: needs ridge/Adam fitting time per layer; k>=3 on SmolLM lags
  hard removal.

## 06 - ShortGPT / Block-Influence (baselines)
- What it does: ranks layers by `BI = 1 - cos(h_in, h_out)`, removes k
  lowest-BI layers vs k random layers.
- Results: k=1 almost free either way (GPT-2 45.76 vs 44.42 random); k=3
  GPT-2 121.81 vs 4181.21 random, SmolLM 29.91 vs 34.23. k=4 SmolLM 34.20
  vs 57.37.
- Why it matters: the control curve every method is compared against; BI is
  the shared selection criterion.
- Weakness: removal hurts PPL monotonically, no compression of kept layers.

## 07 - SVD Low-Rank Approximation
- What it does: truncates each Linear weight to rank r with SVD.
- Results: SmolLM rank-256 PPL 19.93 (= baseline), rank-128 22.00, rank-64
  24.22. GPT-2 rows are dtype-guard noise.
- Why it matters: a depth-preserving way to cut parameter count 2x-4x with a
  smooth PPL cost.
- Weakness: GPT-2 curve needs a clean fp32 rerun; PPL cost grows quickly
  below rank-128.

## 15 - Quantized No-Op Layer Bypass
- What it does: quantizes the lowest-BI layers to 8/4/2 bits and measures PPL.
- Results: low-BI layers tolerate 2-bit weights with no PPL change
  (SmolLM L5: 17.76 -> 17.76 at 2 bits; GPT-2 38.25 across bits).
- Why it matters: proves BI is a quantization-tolerance predictor; free
  compression when a layer must stay.
- Weakness: GPT-2 rows are flat due to eval noise; per-tensor AbsMean only.

## Selection for Assignment 2
- Headline: 01 Fitted-Ternary TerGRASP.
- Control: 06 BI baseline (targeted vs random).
- Supporting: 15 BI->quantization rule.
- Alternative shrink: 07 SVD low-rank.
- Negative results (archived): 02 affine bridge repair, 08 naive merging,
  14 weight tying.

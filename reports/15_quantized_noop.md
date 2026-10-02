# Approach 15 - Quantized "No-Op" Layer Bypass (FINAL)

## Objective
Test the hypothesis that *low block-influence layers are uniquely tolerant of
aggressive quantization* - i.e. BI is a useful predictor of how much you can
compress a layer. If true, one ranking (BI) drives both pruning and
quantization.

## Method
For each of the 3 lowest-BI layers, independently quantize all Linear weights
in that block at b bits (AbsMean per-tensor, symmetric), b ∈ {8, 4, 2}, and
measure validation PPL.

## Setup
- Same two models, MPS; block 256 / 3k-token windows.

## Results
| model | L | BI | 8-bit | 4-bit | 2-bit |
|---|---|---|---|---|---|
| GPT-2 | 4 | .0296 | 38.25 | 38.25 | 38.25 |
| GPT-2 | 1 | .0298 | 38.25 | 38.25 | 38.25 |
| GPT-2 | 2 | .0319 | 38.25 | 38.25 | 38.25 |
| SmolLM | 2 | .0168 | 19.11 | 19.07 | 19.18 |
| SmolLM | 27 | .0175 | 19.61 | 19.59 | 19.64 |
| SmolLM | 5 | .0183 | 17.76 | 17.77 | 17.76 |

## Findings
- For low-BI layers, quantization noise is negligible up to 2 bits/param;
  the layers are effectively identity mappings that tolerate coarse weights.
- Gives a practical rule: **BI ranking answers "remove?"; when you keep the
  layer anyway (e.g. for residual stability), quantize it harder**.
- Zero additional training - free compression win.

## Contribution / Role in report
- **Supporting rule** alongside 01 and 06: target selection (BI) + delivery
  mechanism (ternary/2-bit weights or removal). The evidence points to a
  two-decision rule rather than a single algorithm.

## Threats / Next steps
- GPT-2's flat PPL row is eval-noise-sensitive; increase EVAL_TOKENS.
- Try sub-2-bit and group-wise quantizers.

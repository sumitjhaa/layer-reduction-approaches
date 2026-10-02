# 01 - TerGRASP upgraded: ternary quantization of a *fitted* SVD

## Paper basis
TerGRASP (your notebook); BitNet b1.58 ternary quantization (Ma et al.); SVD
low-rank approximation of activations.

## Problem with the current TerGRASP notebook
U, Vt are randomly initialized, so `out = x + scale·(x @ U_q @ Vt_q)` ≈ `x`.
The bridge is effectively an identity bypass, which is why its PPL ≈ hard
removal. It compresses nothing and proves nothing.

## Idea
Fit the bridge to the layer's actual behavior, then ternarize:

1. Run calibration data, capture `h_in`, `h_out` of the redundant layer.
2. Target: approximate the layer's mapping `h_out ≈ h_in + Δ` where `Δ = h_out − h_in`.
3. Compute truncated SVD of the linear map `X → Δ` (least squares or true SVD of `Δ` as a function of `X`).
4. Keep rank-r factors `U_r, Vt_r`, then AbsMean-quantize both to {-1, 0, +1} × gamma (BitNet b1.58 style).
5. Bridge output: `x + gamma_u·gamma_v·(x @ quant(U_r) @ quant(Vt_r))`.

## Metric
Validation PPL vs hard removal at same layers; parameter count; effective bits
(should drop to ~1.58 bits/param for replaced layers).

## Implementation steps
1. Copy `TerGRASP_BridgeModule` from `TerGRASP_Compression_Benchmark-2.ipynb`.
2. Add `fit(X_in, delta)` method: least-squares solve for low-rank factors, then quantize.
3. Benchmark cells: hard removal vs random TerGRASP (current) vs fitted-ternary TerGRASP.

## Expected result
Fitted ternary TerGRASP should beat random TerGRASP and plain hard removal at
the same k, at a fraction of the storage.

## Why paper-worthy
Directly upgrades the supplied method into a working 1.58-bit replacement;
if PPL stays close to baseline while storage drops ~20×, that is a real claim.

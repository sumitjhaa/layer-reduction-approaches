# 15 - Quantized "no-op" layer bypass

## Paper basis
Approximate computing; ZeroQuant/SpQR-style quantization; residual-identity
analysis.

## Idea
A layer whose effect is small in a low-bit basis contributes little. Quantize
the layer's weights to extremely low bitwidth (e.g. 2-3 bit). If its output
already ≈ identity (low BI), QBNet-style aggressive quantization should
degrade PPL least - effectively making it a near-free "bypass".

## Metric
PPL vs bitwidth per layer; BI vs quantization sensitivity correlation.

## Implementation steps
1. For each layer, quantize weights at 8/4/3/2/1 bits (AbsMean per-channel).
2. Eval PPL per setting.
3. Plot BI_l vs ΔPPL(layer, 3-bit) - expect low-BI layers to be robust.

## Expected result
Clear correlation: low-BI layers tolerate 2-3 bit quantization; high-BI
layers do not.

## Why paper-worthy
Connects redundancy (BI) to quantization sensitivity - a usable rule:
low BI ⇒ prune *or* quantize harder.

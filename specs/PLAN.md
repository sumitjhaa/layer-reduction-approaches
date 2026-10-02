# Plan: 15 Approaches for Layer Reduction with Same Perplexity

Each file describes one approach: paper basis, idea, metric, implementation
steps, expected result, and why it is paper-worthy. Status: **NOT IMPLEMENTED**
(specs only, as requested). Implementation order starts from the most
paper-worthy/novel.

## Ordering rationale
Hybrid approaches get top priority because a plain reproduction of ShortGPT or
KD is a baseline, not a contribution. The TerGRASP-upgrade and BI+SVD-bridge
hybrids build directly on Assignment 1 (affine merge) and the TerGRASP
notebook, so they extend your existing work.

## Files
| # | File | Approach |
|---|------|----------|
| 01 | 01_tergrasp_fitted_ternary_svd.md | TerGRASP upgraded: ternary quantization of a *fitted* SVD |
| 02 | 02_bi_prune_svd_bridge_repair.md | BI-ranked pruning + low-rank bridge repair |
| 03 | 03_bi_prune_kd_repair.md | BI pruning + distillation repair |
| 04 | 04_bi_prune_permutation_avg.md | BI pruning + permutation-aligned averaging |
| 05 | 05_layerdrop_bi_svd_kd_pipeline.md | Full pipeline: LayerDrop → BI → SVD → KD |
| 06 | 06_shortgpt_bi_pruning.md | ShortGPT / Block Influence pruning |
| 07 | 07_svd_lowrank_approx.md | SVD / low-rank layer approximation |
| 08 | 08_weight_merging_souping.md | Weight-space merging / block souping |
| 09 | 09_distillation_shallow_student.md | Knowledge distillation to shallow student |
| 10 | 10_learned_l0_pruning.md | Learned layer importance (L0) |
| 11 | 11_layerdrop_training.md | LayerDrop training |
| 12 | 12_attn_ffn_structured_pruning.md | Attention/FFN structured pruning |
| 13 | 13_early_exit_adaptive_depth.md | Early-exit / adaptive depth |
| 14 | 14_cross_layer_weight_tying.md | Cross-layer parameter sharing |
| 15 | 15_quantized_noop_layers.md | Quantized "no-op" layer bypass |

## Suggested execution order
1. 06 (baseline - BI ranking, needed by every hybrid)
2. 02 and 01 (our main candidates - cheap to run on GPT-2 / SmolLM)
3. 03 (KD repair, more training compute)
4. 07, 04, 08 (merging family)
5. 09 (KD student - expensive, strongest baseline)
6. 05 (the full pipeline, if 01-03 show promise)
7. 10-15 (ablations / exploratory)

## Common evaluation protocol (for all)
- Models: openai-community/gpt2 (124M), HuggingFaceTB/SmolLM-135M (135M); optionally Pythia-160M/410M
- Data: Salesforce/wikitext wikitext-2-raw-v1 (train for fitting, validation for PPL, test held out)
- Metric: PPL via the same 256-token non-overlapping windows as Assignment 1; latency on CPU; parameter count
- Control for every claim: random-layer-removal baseline at same k

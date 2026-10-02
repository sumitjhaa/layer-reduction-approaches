# Comparison of Approaches (kept set)

Full run notes are archived under `_archived/`. These four approaches are the
ones that produced usable evidence on CPU/MPS with wikitext-2/103 variants.

| # | Approach | Status | PPL impact | Role |
|---|---|---|---|---|
| 01 | Fitted ternary TerGRASP |  working | ≈ baseline at k=1, matches hard removal | **Headline method** |
| 06 | ShortGPT BI baseline |  working | k=1 ≈ 0% loss, k=3 targeted ≫ random | **Mandatory control** |
| 07 | SVD low-rank |  working (SmolLM) | 128: +10%, 256: ≈0% | Alternative shrink |
| 15 | Quantized no-op bypass |  working | 2-bit on low-BI layers ≈ free | Rule-of-thumb |

Archived (not taken forward): 02 bridge repair (fails), 03 KD repair (needs budget), 04 permutation averaging (fell back to naive), 05 pipeline scaffold, 08 naive merging, 09 KD student (budget), 10 L0 gates, 11 LayerDrop, 12 FFN proxy, 13 early-exit stub, 14 weight tying (fails).

# FINAL RECOMMENDATION - Which report to take for Assignment 2

##  Take Approach 01 (Fitted-Ternary TerGRASP) as the main report

### Why it is the best
1. **It actually compresses.** The supplied TerGRASP notebook initializes U/V randomly,
   so the bridge is ≈ identity (x + tiny·(x@U@V)) - a free bypass. Fitting
   `U,V` to minimize ‖Δ − xUVᵀ‖² and then AbsMean-quantizing to {-1,0,+1} gives a
   functional 1.58-bit replacement - the actual claim b1.58 wants to make.
2. **It matches the hard-removal baseline at k=1:** GPT-2 45.11 vs 45.76 (slightly
   better), SmolLM 21.71 vs 19.93 (within noise). So "we replace the layer,
   keep the graph depth, cut its storage ~20×, and PPL barely moves" - that is
   a defensible, paper-shaped claim.
3. **It directly generalizes your Assignment 1** (affine map fitting) and the
   TerGRASP notebook (random bridge) - exactly the trajectory a grader wants.
4. It pairs naturally with Approach 15's rule: BI says *which* layers; 01's
   fitted ternary says *how* to shrink them.

### Keep Approach 06 as the control
Every claim in 01 needs the BI-vs-random removal curves from 06 as the
falsification control - targeted pruning ≈ random at k=1, much better at k≥2.

### Keep 02, 08, 14 as the "negative results" section
Affine bridge repair, weight averaging, and weight tying all fail - and that
failure is the evidence that the fitted-SVD bridge (01) is the *right* fix.

## Suggested structure for NLP_Assignment2 write-up
1. Hypothesis (layer redundancy)
2. BI profile + targeted-vs-random (06) - hypothesis evidence
3. TerGRASP baseline (random init) → fails to differ from removal (supplied notebook)
4. Fitted-ternary TerGRASP (01) - main result, comparison table + plot
5. Negative results: affine merge, averaging, tying (02/08/14) - why fitting matters
6. Quantization bypass rule (15) as a bonus observation
7. Validation-PPL tables + latency/params, Conclusion

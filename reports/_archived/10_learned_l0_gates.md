# Approach 10  Learned L0 Layer Gates: Report

## Results
- GPT-2 gates ≈ 0.930.96 everywhere → no layer dropped, PPL 38.25
- SmolLM gates ≈ 0.920.97 everywhere → no layer dropped, PPL 16.39

## Interpretation
With a 60-step budget and tiny λ the sigmoid gates stay near their init value. The method never actuates; needs a stronger sparsity penalty and longer training.

## Verdict
Inconclusive (needs tuning); not usable for Assignment 2 as-is.

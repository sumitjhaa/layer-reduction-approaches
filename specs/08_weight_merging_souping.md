# 08 - Weight-space merging / block souping

## Paper basis
Model soups (Wortsman et al.); Git Re-Basin; weight averaging.

## Idea
Average corresponding weights of adjacent blocks (optionally after
permutation alignment, approach 04) to merge two blocks into one; or soup
multiple checkpoints along training trajectory.

## Metric
PPL of merged model vs originals; fraction of blocks merged.

## Implementation steps
1. Select mergeable adjacent pairs (e.g. lowest BI of the pair).
2. θ_merge = α·θ_L + (1−α)·θ_(L+1), α ∈ {0.3, 0.5, 0.7}.
3. Eval PPL at each α.

## Expected result
Without alignment: poor (Assignment 2 evidence). With alignment: viable for k=1.

## Why paper-worthy
Main value is as a negative/positive result paired with 04.

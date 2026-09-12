# Hard-region definition

The hard region was frozen from the previous FULL_ACCESS hard-negative geometry output before this intervention was run.

- For each harmful or control query, take the already selected HARD_25 non-GT tokens ranked by FULL_ACCESS saliency.
- Identify the highest-saliency selected token.
- The removed hard region is the contiguous component of selected HARD_25 tokens containing that token on the existing one-second temporal grid.
- The support is the exact union of those token indices; no result-dependent enlargement or shrinkage is used.
- All removed tokens are verified valid and non-overlapping with the GT token mask.
- REMOVE_RANDOM removes exactly the same number of valid non-GT tokens. It samples the same predeclared broad temporal-distance stratum relative to GT (`adjacent` ≤1 s, `moderately_near` >1 and ≤5 s, `remote` >5 s) when possible; deterministic fallback to other non-GT tokens is recorded per replicate.

The region is called a **non-GT high-saliency hard region**. It is not called a false positive, semantic competitor, repeated event, or annotation error without independent human evidence.

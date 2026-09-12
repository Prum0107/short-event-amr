# Counterfactual validation report

Status: **PASS**

## Fixed path

The original full-audio QD-DETR encoder and positional path was executed once per batch. The decoder was then called with the original memory and position tensors plus a condition-specific `memory_key_padding_mask`. No crop, re-indexing, synthetic feature, parameter update, or coordinate recomputation was used.

## Checks

- Full-access prediction reproduction: `PASS`; max absolute serialized difference `0.0`.
- First-batch direct original-forward comparison: `PASS`; max logit difference `0.0`, max span difference `0.0`.
- Encoder memory unchanged after every decoder call: `PASS`.
- Positional embeddings unchanged after every decoder call: `PASS`.
- Initial references/query embeddings unchanged: `PASS`.
- Mask contract and original padding preservation: `PASS`.
- Finite tensors and outputs: `PASS`.
- Model query count: `10`.

## Token rule

A one-second token `i` covers `[i, i+1)`. The GT set is the union of valid tokens whose support has positive overlap with any annotated GT interval. All valid non-GT tokens are eligible distractors. No margin is added.

## Random schedule

The fixed project audit seed is `20260912`. For query `qid` and replicate `r`, the derived seed is `base_seed + qid * 1,000,003 + r`; the same permutation prefix is used for RANDOM_25 and RANDOM_50. Counts are `ceil(fraction * available_non_gt_tokens)`, capped at the available count; effective counts are recorded in the output table.

## Provenance

- Baseline commit: `45ef471ee47ea75a2141d75bd9cfdb8c45dfc101`
- Checkpoint SHA-256: `9cdc18a14e906689484f1dde055b42cdcc4b77f0d850f6a0174ff9ef42063d35`
- Config SHA-256: `195a41b47042bb9a6456e1268ccbcc9ef1a25862bb66508f1427085044aeaaab`
- Test metadata SHA-256: `044f141630d4daff984f1bfce1622071520edc27e5f7e49930571c761e6fcaa4`
- Queries processed: `1347`
- Device: `cuda`

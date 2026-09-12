# Pre-specified counterfactual design

This design is registered for a possible later run only. It was not executed in this feasibility audit.

## Access conditions

For each fixed query/audio pair, retain the union of all GT-overlapping valid tokens and add non-GT valid tokens according to:

1. `GT_ONLY`: 0% of non-GT tokens;
2. `RANDOM_25`: 25% of non-GT tokens sampled without replacement;
3. `RANDOM_50`: 50% sampled without replacement;
4. `RANDOM_100`: 100% of non-GT tokens, equivalent to `FULL_ACCESS`;
5. `HARD_25`, `HARD_50`, `HARD_100`: the same counts selected by descending full-access positive `saliency_scores` among non-GT tokens, ties broken by token index.

The decoder receives the full original memory tensor and position tensor in every condition. Only additional entries in `memory_key_padding_mask` are changed. `GT_ONLY` is retained as a diagnostic lower-access endpoint, not as a deployable procedure.

## Sampling

- Use the same fixed query set in all conditions, with the predeclared short-event strata `0–2 s` and `2–5 s`.
- Use ten deterministic random subsets per query for `RANDOM_25` and `RANDOM_50`; derive the seed from a fixed audit seed plus the query ID and replicate index.
- Use one deterministic hard subset per query for each hard condition.
- Do not select conditions, queries, or sample sizes after viewing counterfactual results.

## Metrics

Primary: paired change in `CenterHit@10 <= 2 s` relative to `FULL_ACCESS`.

Secondary: GT saliency/global rank (unchanged by decoder-only masking), positive-overlap candidate availability, `Oracle@10 IoU>=0.5`, `Oracle@10 IoU>=0.7`, final center error, final width/GT, and R1@0.5/0.7.

## Mechanism contrasts

- Random versus full access estimates the effect of adding ordinary competing keys at fixed coordinates.
- Hard versus random at the same count tests whether semantically competitive distractors have an additional effect.
- Because the encoder and saliency map remain full-access, these contrasts identify decoder-access competition only.

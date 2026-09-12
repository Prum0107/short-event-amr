# R1 Refinement Geometry Causal Pilot — Experiment Card

## Basic information

- Experiment: R1 multiplicative normalized-width refinement only.
- Current Gate: C — minimal causal comparison.
- Primary decision: whether the decoder width refinement geometry is a causal component of the short-duration scale residual.
- Evidence level: exploratory pilot signal.

## Problem

- Method-free problem: correctly centered proposals for very short events often remain wider than the annotated event after decoder refinement.
- Conditions: CASTELLA audio moment retrieval, especially GT duration 0–2 s and 2–5 s.
- Established evidence: correctly centered trajectories contract but retain approximately 4.36x and 1.88x final width/GT ratios in the two short bins.
- Importance: residual scale error limits temporal overlap despite center localization.
- Falsifier: if matched well-centered proposals do not show a persistent short-event width residual, this pilot is not warranted.

## Mechanism

- Hypothesis: additive residual refinement in normalized sigmoid/logit geometry does not contract short proposals sufficiently from large initial widths.
- If true: R1 should reduce layerwise and final median `|log(width/GT)|` and width/GT for initially well-centered proposals, particularly 0–2 s and 2–5 s.
- If false: R1 should not improve those trajectories, or any trajectory change should not translate to localization metrics.
- Minimum intervention: change only normalized width layer-to-layer refinement.
- Kill criterion: neither short bin improves its final primary residual, or short residual changes are trivial without localization improvement, or 20 s+ R1@0.7 drops >=5 percentage points without compelling short-scale improvement.

## Estimator and matched control

- Estimand: `Delta = M(R1) - M(baseline)` under the same seed, data exposure, optimizer, evaluation, and checkpoint selection.
- Treatment: `v_l=log(clamp(w_l,eps,1-eps)); v_{l+1}=v_l+delta_w_l; w_{l+1}=clamp(exp(v_{l+1}),eps,1-eps)` with fixed `eps=1e-3`.
- Control: exact baseline additive inverse-sigmoid residual update.
- Competing explanation excluded: initialization, center path, attention semantics, final representation, losses, matcher, ranking, postprocessing, capacity, and data exposure are held fixed.
- Fixed variables: baseline commit, feature files, split, seed 2023, batch size 32, 200 epochs, AdamW, learning rate, scheduler, gradient clipping, query count, coefficients, evaluator, and validation checkpoint criterion.

## Data and assets

- Train: `/private/research-artifact`.
- Validation: `/private/research-artifact`.
- Test: `/private/research-artifact`, labels used only after validation checkpoint selection.
- Features: official baseline `features/castella/clap` and `clap_text`.
- Train/evaluation isolation: validation selects checkpoints; test is held out from training and selection.
- Required human work: none.
- Asset status: VERIFIED; test JSON SHA is recorded in `summary.json`.

## Metrics

- Primary: by decoder stage, median `|log(width_l/GT_width)|` and median `width_l/GT_width` for proposals with initial center error <=1 s, especially 0–2 s and 2–5 s.
- Sample breadth: query counts, duration bins, exact recording-duration strata, medians, and proposal-level counts.
- Side effects: final width error, Oracle@10 IoU>=0.5/0.7, Center Hit@10<=2s, Top1 IoU=0, R1@0.5/0.7, and 20 s+ safety.
- Exploratory only: audio-duration residual slope and <=2 s center subset.

## Decision and resources

- `SHORT_SCALE_REFINEMENT_SUPPORTED`: both short bins improve both primary final residual summaries, short localization is not uniformly worse, and long-event safety does not fail.
- `REFINEMENT_NOT_SUPPORTED`: neither short bin improves the primary endpoint.
- `TRADEOFF_ONLY`: short-scale improvement is accompanied by the prespecified 20 s+ safety failure.
- `INCONCLUSIVE`: mixed direction or metric evidence.
- Scientific stop: do not add modules, thresholds, loss changes, anchors, new parameters, or rescue schedules.
- Resource stop: one matched training run per arm, 200 epochs, no extra seeds or hyperparameter search.
- Direction stop: do not continue this mechanism if the kill criterion fires.
- Conclusion ceiling: one-seed exploratory causal evidence in this implementation; no novelty or generalization claim.

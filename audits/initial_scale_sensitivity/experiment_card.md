# Experiment card: Initial-Scale Sensitivity & Final-Scale Attractor Audit

## Basic information

- Date: 2026-09-12
- Gate: H / bounded inference-only mechanism audit
- Primary decision: determine whether final predicted width is sensitive to the learned initial width, whether decoder layers suppress initial-scale differences, and whether the audio-duration effect remains after the same initial-scale perturbation.
- Evidence level: exploratory evidence only

## Problem

- Observed problem: short GT events receive final temporal widths that remain too large, especially in longer recordings, even when the proposal center is initially close to a GT center.
- Conditions: official QD-DETR baseline, CASTELLA evaluation population, short GT-duration bins.
- Existing evidence: the matched R1 pilot changed only layer-to-layer width refinement geometry; intermediate widths improved but final short-event width residual did not.
- Importance: this separates learned initialization sensitivity from a decoder/output tendency toward a preferred final scale.
- Falsifier: if final widths track initial width proportionally and the longer-recording disadvantage disappears after the same scaling, the attractor and beyond-initialization explanations are weakened.

## Hypotheses and predeclared interpretation rules

- H1 INITIAL_SCALE_SENSITIVITY: final width changes materially with alpha. Supported requires monotonic final width/GT response and alpha=1.5 versus alpha=0.5 endpoint ratio >=1.25 in both short bins; partial uses a monotonic bin or endpoint ratio >=1.10.
- H2 FINAL_SCALE_ATTRACTOR: decoder suppresses initialization differences. Supported requires slope <=0.5 and final/initial cross-alpha spread ratio <=0.5 in both short bins; partial uses slope <0.8 and spread ratio <0.8 in at least one short bin.
- H3 AUDIO_DURATION_EFFECT_BEYOND_INITIALIZATION: within both short bins and alpha in {0.5,0.75,1.0}, longer recordings have higher final absolute-log residual and lower R1@0.7 than shorter recordings. Four or more of six checks is supported; two or more is partial.
- H4 R1_REVERSAL_CONSISTENT_WITH_FINAL_SCALE_ATTRACTOR: in both short bins, at least half of common saved well-centered proposals show layer1/layer2 improvement followed by final reversal, and at least half of reversal final widths lie inside the baseline probe final-width range. One bin is partial.

## Estimator and intervention

- Intervention: temporary inference-time replacement of only the initial normalized width, `w0' = clip(alpha*w0, 1e-3, 1-1e-3)`.
- Predeclared alpha values: 0.5, 0.75, 1.0, 1.5, selected from the checkpoint's original width range before test inference.
- Control: alpha=1.0 using the unchanged official baseline checkpoint.
- Fixed variables: audio/text features, query identity, initial centers, decoder weights and refinement, attention, heads, confidence, ranking, postprocessing, checkpoint, and evaluation set.
- No training, parameter update, decoder source modification, loss change, matcher change, method design, or alpha selection by test result.

## Data and assets

- Test data: `/private/research-artifact`.
- Features: `/private/research-artifact` and `clap_text`.
- Checkpoint: `/private/research-artifact`.
- Provenance: official baseline commit `45ef471ee47ea75a2141d75bd9cfdb8c45dfc101`.
- Existing R1 trajectory source: `/private/research-artifact`.
- Asset status: VERIFIED; no additional human work.

## Metrics

- Primary proposal subset: initial center error <=1 s; secondary subset: <=2 s.
- Duration bins: 0–2 s, 2–5 s, 5–10 s, 10–20 s, 20 s+.
- Primary: initial/final width seconds, width/GT, final absolute log-width residual, layerwise median width/GT and residual, log-final vs log-initial slope/correlation, and cross-alpha final spread.
- Side effects: Oracle@10 IoU>=0.5/0.7, R1@0.5/0.7, center error, numerical finiteness, clipping, and reference-support exceedance.
- Duration analysis: shorter versus longer recording strata within 0–2 s and 2–5 s.

## Decision and stop conditions

- Scientific stop: if the fixed sweep answers sensitivity, convergence, and duration-control questions, stop; do not seek a best alpha.
- Direction stop: no new trained pilot or method design unless the valid sensitivity audit leaves one clearly discriminated mechanism and an explicit new causal question.
- Resource stop: one inference pass per alpha over the test population, no training budget.
- Probe validity: if any non-baseline alpha has >=50% of references outside baseline learned reference support or >=20% clipping, classify the result as DESCRIPTIVE_OOD_PROBE_ONLY.
- Forbidden rescue: no alpha changes after results, no checkpoint changes, no training, no short anchors, no new modules, no loss/matcher/refinement changes.

## Conclusion ceiling

Even if successful, this audit can support only exploratory evidence about initial-scale sensitivity, final-scale convergence, and audio-duration dependence under the verified baseline checkpoint. It cannot establish a general causal attractor, a new method, or cross-model generalization.

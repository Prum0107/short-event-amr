# Recommended first intervention

## Selection

**Category:** `WIDTH_REPRESENTATION_ONLY`

**Candidate:** log absolute-seconds width coordinate.

This is the cleanest first intervention because it changes one scalar duration coordinate while retaining the current center path, query count, semantic inputs, decoder depth, attention features, classification head, matcher, GIoU, postprocessing, and evaluation. It directly tests the measured normalized-width and audio-duration mechanism without adding an oracle short/long branch.

## Exact mathematical formulation

For time unit `τ=1 s`, replace the decoder's internal width coordinate with:

`u_w = log(w_sec/τ)`

`w_sec = τ exp(u_w)`

`w_norm = w_sec / D = τ exp(u_w)/D`.

The center remains `c_norm∈(0,1)`. The existing temporal reference and final conversion consume `(c_norm,w_norm)` after this deterministic conversion:

`start_sec = D(c_norm-w_norm/2)` and `end_sec = D(c_norm+w_norm/2)`.

The first intervention should retain the existing span L1/GIoU definitions after conversion, so this pilot isolates the internal width/refinement coordinate rather than simultaneously changing the objective. If a later experiment changes the width loss, it must be registered separately as Family D.

## Affected tensors and entry point

- The second scalar of the decoder reference `(c,w)` becomes `u_w` internally.
- The per-layer width residual is added in `u_w` space.
- `qd_detr_transformer.py` conceptual entry: decoder reference update and the width used by positional modulation; convert `u_w` to `w_norm` only at the existing attention interface.
- `qd_detr.py` conceptual entry: final span tensor is formed from `(c_norm,w_norm)` exactly as before.

No code is written in this stage.

## What stays unchanged

Audio/text features, center coordinate, query count, decoder layers, hidden size, attention, class head, matcher coefficients, GIoU, postprocessing, evaluation, data split, and long-event reporting.

## Mechanism metric

The first metric should be the median `abs(log(w_stage / w_GT))` and median `w_stage/w_GT` at every decoder stage, conditioned on proposals whose initial center is within 1 s of GT, reported separately for 0–2 s and 2–5 s. A successful mechanism effect should appear in the trajectory before headline R1@0.7.

## Falsification condition

In a matched pilot, stop this line if the two short bins do not both show a reduction in the conditioned final log-width residual and final width/GT, or if 20 s+ R1@0.7 falls by at least 5 percentage points without a short-event trajectory improvement. Persistence of the audio-duration slope would also falsify the stronger claim of full length invariance, though it would not by itself falsify the narrower refinement-coordinate hypothesis.

## Prior-art risk

`MODERATE–HIGH`. Log-scale box/segment regression and iterative reference refinement are established patterns. The only defensible claim at this stage is that this is a diagnosis-driven, matched intervention for the observed CASTELLA/QD-DETR scale failure; no novelty or final-method claim is made.

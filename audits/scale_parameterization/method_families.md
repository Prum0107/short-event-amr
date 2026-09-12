# Orthogonal method families

## Frozen baseline geometry

The current model predicts normalized `(c,w)` coordinates. For audio duration `D`:

`start = D(c-w/2)`, `end = D(c+w/2)`, and `width_sec = Dw`.

The decoder update is an inverse-sigmoid residual update in normalized coordinates. The learned initial normalized widths are far larger than the short-event target region, and the optional center intervention changes centers but not this width geometry.

The stress-test CSV uses `D ∈ {60,120,240}s`, `GT width ∈ {1,2,5,20}s`, and an illustrative 40 s initial physical width when an initial width is needed. The 40 s value is a diagnostic reference, not a proposed hyperparameter.

## Family A — width representation

This family changes only the mathematical duration variable. It does not automatically mean that log duration is correct.

### A1: linear absolute seconds

Let `u = w_sec/τ`, `w_sec=τu`, with positive `u` enforced by a positive parameterization such as softplus. The target values for τ=1 s are 1, 2, 5, 20, independent of `D`. The QD-DETR attention interface still receives `w_norm=w_sec/D`, so this formulation removes the regression-space outlier but does not by itself change the attention clock.

### A2: log absolute seconds

Let `u=log(w_sec/τ)` and `w_sec=τ exp(u)`. With τ=1 s, the target values for 1, 2, 5, and 20 s are `0, 0.693, 1.609, 2.996`, independent of audio duration. The reference-point interface converts it to `w_norm=τ exp(u)/D` only when the existing sinusoidal width embedding or final span conversion requires normalized coordinates.

This makes a 40 s → 1 s change `log(1/40)=-3.689`, versus `log(1/5)=-1.609`; the distance is ratio-based and no longer changes merely because `D` changed. It still does not fix a bad physical initial scale by itself.

### A3: log normalized duration

Let `u=log(w_sec/D)` and `w_sec=D exp(u)`. It avoids a hard zero but remains audio-length dependent: a 1 s target is `-4.094`, `-4.787`, and `-5.481` at 60, 120, and 240 s. It is therefore a useful negative control for the claim that any log transform is sufficient.

Compatibility: A1/A2 can be carried through the current center-width reference interface by converting to normalized width at attention and evaluation. A3 is directly compatible but preserves the diagnosed duration interaction.

## Family B — initialization geometry

Use a continuous log-scale coverage prior for initial physical widths, for example `u_0,k = log(s_min)+(k+ξ_k)Δ` with bounded continuous offsets `ξ_k`, then convert `w_0=exp(u_0)` and `w_norm=w_0/D`. The center path is unchanged and no GT duration is used at inference.

The conceptual difference from generic anchors is the mechanism target: the initial references are distributed in a scale coordinate chosen to cover the observed physical-width range, rather than relying on the current static normalized width distribution. It has high prior risk because multi-scale anchors and scale-aware query priors are established patterns.

## Family C — refinement dynamics

Keep the current width variable but predict a multiplicative update:

`w_{l+1}=w_l exp(δ_l)` and `δ_l=log(w_{l+1}/w_l)`.

For 40 s → 1 s the ideal update is `-3.689`; for 5 s → 1 s it is `-1.609`. The corresponding linear physical-width displacements are `-39 s` and `-4 s`. This directly tests whether the decoder's update geometry, rather than its information, limits contraction. If the variable remains normalized, the ratio still contains `D`; using the absolute-time variable from Family A would be a separate combined intervention and is not assumed here.

Compatibility: reference points, query count, attention features, and center branch can remain unchanged; only the width update path changes. Long events are not assigned a separate branch, but exponential updates can be unstable without bounded residuals.

## Family D — optimization geometry

Replace the width component of the regression/matching objective with a relative error, for example:

`e_w = log((w_pred+ε)/(w_gt+ε))`, `L_rel=Huber(e_w)`.

For prediction 5 s versus GT 1 s, `|e_w|=log5=1.609`; for prediction 24 s versus GT 20 s, `|e_w|=log1.2=0.182`. The objective reacts to ratio error rather than only normalized absolute error. A squared log error has derivative `2 log(w_pred/w_gt)/w_pred`, so it can strongly upweight short targets; this is a safety risk, not an automatic benefit.

This family is distinct from Family C: it changes the optimization signal while leaving the reference update law unchanged. GIoU should be retained and reported separately because it already supplies a relative-overlap signal.

## Family E — audio-length-invariant temporal clock

Define a positive effective temporal scale `s(a,q)` from the query-conditioned audio representation, and express both center and width relative to it:

`u_c=(c_sec-μ(a,q))/s(a,q)`, `u_w=w_sec/s(a,q)`;

decode with `c_sec=μ+s u_c` and `w_sec=s u_w`. The invariance requirement is `s(a_D,q)≈s(a_{D'},q)` for the same local event embedded in recordings of different full duration. Full duration then affects only the valid domain and masking, not the numerical scale of the local event.

This does not assume local cropping or coarse-to-fine inference, but it introduces a new scale estimator and changes the temporal clock seen by the decoder. It can cover short and long events if `s` varies continuously; it becomes a short-event bias if `s` is implicitly tied to a hand-coded duration threshold.

## Family F — non-DETR dense boundary control

Predict start and end distributions over the temporal grid, or predict an interior point `p` plus nonnegative distances `d_l,d_r` with interval `[p-d_l,p+d_r]`. The target is expressed in physical seconds or token offsets, not as a learned query width that must contract from a full-recording-normalized prior.

This is a conceptual control, not a recommendation. It changes the object-query and center-width assumptions substantially, so it is less causal as a first intervention for this QD-DETR failure. Dense boundary matching and anchor-free boundary-distance formulations are established prior-art families.

## Stress-test reading guide

- A1/A2 remove `D` from the target duration variable; A3 does not.
- B changes the starting scale but not the decoder update law.
- C changes the update law but does not automatically fix the starting scale.
- D changes gradients/costs but does not change the representation or initialization.
- E changes the temporal clock for both center and width and therefore has the broadest invariance claim.
- F removes the DETR span-reference mechanism entirely and is the strongest architecture-path control.

No family is called novel here. The prior-art precheck is deliberately conservative.

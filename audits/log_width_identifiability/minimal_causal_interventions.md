# Minimal causal interventions

The purpose here is identifiability, not method selection or performance optimization. Each intervention is specified by what changes and what remains fixed.

| intervention | exact variable changed | must remain fixed | cleanly isolatable? | primary mechanism metric | interpretation |
|---|---|---|---|---|---|
| I. Initialization only | `q_w` or its deterministic initialization map, while final coordinate stays normalized | decoder update, attention, final span head, losses, matcher, optimizer, seed/schedule | yes, if the new physical/normalized prior is explicitly declared | initial width-to-GT ratio and layer-0 width residual, stratified by duration and GT duration | tests whether initial scale mismatch accounts for contraction burden; does not test log representation |
| II. Refinement only | normalized layer update `w_{l+1}=exp(log w_l + delta_l)` in place of the current induced logit update | `w_0=sigmoid(q_w)`, center path, attention inputs/outputs in normalized coordinates, final output coordinate, matching/losses | yes, to first order as a refinement-law intervention | layerwise median `|log(w_l/w_GT)|` and multiplicative shrinkage from layer 0 to final; long-event safety | tests decoder contraction geometry without changing the loss or initialization |
| III. Loss only | width term in the final/auxiliary regression loss, with output still normalized | initialization, refinement, attention, final parameterization, matcher unless the loss-only question explicitly includes matching | yes, if matcher is kept unchanged | paired width-loss gradient and GT-vs-pred width error by duration | tests optimization geometry; cannot establish representation causality |
| IV. Audio-length-invariant representation | `u=log(w_sec/tau)` with a stated `w=exp(u)/D` conversion | impossible to hold all of initialization, refinement, attention, final geometry, and normalized loss fixed while also removing `D` | no; partially coupled | normalized-vs-absolute width residual, layerwise `D` slope, plus attention width trajectory | tests a coupled scale package, not a single-variable representation effect |

## Intervention I: initialization only

The clean version changes only the starting normalized width distribution or the query-width initialization tensor. It must not use a GT oracle or a short/long runtime branch. A fixed declared prior, a fixed deterministic transformation of the learned query widths, or a matched checkpoint initialization can be compared. The primary metric should be measured before decoder refinement, then related to the final result; it is not enough to report final Hit@K because the intervention is intended to explain initial geometry.

## Intervention II: refinement only

Use the same normalized initial `w_0`, the same center, the same audio/text features, the same temporal attention input convention, the same final normalized span output, and the same losses/matcher. Change only the width transition from the current logit residual to a predeclared additive log-normalized update. No audio-duration branch is used. The fact that target normalized width still depends on `D` is a retained baseline property and makes the causal question narrow and interpretable.

## Intervention III: loss only

Do not introduce an absolute-seconds output to test loss geometry. Keep the baseline normalized output and replace only a predeclared width regression term. If matching is changed too, the intervention ceases to be loss-only; a separate matching pilot would be needed.

## Intervention IV: invariant representation

An absolute log-seconds coordinate can be made independent of `D` internally, but the official temporal attention and final criterion consume normalized `(center,width)`. Exact baseline initialization needs `u_0(D)`, and original normalized L1 needs `exp(u)/D`. Thus a fully duration-invariant representation requires coordinated changes to at least conversion, initialization, and gradients, and often refinement/attention. It is not cleanly isolatable in the current architecture.

## Global conclusion

I, II, and III have clean narrow versions. IV is only partially identifiable. The previously selected `WIDTH_REPRESENTATION_ONLY` intervention should therefore not be presented as a one-factor causal test without explicitly treating it as a coupled package.

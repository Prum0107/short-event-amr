# Log-width variants and what each one changes

## Baseline coordinate

The baseline predicts normalized width `w in (0,1)` through an inverse-sigmoid residual. Physical width is `w_sec = D w`. The proposed absolute log coordinate is

```text
u = log(w_sec / tau),     tau = 1 s
w_sec = tau exp(u)
w = tau exp(u) / D
```

The unit `tau` only makes the logarithm dimensionless; it is not a learned scale.

## Variant A — representation only, with original normalized geometry

The internal variable is `u`, but every downstream path is reconstructed in the original normalized coordinate:

```text
w(u,D) = tau exp(u)/D
```

The original normalized L1, GIoU, temporal attention width, final output, and matcher are retained after this conversion. This preserves the formulas seen by downstream modules but does not preserve the optimization path with respect to the internal variable. The derivative changes by the chain rule, and an absolute-seconds `u` has a target that is independent of `D` while the reconstructed normalized target is not.

Variant A is therefore not a clean “representation-only” intervention in an optimization experiment. It changes parameterization, initialization unless explicitly remapped, and gradients. It does not remove duration dependence from the normalized L1 term.

## Variant B — representation plus absolute log-width loss

The internal coordinate is still `u`, but the width part of the regression loss is changed to

```text
L_log_width = |u_p-u_g| = |log(w_p_sec/w_g_sec)|
```

Center loss, GIoU, matching, and final geometric conversion may be retained as specified, but the width optimization geometry is now different. This is explicitly a coupled representation-plus-loss intervention. It removes `D` from the width log-error for a fixed physical ratio, but it does not isolate the effect of the coordinate representation.

The GIoU term is still computed from the final normalized span after `w=exp(u)/D`. For the same physical prediction and target, it is unchanged; its gradient with respect to `u` is nevertheless obtained through a different chain rule than its gradient with respect to normalized `w`.

## Variant C — log-space refinement only

Keep the baseline-compatible normalized width as the stored/final geometry and loss coordinate, but use a multiplicative update during decoder refinement:

```text
v_l = log(w_l)
v_{l+1} = v_l + delta_l
tilde_w_{l+1} = exp(v_{l+1})
w_{l+1} = clip(tilde_w_{l+1}, eps, 1-eps)
```

The initial `w_0=sigmoid(q_w)`, attention, final normalized span, GIoU, matcher, and losses stay baseline-compatible. Only the layer-to-layer width update law changes. This is the narrowest practical refinement intervention, but it is not audio-length-invariant: reaching the same physical target requires `delta` that depends on `D` because `w_g=w_g_sec/D`.

The additive log update is not algebraically identical to the current logit update. Exact equivalence would require

```text
delta_l = log(sigmoid(ell(w_l)+Delta_l)) - log(w_l)
```

which is state-dependent and is not a generic constant rescaling of the existing residual. A simple additive log update therefore tests refinement geometry rather than merely renaming a variable.

## Comparison

| variant | internal coordinate | initialization | refinement | attention/final geometry | width loss | duration dependence removed? | identifiability |
|---|---|---|---|---|---|---|---|
| A | absolute `u` | changes unless `u_0(D)` is remapped | changes unless the exact baseline path is replayed | reconstructed baseline normalized path | original normalized L1 through `exp(u)/D` | no | partially coupled |
| B | absolute `u` | changes unless remapped | normally changes | final geometry can be retained | absolute log width | width term only | coupled |
| C | normalized `w`, log update | unchanged | changed | unchanged coordinate and output geometry | unchanged | no | cleanest refinement isolation |

## Short-event interpretation

For a 1-second event in a 240-second audio, the normalized target is approximately `0.00417`; a 1-second event in a 60-second audio is `0.01667`. The same absolute `u_g=log(1)` maps to different normalized targets. Consequently, “absolute log seconds” and “audio-length-invariant downstream geometry” cannot both be obtained merely by choosing `u`; the conversion `exp(u)/D` reintroduces `D` into the normalized model paths.

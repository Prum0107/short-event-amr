# Revised first causal pilot

## Selection

**Recommended first pilot: `REFINEMENT_GEOMETRY_ONLY`.**

The prior `WIDTH_REPRESENTATION_ONLY` choice is revised because an absolute log-seconds coordinate is coupled to initialization, duration conversion, normalized loss gradients, and the attention width trajectory. The refinement-only pilot is the smallest intervention that directly tests the strongest prior mechanism—decoder contraction—while retaining baseline initialization and final geometry.

## Exact intervention

Keep the baseline normalized width `w` and set

```text
w_0 = sigmoid(q_w)
v_0 = log(w_0)
v_{l+1} = v_l + delta_{w,l}
tilde_w_{l+1} = exp(v_{l+1})
w_{l+1} = clip(tilde_w_{l+1}, eps, 1-eps)
```

The clip is a predeclared numerical domain guard required because unconstrained `exp(v)` can exceed the baseline normalized interval. The center update remains the current inverse-sigmoid residual. The width used by temporal sine embedding and modulation is the resulting normalized `w_l`, the final QD-DETR span output remains normalized, `span_cxw_to_xx` and GIoU remain unchanged, and the Hungarian matcher and regression coefficients remain unchanged. The pilot does not convert the internal variable to absolute seconds and does not change the loss.

This changes exactly one mechanism: the decoder's width refinement geometry. It does not claim to remove audio-duration dependence; the normalized target remains `w_GT=w_GT_sec/D`.

## Primary mechanism metric

The primary metric is the paired layerwise multiplicative width residual

```text
R_l = median_q |log(w_{l,q}/w_GT,q)|
```

computed on matched positive query-target pairs and reported separately for `0–2 s`, `2–5 s`, `5–10 s`, and `>=10 s` GT durations. The key contrast is the change from the initial reference to the final reference, `R_final - R_0`, together with the final matched span error. Report audio duration and candidate count as controls. This metric directly measures contraction in relative width units and does not use a GT-based runtime decision.

Secondary safety metrics are the unchanged official localization metrics, with long-event performance reported separately. The pilot should also verify that the initial width distribution is bitwise/within-tolerance identical between arms; otherwise it is not a refinement-only comparison.

## Falsification condition

Falsify the contraction explanation for this pilot if, after confirming identical initialization and unchanged loss/final geometry, the multiplicative update does not reduce the short-event layerwise residual or the final short-event width error relative to the matched baseline, and there is no corresponding improvement in the predeclared short-event localization metric. A material long-event regression under unchanged initialization and loss is a safety failure and also rejects the pilot as a safe general explanation.

Do not declare success from a final score change alone. A final improvement without the predicted layerwise width-residual change is evidence that the intervention affected another pathway or that the mechanism hypothesis was misidentified.

## Why it satisfies the requested criteria

- Diagnosis: directly tests the observed decoder contraction/refinement limitation.
- Minimal mechanisms: changes only the width update law; initialization, output coordinate, attention convention, loss, matching, and coefficients stay fixed.
- Falsifiable: the layerwise relative-width trajectory predicts the direction of change.
- No GT oracle: training/inference has no branch based on GT duration or location.
- No manual short/long threshold: duration bins are evaluation strata fixed before analysis, not runtime control logic.
- Long-event safety: long-event metrics are a prespecified safety slice.
- Interpretability: the intervention is a normalized-coordinate refinement change, not a combined representation/loss/init package.

## Prior-art risk

The pilot is a diagnostic use of standard ideas. Iterative reference refinement is established in DETR-family localization, for example [DAB-DETR](https://arxiv.org/abs/2201.12329), and multiplicative/log parameterizations are standard regression constructions. It is not a novelty claim. Any later method paper would need a separate prior-art audit and ablation matrix.

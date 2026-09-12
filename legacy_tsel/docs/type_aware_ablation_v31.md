# Type-Aware Hard Negative Ablation v3.1

## Question

This ablation was designed around one principle:

> Every change should explain why errors move.

Instead of doing open-ended hyperparameter search, we tested three mechanism-level variants:

1. `semantic-only`: only suppress semantic false peaks.
2. `boundary-only`: only rank GT boundaries above boundary distractors.
3. `semantic strong + boundary weak`: combine both with weaker boundary pressure.

All experiments use the same typed hard negatives from:

`results/typed_hard_negatives_v3/train_typed_hard_negatives.json`

## Configs

| Name | Semantic Loss | Boundary Loss | Purpose |
|---|---:|---:|---|
| semantic-only | 0.25 | 0.00 | Test whether suppressing false semantic peaks improves top1 selection |
| boundary-only | 0.00 | 0.10 | Test whether boundary distractors improve temporal boundary ranking |
| semantic strong + boundary weak | 0.25 | 0.05 | Test whether both mechanisms can combine without conflict |

## Direct Evidence Model

| Model | R1@0.5 | R1@0.7 | R3@0.7 | Best IoU@5 | Evidence Gap |
|---|---:|---:|---:|---:|---:|
| V2 generic HN | 23.30% | 17.05% | 26.42% | 0.4504 | 0.2144 |
| V3 typed | 22.44% | 16.76% | 27.84% | 0.4503 | 0.2131 |
| semantic-only | 22.16% | 16.76% | 25.57% | 0.4489 | 0.2132 |
| boundary-only | 23.01% | 17.61% | 27.84% | 0.4402 | 0.2143 |
| semantic strong + boundary weak | 21.88% | 16.19% | 26.99% | 0.4445 | 0.2125 |

Direct model interpretation:

- `boundary-only` gives the best direct `R1@0.7`.
- `semantic-only` matches V3 on strict top1 but loses top-k.
- The mixed setting is not automatically better.

This supports the idea that boundary pressure directly helps the start/end heads, while semantic pressure mostly changes evidence-driven candidate selection.

## Rule Decoder

| Model | Best Rule | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Semantic Miss |
|---|---|---:|---:|---:|---:|---:|
| semantic-only | peak_drop | 32.95% | 24.15% | 32.95% | 35.51% | 95 |
| boundary-only | dense_contrast | 32.10% | 23.01% | 30.40% | 36.08% | 104 |
| semantic strong + boundary weak | peak_drop | 33.24% | 23.86% | 32.95% | 34.66% | 94 |

Rule decoder interpretation:

- `semantic-only` is best for strict rule-decoder top1.
- `boundary-only` does not reduce semantic misses.
- `semantic strong + boundary weak` slightly reduces semantic misses, but does not beat semantic-only top1.

This suggests semantic pressure makes the evidence curve easier for simple peak-based decoding.

## Learned Decoder

| Model | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Semantic Miss |
|---|---:|---:|---:|---:|---:|---:|
| V2 generic HN | 35.23% | 24.43% | 35.80% | 42.90% | 0.3299 | 84 |
| V3 typed | 35.80% | 24.43% | 36.93% | 41.48% | 0.3345 | 87 |
| semantic-only | 35.23% | 25.28% | 34.38% | 42.33% | 0.3276 | 97 |
| boundary-only | 34.09% | 24.43% | 36.08% | 42.90% | 0.3195 | 98 |
| semantic strong + boundary weak | 34.94% | 24.72% | 37.22% | 42.33% | 0.3257 | 100 |

Learned decoder interpretation:

- `semantic-only` has the best strict `R1@0.7`.
- `semantic strong + boundary weak` has the best `R3@0.7`.
- `boundary-only` preserves high `R5@0.7`, but does not improve strict top1.
- All v3.1 variants have more `semantic_miss` than V2 under this categorization.

This means the category label `semantic_miss` is not equivalent to top1 retrieval quality. The same intervention can improve strict top1 while increasing the number of cases classified as semantic misses.

## Migration From V2

| Target | Semantic Miss Recovered | Good Regressed | Avg Top1 IoU Delta | Avg Top5 IoU Delta |
|---|---:|---:|---:|---:|
| semantic-only | 3 | 15 | -0.0023 | -0.0108 |
| boundary-only | 11 | 15 | -0.0019 | +0.0077 |
| semantic strong + boundary weak | 4 | 20 | -0.0042 | +0.0004 |

Migration interpretation:

- `boundary-only` recovers more V2 semantic misses than semantic-only.
- `semantic-only` still gives the best strict learned-decoder top1.
- The mixed setting causes more category regressions than either single-factor setting.

This is the key mechanism finding:

> Semantic false-peak pressure and boundary pressure are not simply additive. They alter different parts of the retrieval pipeline and can conflict through shared representations.

## Research Conclusion

This is no longer just parameter tuning.

The ablation tells us:

1. Boundary typed loss helps direct start/end prediction and top-k candidate availability.
2. Semantic typed loss helps peak-based/top1 selection more than it reduces the semantic-miss category.
3. Mixing semantic and boundary pressures in the same shared encoder can dilute both effects.

The next research step should not be another scalar weight search. It should change the architecture or training schedule so the two pressures do not fight.

## Follow-up

The proposed two-stage training experiment has now been tested:

`docs/two_stage_semantic_boundary_v1.md`

Result:

- It kept semantic evidence almost unchanged.
- It did not preserve semantic-only's strict top1 advantage.
- It increased semantic misses under the learned decoder.
- It recovered some candidate-level cases, but many did not become correct top1 predictions.

Updated conclusion:

> Boundary knowledge should not live only in separate start/end heads. The decoder needs to read boundary evidence from the temporal shape of semantic evidence.

Recommended next experiment:

### Evidence-shape-aware learned decoder

Use candidate-level features that describe the evidence curve:

- peak height
- peak width
- left/right drop sharpness
- local contrast
- agreement between evidence peaks and start/end logits
- typed hard-negative origin

Alternative:

### Decoupled heads

Use separate adapters for:

- semantic evidence scoring
- boundary refinement

Then type-aware hard negatives update the relevant adapter more than the shared encoder.

This is more research-oriented than tuning loss weights, because it directly tests whether semantic evidence learning and boundary decoding should be disentangled.

# Two-Stage Semantic Evidence Then Boundary Heads v1

## Question

After v3.1, we had a clear mechanism-level observation:

> Semantic false-peak pressure and boundary pressure are not simply additive.

This experiment asks whether a two-stage schedule can reduce that conflict:

1. Start from the best semantic-focused model.
2. Freeze the semantic evidence representation.
3. Train only the boundary heads using typed boundary hard negatives.

The goal is not blind score tuning. The goal is to test whether boundary knowledge can be added without damaging semantic evidence.

## Config

Initial checkpoint:

`results/evidence_baseline_v31_semantic_only_lsem025/best.pt`

Two-stage checkpoint:

`results/evidence_baseline_two_stage_semantic_then_boundary_heads_lbd010/best.pt`

Training mode:

```bash
--init_ckpt_path results/evidence_baseline_v31_semantic_only_lsem025/best.pt
--trainable_parts boundary_heads
--hard_negative_mode type_aware
--lambda_semantic_hard_negative 0.0
--lambda_boundary_hard_negative 0.1
--boundary_hn_margin 0.2
--boundary_hn_topk 3
```

Only `514 / 1,052,675` parameters were trainable. This isolates the boundary-head mechanism.

## Direct Evidence Model

| Model | R1@0.5 | R1@0.7 | R3@0.7 | Best IoU@5 | Evidence Gap |
|---|---:|---:|---:|---:|---:|
| semantic-only | 22.16% | 16.76% | 25.57% | 0.4489 | 0.2132 |
| two-stage boundary heads | 22.44% | 16.76% | 25.57% | 0.4528 | 0.2132 |

Training selected epoch 1 as the best checkpoint by `R1@0.7`.

Interpretation:

- The semantic evidence gap is unchanged, as expected, because the semantic body was frozen.
- Boundary-head training slightly improves loose top1 and best top5 IoU.
- It does not improve strict direct `R1@0.7`.

This means the intervention is clean, but the boundary head alone does not solve strict temporal localization.

## Rule Decoder

Report:

`results/evidence_decoder_experiment_two_stage_semantic_then_boundary_heads_lbd010/report.html`

| Model | Best Rule | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 | Semantic Miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| semantic-only | peak_drop | 32.95% | 24.15% | 32.95% | 35.51% | - | - | 95 |
| two-stage boundary heads | peak_drop | 32.95% | 24.15% | 32.95% | 35.51% | 0.3146 | 0.4868 | 95 |

Interpretation:

- Rule decoding is essentially unchanged.
- This is expected: the rule decoder mainly reads the evidence curve, and the evidence curve did not change.
- Therefore, boundary knowledge stored only in start/end heads is invisible to an evidence-only rule decoder.

## Learned Decoder

Report:

`results/learned_evidence_decoder_two_stage_semantic_then_boundary_heads_lbd010/report.html`

| Model | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 | Semantic Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| V2 generic HN | 35.23% | 24.43% | 35.80% | 42.90% | 0.3299 | - | 84 |
| semantic-only | 35.23% | 25.28% | 34.38% | 42.33% | 0.3276 | - | 97 |
| two-stage boundary heads | 33.81% | 23.86% | 35.23% | 41.76% | 0.3213 | 0.4744 | 101 |

Interpretation:

- Two-stage boundary-head training does not preserve semantic-only's strict top1 advantage.
- It slightly improves `R3@0.7` over semantic-only, but not over V2/V3 typed variants.
- Semantic misses increase from 97 to 101.
- The learned decoder receives more boundary signal, but that signal is not reliably complementary to semantic evidence.

This suggests that adding boundary information through start/end heads alone is too indirect.

## Migration: Semantic-Only to Two-Stage

Report:

`results/decoder_migration_semantic_only_to_two_stage_boundary_heads/report.html`

| Metric | Value |
|---|---:|
| Stable samples | 244 |
| Category improved | 38 |
| Category regressed | 51 |
| IoU improved | 9 |
| IoU regressed | 10 |
| Semantic misses recovered | 10 |
| Semantic miss to good | 1 |
| Good regressed | 21 |
| Avg top1 IoU delta | -0.0062 |
| Avg top5 IoU delta | +0.0074 |
| Avg evidence gap delta | 0.0000 |

Category shift:

| Category | Semantic-Only | Two-Stage |
|---|---:|---:|
| good | 89 | 84 |
| candidate_exists | 60 | 63 |
| boundary_error | 73 | 74 |
| evidence_good_decode_bad | 33 | 30 |
| semantic_miss | 97 | 101 |

Important transitions:

- `semantic_miss -> good`: 1
- `semantic_miss -> boundary_error`: 6
- `semantic_miss -> candidate_exists`: 3
- `good -> candidate_exists`: 20
- `boundary_error -> semantic_miss`: 11

Interpretation:

Two-stage training does recover some misses, but many recovered cases stop at candidate-level improvement instead of becoming correct predictions. At the same time, many previously good samples become only candidate-exists cases.

So the change improves some candidate availability, but weakens final top1 selection.

## Migration: V2 to Two-Stage

Report:

`results/decoder_migration_v2_to_two_stage_boundary_heads/report.html`

| Metric | Value |
|---|---:|
| Stable samples | 215 |
| Category improved | 48 |
| Category regressed | 70 |
| IoU improved | 9 |
| IoU regressed | 10 |
| Semantic misses recovered | 8 |
| Semantic miss to good | 0 |
| Good regressed | 26 |
| Avg top1 IoU delta | -0.0086 |
| Avg top5 IoU delta | -0.0034 |
| Avg evidence gap delta | -0.0013 |

Category shift:

| Category | V2 | Two-Stage |
|---|---:|---:|
| good | 86 | 84 |
| candidate_exists | 65 | 63 |
| boundary_error | 89 | 74 |
| evidence_good_decode_bad | 28 | 30 |
| semantic_miss | 84 | 101 |

Interpretation:

Compared with V2, two-stage reduces boundary-error count but increases semantic-miss count substantially. This is not a good trade-off for the current architecture.

## Mechanism Conclusion

This experiment answers the two-stage question clearly:

> Freezing semantic evidence and tuning only boundary heads is not enough to combine the best parts of semantic and boundary learning.

The reason is structural:

1. The evidence curve still carries the semantic alignment signal.
2. The boundary heads carry local boundary preference.
3. The current decoder does not have a strong enough mechanism to reconcile those two signals when they disagree.

In other words, the problem is no longer only representation learning. It is also evidence decoding.

## Next Direction

The next main direction should be:

### Evidence-shape-aware learned decoding

Instead of only asking the base model to produce better start/end heads, build a decoder that explicitly uses:

- evidence peak height
- evidence peak width
- left/right drop sharpness
- contrast against nearby windows
- agreement between evidence peaks and start/end logits
- typed hard-negative origin of each candidate

Hypothesis:

> Boundary knowledge should be decoded from the temporal shape of semantic evidence, not only injected into separate start/end heads.

This keeps our research direction coherent:

`semantic evidence -> candidate explanation -> boundary-aware decoding`

The next experiment should therefore improve the learned decoder feature set before doing more model-level hard-negative tuning.

Follow-up tested:

`docs/evidence_shape_decoder_v4.md`

The first decoder-side result supports this direction: evidence-shape-aware reranking with a compact candidate set improves strict top1 and reduces semantic misses without retraining the evidence model.

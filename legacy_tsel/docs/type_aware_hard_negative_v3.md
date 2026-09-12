# Type-Aware Hard Negative Training v3

## Purpose

V2 used one generic hard-negative loss: any high-evidence wrong window was pushed below the GT window.

V3 asks a more specific question:

> Should different wrong windows receive different training pressure?

The answer should be yes. A totally wrong semantic peak should be suppressed, but a partially correct boundary window should not be treated as fully wrong. It should teach the model sharper boundary selection.

## Typed Hard Negative Mining

Input evidence:

`results/evidence_baseline_v2_hn_l02/full_train_eval/predictions_evidence_samples.json`

Output:

`results/typed_hard_negatives_v3/train_typed_hard_negatives.json`

Script:

`src/mine_typed_hard_negatives.py`

Types:

- `semantic_false_peak`: IoU `< 0.1`
- `boundary_distractor`: `0.1 <= IoU < 0.7`
- `over_wide`: covers most of GT but is too long
- `under_wide`: mostly inside GT but too short

Mining summary:

| Type | Count | Samples | Avg Score | Avg IoU | Max IoU |
|---|---:|---:|---:|---:|---:|
| semantic_false_peak | 6533 | 2180 | 0.6696 | 0.0117 | 0.0990 |
| boundary_distractor | 3389 | 1406 | 0.6920 | 0.3656 | 0.6970 |
| over_wide | 3416 | 1818 | 0.6542 | 0.4053 | 0.6977 |
| under_wide | 2500 | 1335 | 0.7196 | 0.3449 | 0.6923 |

The distribution is healthy: all four types have enough examples to support training.

## Training Change

New type-aware loss:

- Semantic false peaks use evidence ranking:
  `GT evidence > semantic false peak evidence`
- Boundary distractors / over-wide / under-wide windows use boundary ranking:
  `GT start/end logits > candidate start/end logits`

Training command used:

```bash
--hard_negative_mode type_aware
--lambda_semantic_hard_negative 0.15
--lambda_boundary_hard_negative 0.1
--semantic_hn_margin 0.1
--boundary_hn_margin 0.2
```

Checkpoint:

`results/evidence_baseline_v3_typed_hn_lsem015_lbd010/best.pt`

## Direct Evidence Model Result

| Model | R1@0.5 | R1@0.7 | R3@0.7 | Best IoU@5 | Evidence Gap |
|---|---:|---:|---:|---:|---:|
| V1 evidence | 23.01% | 17.33% | 25.85% | 0.4578 | 0.2178 |
| V2 generic HN | 23.30% | 17.05% | 26.42% | 0.4504 | 0.2144 |
| V3 typed HN | 22.44% | 16.76% | 27.84% | 0.4503 | 0.2131 |

Direct top1 is slightly lower, but top3 recall is higher. This suggests typed negatives may improve the candidate set while not yet solving top1 selection.

## Rule Decoder Result

Report:

`results/evidence_decoder_experiment_v3_typed_hn_lsem015_lbd010/report.html`

| Model | Best Rule | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| V2 generic HN | peak_drop | 32.67% | 23.01% | 32.10% | 34.09% | 0.3130 | 0.4871 |
| V3 typed HN | peak_drop | 31.53% | 21.59% | 34.09% | 36.36% | 0.2978 | 0.4894 |

Rule decoding again shows the same pattern:

- top1 gets worse
- top3/top5 get better
- best candidate quality stays strong

So V3 seems to produce a richer candidate list, but peak_drop is not the right selector.

## Learned Decoder Result

Report:

`results/learned_evidence_decoder_v3_typed_hn_lsem015_lbd010/report.html`

| Model | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 |
|---|---:|---:|---:|---:|---:|---:|
| V1 learned | 33.24% | 23.58% | 34.66% | 39.20% | 0.3131 | 0.4537 |
| V2 learned | 35.23% | 24.43% | 35.80% | 42.90% | 0.3299 | 0.4778 |
| V3 learned | 35.80% | 24.43% | 36.93% | 41.48% | 0.3345 | 0.4695 |

V3 does not beat V2 on strict `R1@0.7`, but it improves:

- `R1@0.5`: 35.23% -> 35.80%
- `R3@0.7`: 35.80% -> 36.93%
- `Top1 IoU`: 0.3299 -> 0.3345

This is consistent with the candidate-quality hypothesis.

## V2 to V3 Migration

Report:

`results/decoder_migration_v2_to_v3_typed/report.html`

Key migration counts:

| Status | Count |
|---|---:|
| stable | 231 |
| category_improved | 45 |
| category_regressed | 48 |
| iou_improved | 19 |
| iou_regressed | 9 |

Category counts:

| Category | V2 | V3 |
|---|---:|---:|
| good | 86 | 86 |
| candidate_exists | 65 | 60 |
| boundary_error | 89 | 88 |
| evidence_good_decode_bad | 28 | 31 |
| semantic_miss | 84 | 87 |

Semantic miss migration:

- V2 semantic misses: 84
- recovered by V3: 14
- semantic miss -> boundary error: 11
- semantic miss -> evidence_good_decode_bad: 3

Regression to semantic miss:

- boundary error -> semantic miss: 14
- good -> semantic miss: 1
- evidence_good_decode_bad -> semantic miss: 2

This is not a clean semantic improvement over V2. V3 improves some cases but also loses others.

## Interpretation

V3 is scientifically useful but not yet a clear replacement for V2.

The evidence says:

1. Type-aware negatives improve candidate richness and average localization.
2. They do not yet improve strict top1 correctness.
3. Boundary-specific pressure may be helping top-k candidate quality.
4. The semantic false-peak pressure is probably too weak or not selective enough, because `semantic_miss` rises from 84 to 87.

The most important lesson:

> Type-aware hard negatives are promising, but the loss weights need to be separated more carefully.

## Next V3.1

Recommended next experiment:

- Keep boundary typed loss.
- Increase semantic false-peak pressure slightly.
- Reduce boundary pressure if top1 keeps becoming unstable.

Suggested config:

```bash
--lambda_semantic_hard_negative 0.25
--lambda_boundary_hard_negative 0.05
```

Hypothesis:

This should preserve the top-k boundary/candidate benefits while reducing the semantic regressions.

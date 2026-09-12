# Temporal Evidence Hard Negative V4: Semantic to Width Fusion

## Goal

V3 showed that boundary hard-negative gains mainly come from width correction:

```text
over_wide + under_wide
```

V4 asks whether semantic false peaks can be used safely as a light auxiliary or
curriculum before width correction. The target is not only a better adapter
decoder, but also a better candidate-level fusion result against the previous
Fusion V1 reference.

## Stage Summary

Stage 1 already established:

```text
boundary_distractor only is weak
width-only is the useful boundary subtype
boundary_distractor -> width staging hurts
```

Stage 2 tested semantic and width schedules:

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V3 width-only | 39.49 | 29.83 | 40.34 | 47.16 | 0.3787 | 0.5402 | 63 | 105 |
| V4 semantic+width auxiliary | 38.64 | 29.83 | 41.19 | 46.88 | 0.3749 | 0.5399 | 64 | 105 |
| V4 semantic->width staged | 39.20 | 30.11 | 41.48 | 46.31 | 0.3799 | 0.5459 | 60 | 106 |

Conclusion:

```text
Semantic false peaks are useful as a short warmup before width correction,
not as a simultaneous auxiliary loss.
```

The staged schedule gives the best adapter strict top1, the best top-k IoU, and
the lowest semantic miss among width-focused variants.

## Fusion Setup

Candidate-level fusion keeps the MS-CLAP branch fixed:

```text
results/evidence_baseline_release_v1/full_train_eval/predictions_evidence_samples.json
results/evidence_baseline_release_v1/full_val_eval/predictions_evidence_samples.json
results/learned_evidence_decoder_shape_v2_semantic_only_top2/best.pt
```

The adapter branch is replaced with V4 staged semantic->width:

```text
results/temporal_evidence_hn_v4_stage_semantic_width/full_train_eval/predictions_evidence_samples.json
results/temporal_evidence_hn_v4_stage_semantic_width/full_val_eval/predictions_evidence_samples.json
results/learned_evidence_decoder_temporal_hn_v4_stage_semantic_width_shape_v2_top2/best.pt
```

Main output:

```text
results/candidate_level_representation_fusion_temporal_hn_v4_stage_semantic_width
```

## Fusion Result

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate-level fusion V1 | 41.48 | 30.97 | 40.91 | 47.44 | 0.3935 | 0.5624 | 76 | 109 |
| Width-only candidate fusion | 41.19 | 31.53 | 41.19 | 48.01 | 0.3847 | 0.5447 | 78 | 111 |
| V4 semantic->width fusion | 42.90 | 33.24 | 42.61 | 50.57 | 0.4081 | 0.5721 | 72 | 117 |

V4 is the first fusion result that improves all of the following over Fusion V1:

```text
strict R1@0.7
top1 IoU
top5 IoU
semantic miss
good case count
```

## Five-Seed Fusion Check

Seeds:

```text
2026, 2027, 2028, 2029, 2030
```

| Seed | Best Epoch | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 15 | 42.90 | 33.24 | 42.61 | 50.57 | 0.4081 | 0.5721 | 72 | 117 |
| 2027 | 8 | 41.19 | 32.10 | 40.62 | 49.43 | 0.3920 | 0.5560 | 83 | 113 |
| 2028 | 9 | 43.47 | 31.82 | 43.18 | 48.86 | 0.4080 | 0.5741 | 76 | 112 |
| 2029 | 7 | 40.91 | 32.10 | 42.05 | 50.57 | 0.4006 | 0.5596 | 74 | 113 |
| 2030 | 8 | 42.05 | 31.82 | 40.06 | 45.45 | 0.3941 | 0.5409 | 72 | 112 |

Mean and population standard deviation:

```text
mean R1@0.7 = 32.22
std  R1@0.7 = 0.53
mean R1@0.5 = 42.10
mean top1 IoU = 0.4006
mean top5 IoU = 0.5605
mean semantic miss = 75.4
mean good = 113.4
```

Every seed is above Candidate-level fusion V1's `R1@0.7=30.97`.

## Migration Against Fusion V1

Best seed V4 staged fusion vs Candidate-level fusion V1:

```text
strict gains = 23
strict losses = 15
net strict = +8
avg top1 IoU delta = +0.0147
avg top5 IoU delta = +0.0097
```

Strict gains by previous category:

| Previous Category | Gains |
|---|---:|
| candidate_exists | 14 |
| boundary_error | 7 |
| semantic_miss | 2 |

Losses:

```text
good -> non-good: 15
```

Interpretation:

- The staged evidence improves both candidate availability usage and boundary
  correction.
- Unlike width-only fusion, this is not just a strict-threshold win; the best
  seed also improves average top1 and top5 IoU.
- Some reliable good cases still regress, so risk calibration remains relevant.

## Frozen Test Evaluation

The validation-selected V4 fusion checkpoint was evaluated on:

```text
data/castella_test_release.jsonl
```

using the frozen MS decoder, frozen V4 adapter decoder, and frozen candidate
fusion checkpoint. No training or model selection was performed on the test
split.

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 23.83 | 16.11 | 26.06 | 31.03 | 0.2436 | 0.4064 | 414 | 217 |
| V4 adapter shape_v2 top2 | 33.04 | 21.97 | 34.08 | 40.16 | 0.3196 | 0.4892 | 239 | 296 |
| V4 candidate-level fusion | 34.82 | 24.28 | 36.30 | 43.43 | 0.3423 | 0.5220 | 307 | 327 |
| Merged candidate oracle | 66.07 | 54.49 | 54.49 | 54.49 | 0.6293 | 0.6293 | 78 | 734 |

The absolute test scores are lower than validation, but the ordering transfers:

```text
MS-CLAP < V4 adapter < V4 candidate-level fusion
```

The frozen fusion improves strict test `R1@0.7` by `+8.17` points over MS-CLAP
and `+2.30` points over the V4 adapter alone.

Five-seed frozen test:

| System | R1@0.5 | R1@0.7 | mAP(avg) | mAP@0.5 | mAP@0.75 | Top1 IoU |
|---|---:|---:|---:|---:|---:|---:|
| V4 candidate-level fusion mean+-std | 34.65+-0.45 | 23.42+-0.60 | 18.66+-0.31 | 28.15+-0.41 | 18.98+-0.35 | 0.3359+-0.0045 |

Against the stronger released baseline
`Clotho-Moment pretrain + CASTELLA finetune`, the five-seed gains are:

```text
R1@0.5  +8.79
R1@0.7  +9.57
mAP(avg) +6.92
mAP@0.5  +5.01
mAP@0.75 +8.44
```

Validation-selected conservative gates did not transfer on test:

| System | R1@0.7 | Top1 IoU | Changed | Improved | Regressed | Intervention Precision |
|---|---:|---:|---:|---:|---:|---:|
| Ungated candidate fusion | 24.28 | 0.3423 | 1212 | 577 | 283 | 0.476 |
| Balanced gate | 21.16 | 0.2939 | 391 | 243 | 103 | 0.621 |
| Precision gate | 16.56 | 0.2480 | 41 | 29 | 12 | 0.707 |

The gates are cleaner but too conservative under test shift. The next selection
problem is distribution-robust intervention, not another validation-only gate
threshold.

Full test report:

```text
docs/test_temporal_evidence_hn_v4_semantic_width_fusion.md
results/test_candidate_level_fusion_temporal_hn_v4_stage_semantic_width/summary.json
```

## Decision

V4 semantic->width staged evidence is now the active main branch.

The current best validation result is:

```text
V4 semantic->width candidate fusion
best seed R1@0.7 = 33.24
five-seed mean R1@0.7 = 32.22
frozen test best checkpoint R1@0.7 = 24.28
frozen test five-seed R1@0.7 = 23.42+-0.60
frozen test five-seed mAP(avg) = 18.66+-0.31
```

Next work should not go back to broad hard-negative mining yet. The next useful
step is to make candidate selection robust under validation-to-test shift:

```text
reduce good-case regressions
keep the width-correction gains
preserve the semantic miss improvement
close the merged-candidate oracle gap
```

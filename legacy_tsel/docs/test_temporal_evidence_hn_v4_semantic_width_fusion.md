# Test Evaluation: Temporal Evidence HN V4 Semantic to Width Fusion

## Setup

This is a frozen test evaluation of the current best validation branch:

```text
Temporal Evidence HN V4 semantic false-peak warmup -> width correction
```

The test split is:

```text
data/castella_test_release.jsonl
```

Number of test queries:

```text
1347
```

The evaluated fusion checkpoint is the already-selected validation checkpoint:

```text
results/candidate_level_representation_fusion_temporal_hn_v4_stage_semantic_width/best.pt
```

The test script does not train or reselect on the test split. It applies the
frozen MS-CLAP decoder, the frozen V4 adapter decoder, and the frozen
candidate-level fusion checkpoint.

Output:

```text
results/test_candidate_level_fusion_temporal_hn_v4_stage_semantic_width/summary.json
```

## Main Test Result

Single selected checkpoint:

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 23.83 | 16.11 | 26.06 | 31.03 | 0.2436 | 0.4064 | 414 | 217 |
| V4 adapter shape_v2 top2 | 33.04 | 21.97 | 34.08 | 40.16 | 0.3196 | 0.4892 | 239 | 296 |
| Frozen candidate-level fusion | 34.82 | 24.28 | 36.30 | 43.43 | 0.3423 | 0.5220 | 307 | 327 |

Official-style metrics for the same checkpoint:

| System | R1@0.5 | R1@0.7 | mAP(avg) | mAP@0.5 | mAP@0.75 |
|---|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 23.83 | 16.11 | 12.77 | 20.20 | 12.91 |
| V4 adapter shape_v2 top2 | 33.04 | 21.97 | 17.37 | 26.62 | 17.67 |
| Frozen candidate-level fusion | 34.82 | 24.28 | 19.11 | 28.55 | 19.65 |

The ordering from validation holds on test:

```text
MS-CLAP < V4 temporal adapter < candidate-level fusion
```

Strict top1 gains:

```text
adapter over MS-CLAP = +5.86 R1@0.7 points
fusion over adapter = +2.30 R1@0.7 points
fusion over MS-CLAP = +8.17 R1@0.7 points
```

## Five-Seed Frozen Test

The five validation-selected fusion checkpoints were evaluated on test without
test-time tuning:

| Seed | Best Epoch | R1@0.5 | R1@0.7 | mAP(avg) | mAP@0.5 | mAP@0.75 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 15 | 34.82 | 24.28 | 19.11 | 28.55 | 19.65 | 0.3423 | 0.5220 | 307 | 327 |
| 2027 | 8 | 34.00 | 23.46 | 18.21 | 27.51 | 18.63 | 0.3311 | 0.4911 | 323 | 316 |
| 2028 | 9 | 35.34 | 23.83 | 18.88 | 28.64 | 18.94 | 0.3402 | 0.5237 | 316 | 321 |
| 2029 | 7 | 34.74 | 22.72 | 18.59 | 28.08 | 18.84 | 0.3336 | 0.5070 | 321 | 306 |
| 2030 | 8 | 34.37 | 22.79 | 18.52 | 27.97 | 18.84 | 0.3324 | 0.5111 | 288 | 307 |
| Mean+-std | - | 34.65+-0.45 | 23.42+-0.60 | 18.66+-0.31 | 28.15+-0.41 | 18.98+-0.35 | 0.3359+-0.0045 | 0.5110+-0.0118 | 311.0+-12.8 | 315.4+-8.1 |

Against the released CASTELLA score statistics:

| System | R1@0.5 | R1@0.7 | mAP(avg) | mAP@0.5 | mAP@0.75 |
|---|---:|---:|---:|---:|---:|
| Only CASTELLA baseline | 22.74+-0.77 | 10.17+-0.86 | 10.49+-0.53 | 21.93+-0.57 | 8.85+-0.58 |
| Clotho-Moment pretrain + CASTELLA finetune baseline | 25.86+-0.74 | 13.85+-1.47 | 11.74+-0.39 | 23.14+-0.33 | 10.54+-0.54 |
| V4 candidate-level fusion | 34.65+-0.45 | 23.42+-0.60 | 18.66+-0.31 | 28.15+-0.41 | 18.98+-0.35 |

Five-seed gains over the stronger baseline:

```text
R1@0.5  +8.79
R1@0.7  +9.57
mAP(avg) +6.92
mAP@0.5  +5.01
mAP@0.75 +8.44
```

This resolves the earlier single-checkpoint caveat: the final claim can be made
with five-seed mean+-std, not only a lucky checkpoint.

## Validation to Test Drop

The absolute test numbers are much lower than validation:

| System | Validation R1@0.7 | Test R1@0.7 | Drop |
|---|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 25.85 | 16.11 | -9.74 |
| V4 adapter shape_v2 top2 | 30.11 | 21.97 | -8.14 |
| V4 candidate-level fusion | 33.24 | 24.28 | -8.96 |

Interpretation:

- The release test split is harder or distribution-shifted relative to
  validation.
- The research direction still transfers because the relative improvements
  remain visible.
- Future reports should show validation and test separately. The validation
  result should not be presented as the final expected performance.

## Gate Check

The validation-selected gate parameters did not transfer well:

| System | R1@0.7 | Top1 IoU | Semantic Miss | Changed | Improved | Regressed | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen candidate-level fusion | 24.28 | 0.3423 | 307 | 1212 | 577 | 283 | 0.476 |
| Balanced gate | 21.16 | 0.2939 | 393 | 391 | 243 | 103 | 0.621 |
| Precision gate | 16.56 | 0.2480 | 414 | 41 | 29 | 12 | 0.707 |

The gates are cleaner in intervention precision, but they give up too many
useful adapter/fusion changes. For this test split, the ungated frozen
candidate-level fusion is the best submitted-style system among these options.

## Fusion Case Analysis

Best-checkpoint fusion case analysis:

```text
changed = 1212
improved = 577
regressed = 283
same = 352
good_regressed = 57
semantic_recovered = 170
intervention_precision = 0.476
```

Semantic-temporal role decomposition:

| Role | Count | Improved | Regressed | Same | Avg Top1 IoU Delta | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| semantic_recovery | 170 | 152 | 0 | 18 | +0.3473 | MS semantic miss is moved into a plausible temporal region |
| temporal_correction_to_good | 136 | 136 | 0 | 0 | +0.5909 | Non-good MS case becomes strict-good |
| temporal_refinement | 224 | 224 | 0 | 0 | +0.2126 | Temporal IoU improves without necessarily crossing to good |
| anchor_refinement | 65 | 65 | 0 | 0 | +0.1167 | Already-good MS anchor is improved |
| anchor_regression | 57 | 0 | 57 | 0 | -0.4790 | Already-good MS anchor is replaced badly |
| anchor_weakening | 38 | 0 | 38 | 0 | -0.0819 | Still good, but a reliable anchor is weakened |
| semantic_regression | 62 | 0 | 62 | 0 | -0.1175 | Intervention loses semantic grounding |
| temporal_regression | 126 | 0 | 126 | 0 | -0.1903 | Intervention worsens temporal localization |

This is the current semantic-temporal framing:

```text
positive evidence = semantic recovery + temporal correction/refinement
risk evidence = anchor regression/weakening + semantic/temporal regression
```

By previous MS-CLAP category:

| MS Category | Changed | Improved | Regressed | Same | Avg Top1 IoU Delta | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| semantic_miss | 409 | 152 | 0 | 257 | +0.1444 | Fusion safely recovers many missed semantic cases |
| boundary_error | 320 | 169 | 146 | 5 | +0.0772 | Mixed boundary correction; needs risk control |
| candidate_exists | 183 | 128 | 42 | 13 | +0.2274 | Strong benefit when MS has a useful non-top candidate |
| evidence_good_decode_bad | 138 | 63 | 0 | 75 | +0.2205 | Decoder-level errors are often repairable |
| good | 162 | 65 | 95 | 2 | -0.1410 | Main regression source |

Most useful transitions:

| Transition | Count | Improved | Avg Top1 IoU Delta |
|---|---:|---:|---:|
| candidate_exists -> good | 79 | 79 | +0.5122 |
| semantic_miss -> boundary_error | 100 | 100 | +0.2362 |
| semantic_miss -> good | 31 | 31 | +0.8627 |
| boundary_error -> good | 38 | 38 | +0.6283 |
| evidence_good_decode_bad -> good | 19 | 19 | +0.8434 |

Main harmful transitions:

| Transition | Count | Regressed | Avg Top1 IoU Delta |
|---|---:|---:|---:|
| good -> candidate_exists | 50 | 50 | -0.4649 |
| boundary_error -> semantic_miss | 59 | 59 | -0.1179 |
| boundary_error -> evidence_good_decode_bad | 20 | 20 | -0.2603 |
| good -> boundary_error | 6 | 6 | -0.5468 |

Candidate-source signal:

| Source | Changed | Improved | Regressed | Avg Top1 IoU Delta |
|---|---:|---:|---:|---:|
| current_start_end | 238 | 140 | 46 | +0.1625 |
| dense_contrast | 185 | 97 | 49 | +0.1756 |
| threshold_p70 | 267 | 126 | 60 | +0.1018 |
| peak_drop | 180 | 83 | 40 | +0.1005 |
| threshold_p60 | 112 | 27 | 35 | -0.0361 |

Interpretation:

- The adapter/fusion branch is valuable for recovering semantic misses and
  candidate-exists cases.
- The main risk is replacing a reliable MS top1 window that was already good.
- `threshold_p60` is the clearest weak candidate source under test shift.
- The next selector should protect strong MS anchors and downweight weak
  threshold-style interventions, rather than simply reducing all adapter usage.

Detailed semantic-temporal analysis:

```text
results/test_candidate_level_fusion_temporal_hn_v4_stage_semantic_width/semantic_temporal_analysis.md
results/test_candidate_level_fusion_temporal_hn_v4_stage_semantic_width/semantic_temporal_analysis.json
```

Source/confidence rule probe:

```text
results/test_candidate_level_fusion_temporal_hn_v4_stage_semantic_width/source_rule_probe.json
```

The best validation rule did not block any candidate source and was effectively
the full fusion policy. This means simple source filtering is not enough. The
next selector needs to learn semantic recovery, temporal correction, and anchor
risk jointly.

## Candidate Ceiling

Candidate availability remains much higher than current selection quality:

| Oracle | R1@0.7 | Top1 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|
| MS candidate oracle | 38.01 | 0.4826 | 184 | 512 |
| Adapter candidate oracle | 47.66 | 0.5649 | 125 | 642 |
| Merged candidate oracle | 54.49 | 0.6293 | 78 | 734 |

This supports the same bottleneck diagnosis as validation:

```text
the candidate set is strong enough; the main unsolved problem is reliable
candidate selection under distribution shift.
```

## Decision

Keep V4 semantic->width evidence as the current main research branch.

For test-style performance, use:

```text
Frozen candidate-level fusion
best checkpoint R1@0.7 = 24.28
five-seed mean R1@0.7 = 23.42+-0.60
five-seed mean mAP(avg) = 18.66+-0.31
```

Next work should not discard temporal evidence hard negatives. The more useful
next step is distribution-robust candidate selection:

```text
reduce over-intervention and validation-specific gating
keep the adapter's semantic coverage advantage
close the large gap to the merged candidate oracle
```

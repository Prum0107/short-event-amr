# Semantic-Temporal Candidate Selector V1

## Goal

This experiment continues the Temporal Semantic Evidence Learning direction:

```text
semantic false peak and temporal boundary error should be modeled separately.
```

V1 tests whether a default-anchored selector can decide when the frozen V4
candidate fusion should override the MS-CLAP default by predicting explicit
semantic-temporal roles:

```text
positive heads:
semantic_recovery
temporal_positive

risk heads:
anchor_risk
semantic_risk
temporal_risk

utility head:
intervention_utility
```

The selector is trained on train, tuned on validation, and then the best
validation seed is evaluated frozen on test.

## Validation Result

Reference systems:

| System | R1@0.5 | R1@0.7 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 34.09 | 25.85 | 0.3271 | 0.4887 | 85 | 91 |
| V4 adapter shape_v2 top2 | 39.20 | 30.11 | 0.3799 | 0.5459 | 60 | 106 |
| Frozen V4 candidate fusion | 42.90 | 33.24 | 0.4081 | 0.5721 | 72 | 117 |

Semantic-temporal selector:

| System | R1@0.5 | R1@0.7 | Top1 IoU | Top5 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|
| Balanced selector best seed | 42.05 | 32.95 | 0.3980 | 0.5428 | 71 | 116 |
| Balanced selector five-seed mean | 41.19 | 31.31 | 0.3864 | 0.5354 | - | - |
| Precision selector five-seed mean | 36.42 | 27.10 | 0.3456 | 0.5044 | - | - |

Best-seed intervention quality:

| System | Changed | Improved | Regressed | Semantic Recovered | Good Regressed | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Frozen V4 candidate fusion | 305 | 142 | 83 | 28 | 24 | 0.466 |
| Balanced selector best seed | 145 | 85 | 37 | 17 | 14 | 0.586 |
| Precision selector best seed | 36 | 24 | 5 | 6 | 4 | 0.667 |

Validation interpretation:

- The selector is cleaner than full fusion.
- It does not beat full fusion strict top1.
- It gives up too many useful temporal-positive interventions.

## Frozen Test Result

The best validation seed is:

```text
seed = 2030
checkpoint = results/semantic_temporal_candidate_selector_v1_fast/selector_seed2030.pt
```

Official-style test metrics:

| System | R1@0.5 | R1@0.7 | mAP(avg) | mAP@0.5 | mAP@0.75 |
|---|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 23.83 | 16.11 | 12.77 | 20.20 | 12.91 |
| V4 adapter shape_v2 top2 | 33.04 | 21.97 | 17.37 | 26.62 | 17.67 |
| Frozen V4 candidate fusion | 34.82 | 24.28 | 19.11 | 28.55 | 19.65 |
| Balanced selector | 32.59 | 22.49 | 17.62 | 26.79 | 18.21 |
| Precision selector | 28.06 | 19.23 | 15.04 | 23.58 | 15.51 |

Test intervention quality:

| System | Changed | Improved | Regressed | Semantic Recovered | Good Regressed | Precision |
|---|---:|---:|---:|---:|---:|---:|
| Frozen V4 candidate fusion | 1212 | 577 | 283 | 170 | 57 | 0.476 |
| Balanced selector | 590 | 350 | 136 | 97 | 36 | 0.593 |
| Precision selector | 135 | 94 | 16 | 46 | 4 | 0.696 |

Test interpretation:

- The balanced selector is more conservative and cleaner than full fusion.
- It remains above the adapter-only branch at strict `R1@0.7`, but it is below
  full fusion.
- The precision selector protects anchors, but discards too many useful
  semantic/temporal interventions.

## Decision

V1 is a useful partial result, not the new main system.

The positive result:

```text
explicit semantic/temporal/risk heads improve intervention cleanliness.
```

The negative result:

```text
a post-hoc gate after candidate fusion loses too many temporal-positive cases.
```

Next step:

```text
move semantic_recovery / temporal_positive / anchor_risk modeling into the
candidate scorer itself, instead of applying it only as a late gate.
```

This keeps the project aligned with the main framework:

```text
Temporal Semantic Evidence Learning
= learn semantic evidence + temporal boundary evidence + reliability evidence
```

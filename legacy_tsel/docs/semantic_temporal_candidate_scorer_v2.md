# Semantic-Temporal Candidate Scorer V2

## Goal

This experiment turns the semantic-temporal evidence idea into an actual
candidate-selection box:

```text
candidate windows
-> semantic evidence heads
-> temporal evidence heads
-> anchor/risk evidence heads
-> selected windows with explanations
```

The key change from `Semantic-Temporal Candidate Selector V1` is that V1 was a
late gate after candidate fusion, while V2 puts the evidence heads inside the
candidate scorer itself.

## Evidence Heads

For every MS-CLAP / temporal-adapter candidate, the model predicts:

```text
quality_head
anchor_reliability
semantic_gain
temporal_gain
strict_gain
anchor_risk
semantic_risk
temporal_risk
utility
anchor_guard
```

Interpretation:

- `semantic_gain`: candidate may recover an MS semantic miss.
- `temporal_gain`: candidate may fix boundary/candidate localization.
- `anchor_reliability`: MS anchor looks trustworthy.
- `anchor_guard`: candidate disagrees with a reliable MS anchor.
- risk heads explain why a candidate should be down-weighted.

## Validation Results

Reference:

| System | R1@0.5 | R1@0.7 | R5@0.7 | Top1 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 34.09 | 25.85 | 42.61 | 0.3271 | 85 | 91 |
| V4 adapter shape_v2 top2 | 39.20 | 30.11 | 46.31 | 0.3799 | 60 | 106 |
| V4 candidate-level fusion | 42.90 | 33.24 | 50.57 | 0.4081 | 72 | 117 |

V2 variants:

| System | R1@0.5 | R1@0.7 | R5@0.7 | Top1 IoU | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|
| V2 hybrid evidence score | 43.18 | 31.53 | 49.72 | 0.4013 | 66 | 111 |
| V2 hybrid + anchor guard | 41.76 | 31.53 | 49.43 | 0.3941 | 60 | 111 |
| V2 quality-dominant hybrid | 42.05 | 31.53 | 49.15 | 0.3999 | 63 | 111 |
| V2 quality_guard + fusion target | 43.18 | 32.39 | 51.14 | 0.4060 | 67 | 114 |
| V2 quality_guard + fusion target, 5-seed mean | 42.56+-1.28 | 32.27+-0.93 | 50.11+-0.98 | 0.4070+-0.0094 | 70.8+-3.7 | 113.6+-3.3 |

Intervention quality:

| System | Changed | Improved | Regressed | Semantic Recovered | Good Regressed | Precision |
|---|---:|---:|---:|---:|---:|---:|
| V4 candidate-level fusion | 305 | 142 | 83 | 28 | 24 | 0.466 |
| V2 hybrid evidence score | 333 | 156 | 110 | 39 | 32 | 0.468 |
| V2 hybrid + anchor guard | 319 | 153 | 101 | 42 | 32 | 0.480 |
| V2 quality_guard + fusion target | 290 | 137 | 85 | 36 | 22 | 0.472 |
| V2 quality_guard + fusion target, 5-seed mean | 292.6+-4.9 | 133.2+-4.4 | 84.4+-3.1 | 32.0+-2.6 | 23.2+-1.9 | 0.455+-0.015 |

## Current Interpretation

V2 is the first working version of the intended framework box.

Positive:

```text
selection is now paired with explicit semantic/temporal/risk evidence;
quality_guard nearly matches V4 fusion while reducing good-anchor regressions.
```

Negative:

```text
direct hybrid evidence scoring still over-trusts temporal_gain and causes too
many regressions. The evidence heads are useful, but their calibration is not
yet strong enough to replace the original fusion scorer.
```

The best current V2 setting is:

```text
results/semantic_temporal_candidate_scorer_v2_quality_guard_fusiontarget_seed2026
results/semantic_temporal_candidate_scorer_v2_quality_guard_fusiontarget_cached_5seed_summary.json
```

This is now a validated framework milestone on validation: the 5-seed mean
`R1@0.7=32.27+-0.93` essentially matches the V4 candidate-level fusion mean
`32.22`, while keeping an explicit semantic/temporal/risk explanation for every
selected candidate.

It is still not the final test-facing system until the validation-selected V2
checkpoint protocol is frozen and evaluated on the frozen test split.

## Frozen Test Result

Protocol:

```text
quality_guard + fusion target
seeds = 2026, 2027, 2028, 2029, 2030
validation checkpoints are evaluated frozen on test
```

Official-style test metrics:

| System | R1@0.5 | R1@0.7 | mAP(avg) | mAP@0.5 | mAP@0.75 |
|---|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 23.83 | 16.11 | 12.77 | 20.20 | 12.91 |
| V4 adapter shape_v2 top2 | 33.04 | 21.97 | 17.37 | 26.62 | 17.67 |
| V4 candidate-level fusion 5-seed | 34.65+-0.45 | 23.42+-0.60 | 18.66+-0.31 | 28.15+-0.41 | 18.98+-0.35 |
| V2 quality_guard scorer 5-seed | 35.13+-0.24 | 23.59+-0.28 | 18.53+-0.08 | 28.06+-0.12 | 18.71+-0.19 |

Intervention quality on test:

| System | Changed | Improved | Regressed | Semantic Recovered | Good Regressed | Precision |
|---|---:|---:|---:|---:|---:|---:|
| V4 candidate-level fusion best seed | 1212 | 577 | 283 | 170 | 57 | 0.476 |
| V2 quality_guard scorer 5-seed mean | 1151.4+-11.8 | 550.4+-13.1 | 254.2+-5.5 | 174.0+-8.5 | 43.4+-2.4 | 0.478+-0.011 |

Interpretation:

```text
V2 slightly improves strict R1@0.7 mean and stability over V4 fusion, while
reducing anchor/good-case regressions. It gives up a small amount of mAP(avg)
and mAP@0.75, so the current story is not "dominates every metric"; it is
"comparable strict localization with explicit evidence and cleaner risk".
```

The frozen test summary is:

```text
results/test_semantic_temporal_candidate_scorer_v2_quality_guard_fusiontarget_cached_5seed_summary.json
```

## V2 vs V4 Case Analysis

A representative test-side comparison between V4 fusion and V2 seed 2027 shows:

```text
V2 avoids V4 regressions: 90
V2 introduces new regressions: 67
V2 protects anchors regressed by V4: 23
V2 introduces anchor risk not present in V4: 9
```

The role migration is:

| Metric | V4 fusion | V2 scorer |
|---|---:|---:|
| Changed | 1212 | 1166 |
| Improved | 577 | 547 |
| Regressed | 283 | 259 |
| Semantic recovery | 170 | 171 |
| Temporal positive | 360 | 341 |
| Anchor regression | 57 | 43 |
| Anchor weakening | 38 | 25 |
| Good regressed | 57 | 43 |

This supports the intended interpretation: V2 keeps semantic recovery roughly
constant, sacrifices some temporal-positive interventions, and reduces the risk
side of fusion, especially good-anchor failures.

Detailed analysis:

```text
docs/test_semantic_temporal_candidate_scorer_v2_vs_v4_analysis.md
results/test_semantic_temporal_candidate_scorer_v2_quality_guard_seed2027_cases/v2_vs_v4_analysis.md
```

## Next Step

Do not tune on test. The next work should be:

```text
1. generate paper cases explaining semantic recovery,
   temporal correction, and anchor protection.
2. decide whether the main table should report V4 fusion as the strongest mAP
   system and V2 scorer as the interpretable evidence-selection system, or
   whether to tune V2 only on validation for mAP recovery before any further
   test run.
```

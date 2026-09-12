# Candidate-Level Risk Calibration V2

## Purpose

Candidate-Level Representation Fusion V1 proved that merged MS-CLAP and temporal-adapter candidates improve strict localization:

```text
Candidate-level fusion V1 R1@0.7: 30.97
```

But V1 over-intervened:

```text
changed: 305 / 352
improved: 128
regressed: 96
good MS-CLAP cases regressed: 25
```

V2 tests whether explicit candidate-level risk calibration can keep the candidate-fusion benefit while reducing unsafe interventions.

## V1 Regression Analysis

Script:

```text
src/analyze_candidate_fusion_cases.py
```

Output:

```text
results/candidate_level_representation_fusion_v1/case_analysis.json
```

Main findings:

| Group | Changed | Improved | Regressed | Good regressed | Semantic recovered | Avg top1 delta |
|---|---:|---:|---:|---:|---:|---:|
| MS boundary_error | 84 | 41 | 43 | 0 | 0 | +0.1077 |
| MS candidate_exists | 51 | 34 | 11 | 0 | 0 | +0.2284 |
| MS evidence_good_decode_bad | 24 | 9 | 0 | 0 | 0 | +0.1174 |
| MS good | 63 | 18 | 42 | 25 | 0 | -0.1965 |
| MS semantic_miss | 83 | 26 | 0 | 0 | 29 | +0.1474 |

Interpretation:

```text
V1 is useful when MS-CLAP is weak or incomplete.
V1 is risky when MS-CLAP is already good.
```

Two important failure transitions:

```text
good -> candidate_exists: 16 cases
good -> boundary_error: 8 cases
boundary_error -> semantic_miss: 18 cases
```

This motivates explicit gain/risk/good-risk modeling instead of a single candidate quality score.

## Script

```text
src/train_candidate_level_risk_calibrator.py
```

## Method

V2 uses the same merged candidate pool as V1, but replaces the single scalar reranker with a multi-head model:

```text
quality: candidate quality
gain: candidate is better than MS-CLAP default
risk: candidate is worse than MS-CLAP default
good_risk: candidate breaks a good MS-CLAP case
utility: threshold-aware candidate utility relative to MS-CLAP
```

Training labels are candidate-level and default-relative:

```text
candidate key = (IoU >= 0.7, IoU >= 0.5, IoU)
gain = candidate key > MS-CLAP key
risk = candidate key < MS-CLAP key
good_risk = MS-CLAP is good and candidate IoU < 0.7
```

Candidate utility includes:

```text
IoU delta
R1@0.7 threshold crossing
R1@0.5 threshold crossing
semantic-miss recovery bonus
good-case regression penalty
boundary-error-to-semantic-miss penalty
```

## Label Counts

Merged candidate labels:

| Split | Gain | Risk | Good risk | Semantic recovery |
|---|---:|---:|---:|---:|
| Train | 16,368 | 48,361 | 6,859 | 943 |
| Val | 2,591 | 7,903 | 1,778 | 124 |

The label distribution is highly asymmetric. Most candidates are worse than the MS-CLAP default, which explains why a merged candidate pool needs a conservative policy.

## Results

The stronger V2 run is the quality-anchored variant:

```text
results/candidate_level_risk_calibrator_v2b/
```

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 34.09 | 25.85 | 38.07 | 42.61 | 0.3271 | 0.4887 | 85 | 91 |
| Temporal-adapter shape_v2 top2 | 38.07 | 26.70 | 41.48 | 47.73 | 0.3606 | 0.5437 | 64 | 94 |
| Risk-aware representation gate V2 | 38.35 | 29.83 | 42.61 | 46.59 | 0.3738 | 0.5290 | 77 | 105 |
| Candidate-level fusion V1 | 41.48 | 30.97 | 40.91 | 47.44 | 0.3935 | 0.5624 | 76 | 109 |
| Candidate risk reranker V2b | 38.92 | 30.40 | 40.91 | 48.58 | 0.3751 | 0.5647 | 62 | 107 |
| Candidate risk balanced gate V2b | 39.20 | 30.40 | 39.20 | 44.60 | 0.3740 | 0.5274 | 72 | 107 |
| Merged candidate oracle | 74.15 | 63.07 | 63.07 | 63.07 | 0.6970 | 0.6970 | 15 | 222 |
| Merged candidate oracle 20% budget | 54.26 | 46.02 | 53.69 | 56.25 | 0.5015 | 0.5894 | 70 | 162 |

## Intervention Quality

| System | Changed | Improved | Regressed | Recovered semantic miss | Good regressed | Adapter top1 changed | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidate-level fusion V1 | 305 | 128 | 96 | 29 | 25 | 206 | 41.97 |
| Candidate risk reranker V2b | 334 | 153 | 108 | 41 | 28 | 243 | 45.81 |
| Candidate risk balanced gate V2b | 176 | 98 | 57 | 19 | 16 | 134 | 55.68 |
| Candidate risk precision gate V2b | 11 | 8 | 3 | 1 | 2 | 11 | 72.73 |
| Merged candidate oracle | 316 | 300 | 0 | 70 | 0 | 152 | 94.94 |

## Interpretation

V2b does not replace V1 as the best strict-top1 system:

```text
Candidate-level fusion V1 R1@0.7: 30.97
Candidate risk reranker V2b R1@0.7: 30.40
```

But it exposes a useful tradeoff:

```text
semantic miss: 76 -> 62
R5@0.7: 47.44 -> 48.58
top5 IoU: 0.5624 -> 0.5647
```

So the risk-calibrated model is better at semantic coverage and top-k candidate quality, but it still does not solve good-case regression.

The balanced gate is cleaner than raw V2b:

```text
changed: 334 -> 176
regressed: 108 -> 57
good regressed: 28 -> 16
precision: 45.81% -> 55.68%
```

However, this comes with reduced top-k performance and still does not beat V1 on strict top1.

## Research Takeaway

V2 is a useful diagnostic result:

```text
Explicit risk heads improve semantic coverage,
but candidate-level risk calibration is not solved by independent gain/risk heads.
The model still needs a decision objective tied directly to intervention against the MS-CLAP default.
```

In other words:

```text
The candidate pool is strong.
The model can identify many useful adapter candidates.
But the policy for replacing reliable MS-CLAP decisions remains under-calibrated.
```

This strengthens the research framing. The problem is not simply representation fusion; it is reliability of temporal semantic evidence under intervention.

## Next Step

The next version should not only predict candidate labels independently.

Recommended next experiment:

```text
Default-anchored listwise intervention learning
```

Specifically:

```text
1. keep the V1 candidate quality scorer as the candidate generator
2. train a separate intervention gate on the top merged candidate versus MS-CLAP default
3. use pairwise/listwise labels: do not only ask whether a candidate is good, ask whether it should replace MS
4. add hard negatives from V1 good-regression cases
5. tune intervention budgets for high precision before optimizing full recall
```

The target should be:

```text
keep V1-level R1@0.7,
recover V2-level semantic coverage where possible,
and reduce good-regressed cases below V1.
```

## Reports

```text
results/candidate_level_risk_calibrator_v2/report.html
results/candidate_level_risk_calibrator_v2/summary.json
results/candidate_level_risk_calibrator_v2b/report.html
results/candidate_level_risk_calibrator_v2b/summary.json
```

# Candidate-Level Representation Fusion V1

## Purpose

Risk-Aware Representation Gate V2 showed that MS-CLAP and the temporal-adapter evidence are complementary, but whole-prediction selection is too blunt.

This experiment moves the fusion decision from:

```text
choose MS-CLAP prediction or adapter prediction
```

to:

```text
merge MS-CLAP and adapter candidate lists
then train a candidate-level reranker
```

The research question is:

```text
When two representations expose different temporal semantic evidence,
can candidate-level selection recover better query-grounded windows
without blindly replacing reliable MS-CLAP decisions?
```

## Script

```text
src/train_candidate_level_representation_fusion.py
```

## Setup

Inputs:

```text
MS-CLAP evidence:
results/evidence_baseline_release_v1/full_train_eval/predictions_evidence_samples.json
results/evidence_baseline_release_v1/full_val_eval/predictions_evidence_samples.json

Temporal-adapter evidence:
results/evidence_baseline_tclap_inspired_v1/full_train_eval/predictions_evidence_samples.json
results/evidence_baseline_tclap_inspired_v1/full_val_eval/predictions_evidence_samples.json
```

Decoders:

```text
results/learned_evidence_decoder_shape_v2_semantic_only_top2/best.pt
results/learned_evidence_decoder_tclap_inspired_shape_v2_top2/best.pt
```

Candidate generation:

```text
topn_per_source: 2
feature_version: shape_v2
```

Candidate pool:

```text
train candidates: 64,729
val candidates: 10,494
feature dimension: 151
val candidates with IoU >= 0.7: 1,459
val candidates with IoU >= 0.5: 2,562
```

## Method

For each query, the script builds a merged candidate pool:

```text
MS-CLAP evidence -> MS candidate windows
temporal-adapter evidence -> adapter candidate windows
merged candidate pool -> MLP reranker
```

Each candidate contains:

```text
1. representation-specific evidence-shape features
2. pair-level MS/adpater evidence context
3. MS-vs-adapter curve similarity features
4. top-window disagreement features
5. candidate-vs-MS and candidate-vs-adapter geometry features
6. representation identity: MS-CLAP or adapter
```

The target is risk-aware candidate quality:

```text
base: candidate IoU
bonus: IoU threshold crossing at 0.5 and 0.7
bonus: recovering MS-CLAP semantic miss
penalty: regressing a good MS-CLAP case
```

Training uses MSE on the risk-aware target plus pairwise ranking loss within each query.

## Validation Results

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 34.09 | 25.85 | 38.07 | 42.61 | 0.3271 | 0.4887 | 85 | 91 |
| Temporal-adapter shape_v2 top2 | 38.07 | 26.70 | 41.48 | 47.73 | 0.3606 | 0.5437 | 64 | 94 |
| Risk-aware representation gate V2 | 38.35 | 29.83 | 42.61 | 46.59 | 0.3738 | 0.5290 | 77 | 105 |
| Candidate-level fusion V1 | 41.48 | 30.97 | 40.91 | 47.44 | 0.3935 | 0.5624 | 76 | 109 |
| Candidate-level balanced gate V1 | 38.35 | 28.69 | 39.77 | 45.45 | 0.3674 | 0.5263 | 76 | 101 |
| Candidate-level precision gate V1 | 34.09 | 26.14 | 38.07 | 42.61 | 0.3307 | 0.4904 | 85 | 92 |
| Whole-prediction oracle | 48.01 | 36.08 | 46.88 | 52.27 | 0.4466 | 0.5981 | 64 | 127 |
| Merged candidate oracle | 74.15 | 63.07 | 63.07 | 63.07 | 0.6970 | 0.6970 | 15 | 222 |
| Merged candidate oracle 20% budget | 54.26 | 46.02 | 53.69 | 56.25 | 0.5015 | 0.5894 | 70 | 162 |

## Intervention Quality

| System | Changed | Improved | Regressed | Recovered semantic miss | Good regressed | Adapter top1 changed | Precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidate-level fusion V1 | 305 | 128 | 96 | 29 | 25 | 206 | 41.97 |
| Candidate-level balanced gate V1 | 125 | 70 | 38 | 10 | 14 | 92 | 56.00 |
| Candidate-level precision gate V1 | 11 | 8 | 2 | 0 | 0 | 9 | 72.73 |
| Merged candidate oracle | 316 | 300 | 0 | 70 | 0 | 152 | 94.94 |
| Merged candidate oracle 20% budget | 71 | 71 | 0 | 15 | 0 | 0 | 100.00 |

## Regression Analysis

Post-hoc analysis script:

```text
src/analyze_candidate_fusion_cases.py
```

Output:

```text
results/candidate_level_representation_fusion_v1/case_analysis.json
```

The main regression pattern is not random:

```text
MS good cases changed: 63
MS good cases regressed: 42
MS good cases no longer good: 25
```

But V1 is useful when the MS-CLAP default is weak:

```text
MS semantic_miss cases changed: 83
semantic miss recovered: 29

MS candidate_exists cases changed: 51
improved: 34
```

This supports the next step: do not make the candidate scorer larger; learn when the merged candidate should be allowed to replace the MS-CLAP default.

## Interpretation

Candidate-level fusion is a real improvement over whole-prediction selection:

```text
Risk-aware representation gate V2 R1@0.7: 29.83
Candidate-level fusion V1 R1@0.7: 30.97
```

It also improves top1 IoU and top5 IoU:

```text
top1 IoU: 0.3738 -> 0.3935
top5 IoU: 0.5290 -> 0.5624
```

This supports the hypothesis that MS-CLAP and the temporal adapter expose complementary candidate-level temporal semantic evidence.

However, V1 is still not clean enough:

```text
changed cases: 305 / 352
regressions: 96
good MS-CLAP cases regressed: 25
```

So the candidate-level reranker is better at finding high-IoU windows, but it is not yet a reliable intervention policy.

The oracle gap is very large:

```text
candidate-level fusion V1 R1@0.7: 30.97
whole-prediction oracle R1@0.7: 36.08
merged candidate oracle R1@0.7: 63.07
merged candidate oracle 20% budget R1@0.7: 46.02
```

This means the merged candidate pool contains many correct windows, but learned selection is still the main bottleneck.

## Research Takeaway

The strongest current claim is:

```text
Temporal-adapter evidence does not only improve whole predictions.
It adds useful candidate-level temporal evidence that MS-CLAP often misses.
Candidate-level fusion can exploit this evidence and improve strict top1 localization,
but the intervention policy still needs explicit risk calibration.
```

This is a better framing than treating the experiment as an ensemble. The important result is the gap between available candidate evidence and reliable candidate selection.

## Next Step

The next version should not simply make the MLP larger.

Recommended next experiment:

```text
Candidate-Level Risk Calibration V2
```

Specifically:

```text
1. keep MS-CLAP as the default anchor
2. train separate heads for candidate gain, candidate regression risk, and good-case risk
3. add a listwise objective that compares candidate fusion against the MS-CLAP default
4. mine hard negatives from candidates that regress good MS-CLAP cases
5. evaluate both raw reranking and conservative intervention budgets
```

The target should be cleaner intervention quality, not only higher raw R1@0.7.

## Report

```text
results/candidate_level_representation_fusion_v1/report.html
results/candidate_level_representation_fusion_v1/summary.json
```

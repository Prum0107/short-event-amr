# Learned Evidence Decoder V1

## Purpose

The previous rule-based decoder experiment showed that the evidence curve `S(t, q)` contains boundary information. The best hand-crafted rule was `peak_drop`.

This experiment asks a sharper question:

```text
Can a model learn boundary quality from evidence-window features?
```

## Method

We use the trained evidence baseline to export evidence curves for both train and validation splits.

Inputs:

```text
train evidence:
results/evidence_baseline_release_v1/full_train_eval/predictions_evidence_samples.json

val evidence:
results/evidence_baseline_release_v1/full_val_eval/predictions_evidence_samples.json
```

For each query, we generate candidate windows from multiple sources:

```text
current_start_end
threshold_mean
threshold_p60
threshold_p70
peak_drop
peak_expand
multiscale
dense_contrast
```

For each candidate window, we extract evidence-window features:

```text
inside mean/max/top25/std evidence
start/end/center evidence
left/right outside evidence
inside-outside contrast
local boundary drops
window length and center
distance to global evidence peak
candidate source and source rank
```

The target label is:

```text
candidate IoU with ground truth
```

The learned decoder is a small MLP trained with:

```text
MSE(predicted quality, IoU)
+ pairwise ranking loss within the same query
```

## Candidate Dataset

```text
train candidates: 148327
val candidates: 21358
```

Best epoch:

```text
epoch 9
```

## Main Results

| Decoder | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Mean Top1 IoU |
|---|---:|---:|---:|---:|---:|
| learned_scorer | 33.24 | 23.58 | 34.66 | 39.20 | 0.3131 |
| peak_drop | 32.10 | 22.73 | 34.66 | 37.78 | 0.3032 |
| dense_contrast | 31.53 | 21.31 | 30.68 | 36.65 | 0.2889 |
| threshold_mean | 29.55 | 19.32 | 28.41 | 31.82 | 0.2956 |
| current_start_end | 23.01 | 17.33 | 25.85 | 36.65 | 0.2572 |

The learned decoder improves over the original current decoder:

```text
R1@0.7: 17.33 -> 23.58
R1@0.5: 23.01 -> 33.24
```

It also improves over the best hand-crafted rule:

```text
peak_drop R1@0.7: 22.73
learned_scorer R1@0.7: 23.58
```

## Failure Category Change

| Decoder | Good | Boundary Error | Candidate Exists | Semantic Miss |
|---|---:|---:|---:|---:|
| current_start_end | 61 | 116 | 68 | 68 |
| peak_drop | 80 | 93 | 53 | 98 |
| learned_scorer | 83 | 74 | 55 | 105 |

The learned scorer increases good cases:

```text
current: 61
peak_drop: 80
learned: 83
```

It also reduces boundary errors:

```text
current: 116
peak_drop: 93
learned: 74
```

This directly supports the hypothesis that boundary quality can be learned from evidence-window features.

## Important Caveat

Semantic misses increase compared with the original current decoder:

```text
current semantic_miss: 68
learned semantic_miss: 105
```

This suggests that the learned decoder is better at selecting sharp evidence-supported boundaries, but if the evidence curve itself points to the wrong semantic region, the decoder may follow it confidently.

In other words:

```text
learned decoding improves boundary quality when evidence is meaningful,
but it cannot fix semantic evidence failures.
```

## Research Interpretation

This experiment gives us a clean second-stage result:

```text
Stage 1:
  learn temporal semantic evidence S(t, q)

Stage 2:
  learn how to decode S(t, q) into high-quality temporal boundaries
```

The learned decoder does not use raw audio or new audio features. It only uses the evidence curve and candidate-window statistics. Therefore, the improvement is attributable to boundary quality learning rather than a stronger encoder.

## HTML Report

Open:

```text
results/learned_evidence_decoder_v1/report.html
```

The report includes:

```text
metric comparison
failure category comparison
learned-decoder improvement cases
learned-decoder regression cases
```

## Next Step

The next improvement should target semantic misses and overconfident wrong peaks.

Possible directions:

```text
1. Add semantic confidence calibration for the evidence curve.
2. Add hard negatives where evidence is high but IoU is low.
3. Train the evidence model with candidate-level feedback from the learned decoder.
4. Include query-region interaction features, not only scalar evidence statistics.
```

# Evidence Decoder Experiment V1

## Purpose

After the first evidence baseline, we observed:

```text
semantic match -> high S(t, q)
semantic miss -> low S(t, q)
```

The next question is:

```text
How should an evidence curve S(t, q) be decoded into accurate temporal boundaries?
```

This experiment compares several rule-based evidence-to-boundary decoders on the same validation evidence curves.

Input:

```text
results/evidence_baseline_release_v1/full_val_eval/predictions_evidence_samples.json
```

Output:

```text
results/evidence_decoder_experiment_v1/report.html
```

## Compared Decoders

```text
current_start_end:
  original model start/end decoder

threshold_mean / threshold_p60 / threshold_p70:
  threshold evidence curve and merge continuous high-evidence regions

peak_expand:
  find high evidence peaks and expand to a percentile floor

peak_drop:
  find high evidence peaks and expand until evidence drops by a fixed margin

multiscale:
  create fixed-length windows around high-evidence peaks

dense_mean:
  score all windows by mean evidence

dense_contrast:
  score all windows by inside evidence minus neighboring outside evidence
```

## Main Results

| Decoder | R1@0.5 | R1@0.7 | R3@0.7 | Best IoU Top5 |
|---|---:|---:|---:|---:|
| peak_drop | 32.10 | 22.73 | 34.66 | 0.4901 |
| dense_contrast | 31.53 | 21.31 | 30.68 | 0.4052 |
| threshold_mean | 29.55 | 19.32 | 28.41 | 0.4848 |
| threshold_p60 | 27.56 | 17.90 | 25.85 | 0.4566 |
| current_start_end | 23.01 | 17.33 | 25.85 | 0.4578 |
| threshold_p70 | 27.84 | 15.91 | 25.28 | 0.4517 |
| peak_expand | 25.57 | 15.06 | 22.73 | 0.3992 |
| multiscale | 18.75 | 9.38 | 12.22 | 0.3021 |
| dense_mean | 3.12 | 0.28 | 1.99 | 0.1518 |

The best rule is:

```text
peak_drop
```

It improves over the current start/end decoder:

```text
R1@0.7: 17.33 -> 22.73
R1@0.5: 23.01 -> 32.10
R3@0.7: 25.85 -> 34.66
```

## Interpretation

The current start/end decoder can learn boundaries, but it does not always align with the shape of the evidence curve.

`peak_drop` works better because it follows a simple intuition:

```text
Start from high semantic evidence.
Expand while the evidence remains close to the local peak.
Stop when the evidence drops enough.
```

This is closer to the meaning of a temporal semantic evidence curve:

```text
a relevant event occupies the region where semantic support stays high,
not necessarily the region selected by independent start/end maxima.
```

## Failure Category Change

| Decoder | Good | Boundary Error | Candidate Exists | Semantic Miss |
|---|---:|---:|---:|---:|
| current_start_end | 61 | 116 | 68 | 68 |
| peak_drop | 80 | 93 | 53 | 98 |

The good cases increase:

```text
61 -> 80
```

Boundary errors decrease:

```text
116 -> 93
```

This supports the idea that evidence-based boundary decoding can reduce boundary mistakes.

However, semantic misses increase:

```text
68 -> 98
```

This suggests that peak-based decoding may become too dependent on the evidence curve. If the evidence curve is semantically wrong or noisy, the decoder follows the wrong peak more confidently.

## Key Finding

This experiment shows that:

```text
Changing only the evidence-to-boundary decoder improves R1@0.7 by +5.40 points.
```

No new audio features, encoders, raw audio, or extra branches were used.

Therefore, the next research focus is justified:

```text
Evidence-to-boundary decoding is a real bottleneck and deserves explicit study.
```

## HTML Report

Open:

```text
results/evidence_decoder_experiment_v1/report.html
```

The report includes:

```text
decoder metric table
failure category table
improvement SVG cases
regression SVG cases
```

In the visualizations:

```text
green = ground truth
red = current start/end decoder top1
purple = best decoder top1
blue = S(t, q)
```

## Next Step

The next experiment should move from rule-based decoding to a small learned candidate quality scorer:

```text
generate evidence-based candidate windows
extract candidate features from S(t, q)
train a scorer to predict candidate IoU with GT
rank candidates by predicted quality
```

This targets the remaining issues:

```text
1. peak_drop still follows wrong evidence peaks when the evidence curve is noisy
2. candidate selection remains imperfect
3. boundary quality should be learned rather than hand-coded
```

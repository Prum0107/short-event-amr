# Evidence Baseline Release V1

## Purpose

This is the first full release-split experiment for the new research direction:

```text
Query-Guided Temporal Semantic Evidence Learning
```

The goal is not to build a final competition system yet. The goal is to verify whether an explicit evidence curve `S(t, q)` can learn useful time-localized semantic signals.

## Setup

Command:

```bash
python src/train_evidence_baseline.py \
  --train_data_path data/castella_train_release.jsonl \
  --val_data_path data/castella_val_release.jsonl \
  --epochs 3 \
  --batch_size 16 \
  --eval_batch_size 16 \
  --results_dir results/evidence_baseline_release_v1 \
  --save_path results/evidence_baseline_release_v1/best.pt
```

Data:

```text
train size: 2182
val size: 352
```

Model:

```text
audio CLAP + TEF features
query CLAP text features
query-guided temporal interaction
temporal Conv1D layers
evidence head S(t, q)
start head
end head
```

Training losses:

```text
dense evidence BCE
soft start/end boundary CE
```

## Validation Results

| Epoch | R1@0.5 | R1@0.7 | R3@0.5 | R3@0.7 | best IoU top5 | evidence gap |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.64 | 8.24 | 25.57 | 17.33 | 36.02 | 0.1185 |
| 2 | 19.89 | 12.50 | 34.38 | 23.30 | 41.80 | 0.1733 |
| 3 | 23.01 | 17.33 | 38.92 | 25.85 | 45.78 | 0.2178 |

The best checkpoint is:

```text
results/evidence_baseline_release_v1/best.pt
```

## Initial Takeaways

The first important observation is that the evidence curve becomes more discriminative across epochs.

```text
val evidence gap:
epoch 1: 0.1185
epoch 2: 0.1733
epoch 3: 0.2178
```

Here, evidence gap means:

```text
mean S(t, q) inside GT window - mean S(t, q) outside GT window
```

This suggests that the model is not only learning to output windows. It is also learning to assign higher semantic support to GT regions.

The second observation is that top-k recall is noticeably higher than top-1 recall:

```text
epoch 3 R1@0.7: 17.33
epoch 3 R3@0.7: 25.85
epoch 3 best IoU top5: 45.78
```

This is consistent with previous system lessons: the model often has a useful candidate in the ranked list, but top-1 selection and boundary quality still need work.

## Visualization

Evidence curve SVGs were generated without external plotting dependencies:

```text
results/evidence_baseline_release_v1/plots_epoch3/
```

Each plot shows:

```text
blue line: evidence curve S(t, q)
green span: ground truth window
red span: top-1 prediction
```

## Engineering Note

The original decoder used a Python O(T^2) loop over all start/end pairs and was too slow on release validation. It was replaced by a vectorized torch implementation in:

```text
src/evidence_baseline.py
```

This makes release validation practical.

## Next Steps

The next research step should be evidence analysis rather than more model stacking.

Recommended next experiments:

```text
1. Inspect good and bad evidence curves.
2. Categorize failures into semantic miss, partial match, and boundary error.
3. Compare boundary-only training vs evidence + boundary training.
4. Add hard negatives from old proposal systems.
5. Improve top-1 decoding using evidence-aware boundary refinement.
```

The first run supports the new direction:

```text
explicit temporal semantic evidence is learnable and measurable.
```

# Evidence Case Analysis V1

## Purpose

This analysis studies whether the first evidence baseline actually learns meaningful temporal semantic evidence, and where it fails.

Input:

```text
results/evidence_baseline_release_v1/full_val_eval/predictions_evidence_samples.json
```

Output:

```text
results/evidence_baseline_release_v1/full_val_case_analysis/
```

The analysis covers all 352 CASTELLA validation samples.

## Case Categories

Each sample is assigned to one category:

```text
good:
  top1 IoU >= 0.7

candidate_exists:
  top1 is wrong, but one of top5 predictions has IoU >= 0.7

boundary_error:
  top1 overlaps with GT, but IoU is below 0.7

evidence_good_decode_bad:
  top1 does not overlap GT, but GT evidence is clearly higher than outside evidence

semantic_miss:
  no useful overlap and no strong GT evidence
```

## Summary

| Category | Count | Rate |
|---|---:|---:|
| good | 61 | 17.33% |
| boundary_error | 116 | 32.95% |
| candidate_exists | 68 | 19.32% |
| evidence_good_decode_bad | 56 | 15.91% |
| semantic_miss | 51 | 14.49% |

Overall:

```text
mean top1 IoU: 0.2572
mean top5 IoU: 0.4578
mean evidence gap: 0.1828
```

## Per-Category Diagnostics

| Category | Count | Mean top1 IoU | Mean top5 IoU | Mean evidence gap |
|---|---:|---:|---:|---:|
| good | 61 | 0.8888 | 0.8964 | 0.3534 |
| boundary_error | 116 | 0.1694 | 0.3114 | 0.1450 |
| candidate_exists | 68 | 0.2449 | 0.8932 | 0.2283 |
| evidence_good_decode_bad | 56 | 0.0000 | 0.1120 | 0.1913 |
| semantic_miss | 51 | 0.0000 | 0.0650 | -0.0053 |

## Key Findings

### 1. Evidence gap is meaningful

The evidence gap strongly separates good cases from semantic misses:

```text
good evidence gap: 0.3534
semantic_miss evidence gap: -0.0053
```

This supports the research direction: the learned `S(t, q)` is not random. It carries useful information about whether the query is supported at the correct time.

### 2. Boundary error is the largest failure category

The largest group is:

```text
boundary_error: 116 / 352 = 32.95%
```

This means the model often finds a related region but fails to produce a high-IoU boundary. This is a strong signal that the next model improvement should focus on boundary refinement and evidence-to-window decoding.

### 3. Candidate selection is a separate bottleneck

The `candidate_exists` category has very high top5 IoU:

```text
candidate_exists mean top5 IoU: 0.8932
candidate_exists mean top1 IoU: 0.2449
```

This means the model often has a correct answer in its ranked list, but does not rank it first. This connects directly to the lessons from the previous multi-source systems: top1 selection remains hard even when good candidates exist.

### 4. Some cases have good evidence but bad decoding

The `evidence_good_decode_bad` category has:

```text
mean top1 IoU: 0.0000
mean evidence gap: 0.1913
```

This is especially important. It means the evidence curve may point toward the correct region, but the current start/end decoder fails to convert that curve into a correct window.

### 5. True semantic misses still exist

The `semantic_miss` category has:

```text
mean evidence gap: -0.0053
mean top5 IoU: 0.0650
```

These are likely cases where the current CLAP-based temporal representation does not capture the query semantics well enough.

## Generated SVG Galleries

Grouped evidence curve plots are saved here:

```text
results/evidence_baseline_release_v1/full_val_case_analysis/case_sets/
```

Subfolders:

```text
good/
boundary_error/
candidate_exists/
evidence_good_decode_bad/
semantic_miss/
```

Each SVG shows:

```text
blue line: S(t, q)
green span: ground truth
red span: top1 prediction
```

## Research Implication

The first evidence baseline reveals a useful decomposition:

```text
semantic evidence learning
boundary decoding
top1 candidate selection
```

These should be studied separately rather than hidden inside a single end-to-end score.

## Recommended Next Experiment

The next experiment should be:

```text
evidence-aware boundary refinement
```

Instead of changing the encoder, we should first improve how `S(t, q)` is converted into `[start, end]`.

A good next step:

```text
Use the evidence curve to generate candidate windows,
then train a small boundary/quality scorer to choose the best window.
```

Training signals:

```text
candidate IoU with GT
mean evidence inside candidate
boundary evidence at start/end
candidate length and center
```

This directly targets the two largest non-semantic failure modes:

```text
boundary_error
candidate_exists
```

It also keeps the research story clean:

```text
We first learn temporal semantic evidence,
then study how evidence should be decoded into temporal moments.
```

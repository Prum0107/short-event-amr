# Source-Aware Candidate Gate v1

## Question

`shape_v2 top2` showed that a compact candidate set works better than a large top10 candidate set.

This experiment asks:

> Should every candidate source use the same topN, or should stronger sources get different quotas?

This is a candidate-generation question, not a representation question.

## Candidate Source Diagnosis

Script:

`src/analyze_candidate_sources.py`

Report:

`results/candidate_source_diagnosis_semantic_only_top10/report.html`

Input evidence:

`results/evidence_baseline_v31_semantic_only_lsem025/full_val_eval/predictions_evidence_samples.json`

Any-source oracle using top10 candidates per source:

| Metric | Value |
|---|---:|
| Oracle R@0.5 | 81.53% |
| Oracle R@0.7 | 68.47% |
| Avg best IoU | 0.7525 |

This means the candidate pool contains many correct windows. The main difficulty is choosing the right top1.

Source-level oracle:

| Source | Oracle R@0.7 | Avg Best IoU | Best-Source Wins |
|---|---:|---:|---:|
| dense_contrast | 42.90% | 0.4758 | 41 |
| current_start_end | 39.20% | 0.5041 | 114 |
| peak_drop | 33.24% | 0.4844 | 39 |
| threshold_mean | 28.41% | 0.4669 | 65 |
| threshold_p70 | 27.84% | 0.4496 | 36 |
| multiscale | 26.42% | 0.3838 | 30 |
| threshold_p60 | 26.14% | 0.4400 | 23 |
| peak_expand | 23.01% | 0.3925 | 4 |

Interpretation:

- `current_start_end` wins the most samples, even though many individual candidates are noisy.
- `dense_contrast` has the highest single-source oracle R@0.7.
- `peak_drop` and `threshold_mean` are useful but degrade quickly at deeper ranks.
- A large top10 pool has high oracle quality but creates many semantic false candidates.

## Decoder Support

`src/train_learned_evidence_decoder.py` now supports source quotas:

```bash
--source_quotas current_start_end=2,threshold_mean=2,...
```

Missing sources fall back to `--topn_per_source`.

## Experiments

All experiments use:

- semantic-only evidence
- `shape_v2` decoder features
- 12 decoder epochs
- `lambda_pairwise=0.2`

| Decoder | Candidate Setting | Val Candidates | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic Miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shape_v2 top2 | all sources top2 | 5,202 | 34.94% | 25.85% | 38.35% | 43.18% | 0.3316 | 0.4959 | 77 |
| source gate v1 | loose source-aware | 7,188 | 34.66% | 24.43% | 38.35% | 44.89% | 0.3194 | 0.4999 | 99 |
| source gate v2 | tight source-aware | 4,243 | 37.22% | 26.42% | 39.49% | 43.75% | 0.3425 | 0.4905 | 102 |

Reports:

- `results/learned_evidence_decoder_shape_v2_semantic_only_top2/report.html`
- `results/learned_evidence_decoder_shape_v2_source_gate_v1/report.html`
- `results/learned_evidence_decoder_shape_v2_source_gate_v2_tight/report.html`

## Source Gate v1

Quota:

```text
current_start_end=4,
threshold_mean=3,
threshold_p60=2,
threshold_p70=2,
peak_drop=3,
peak_expand=2,
multiscale_10_20_40_80_150=2,
dense_contrast=4
```

Result:

- improves `R5@0.7` to 44.89%
- improves top5 IoU to 0.4999
- hurts strict top1
- increases semantic miss to 99

Interpretation:

The extra candidates include more good windows, but they also make top1 selection harder.

## Source Gate v2

Quota:

```text
current_start_end=2,
threshold_mean=2,
threshold_p60=1,
threshold_p70=1,
peak_drop=2,
peak_expand=1,
multiscale_10_20_40_80_150=1,
dense_contrast=3
```

Result:

- best current `R1@0.7`: 26.42%
- best current `R1@0.5`: 37.22%
- best current top1 IoU: 0.3425
- semantic miss increases to 102

This is not a pure improvement. It is a different operating point:

> Better top1 strict localization, worse semantic coverage.

## Migration: ShapeV2 Top2 to Source Gate V2

Report:

`results/decoder_migration_shape_v2_top2_to_source_gate_v2/report.html`

| Metric | Value |
|---|---:|
| Stable samples | 232 |
| Category improved | 30 |
| Category regressed | 53 |
| IoU improved | 26 |
| IoU regressed | 11 |
| Semantic misses recovered | 0 |
| Good regressed | 16 |
| Avg top1 IoU delta | +0.0109 |
| Avg top5 IoU delta | -0.0055 |

Important transitions:

- `candidate_exists -> good`: 15
- `boundary_error -> good`: 3
- `boundary_error -> semantic_miss`: 23
- `good -> candidate_exists`: 13
- `semantic_miss -> semantic_miss`: 77

Interpretation:

Source gate v2 improves top1 by selecting better candidates among already recoverable cases. It does not solve persistent semantic misses.

## Feature Ceiling Diagnosis

Report:

`results/feature_ceiling_diagnosis_source_gate_v2/report.html`

For source gate v2:

| Semantic Miss Type | Count |
|---|---:|
| representation ceiling | 47 |
| decoder/candidate ceiling | 55 |

Compared with shape_v2 top2:

| Decoder | Representation Ceiling SM | Decoder/Candidate Ceiling SM |
|---|---:|---:|
| shape_v2 top2 | 38 | 39 |
| source gate v2 | 47 | 55 |

So source gate v2 improves leaderboard-style top1 but worsens the semantic-miss diagnosis.

## Research Conclusion

This is a useful fork:

1. `shape_v2 top2` is better for semantic evidence explanation.
2. `source gate v2` is better for strict top1 retrieval.
3. Adding candidates can improve top-k oracle but confuse semantic coverage.

The candidate gate is not just a performance trick. It controls the trade-off between:

- semantic coverage
- boundary precision
- top1 aggressiveness

## Next Direction

We should not choose one blindly.

Recommended next step:

### Two-Mode Decoder

Use two complementary decoders:

- **coverage mode**: `shape_v2 top2`, lower semantic miss, better explanation
- **precision mode**: source gate v2, higher strict top1

Then learn a small selector that decides which mode to trust for each query based on evidence diagnostics:

- GT-like evidence peak sharpness
- source agreement
- candidate score entropy
- top1/top2 margin
- evidence gap
- source identity of top candidate

Hypothesis:

> The best system should not use one fixed candidate policy for all queries. Some queries need coverage; others need precision.

This keeps the research question coherent:

`representation evidence -> source-aware candidate policy -> evidence-shape decoding`

Follow-up:

`docs/two_mode_decoder_selector_v1.md`

The first two-mode selector confirms that query-adaptive candidate policy is a promising direction. The learned selector improves the semantic-coverage/precision trade-off, and the oracle selector shows a much higher upper bound.

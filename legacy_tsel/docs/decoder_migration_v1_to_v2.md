# Learned Decoder Migration: v1 Evidence to v2 Hard-Negative Evidence

## Purpose

This analysis compares the learned evidence decoder before and after hard-negative evidence training.

The goal is not only to check whether scores improve. The more important question is:

> When hard-negative learning changes an error, what kind of error does it become?

This tells us whether v2 is improving temporal semantic evidence or merely changing the ranking noise.

## Artifacts

Report:

`results/decoder_migration_v1_to_v2/report.html`

Data:

- `results/decoder_migration_v1_to_v2/summary.json`
- `results/decoder_migration_v1_to_v2/case_migrations.json`
- `results/decoder_migration_v1_to_v2/selected_cases.json`

Script:

`src/compare_decoder_migrations.py`

## Global Migration Summary

| Status | Count |
|---|---:|
| stable | 226 |
| category_improved | 65 |
| category_regressed | 34 |
| iou_improved | 13 |
| iou_regressed | 14 |

Average changes:

| Metric | Delta |
|---|---:|
| Top1 IoU | +0.0168 |
| Top5 IoU | +0.0241 |
| Evidence gap | -0.0034 |

The evidence gap is nearly unchanged, but localization quality improves. This suggests the benefit is not simply stronger positive evidence everywhere. It is more about making the evidence landscape more useful for candidate ranking and boundary decoding.

## Category Counts

| Category | v1 Learned | v2 Learned |
|---|---:|---:|
| good | 83 | 86 |
| candidate_exists | 55 | 65 |
| boundary_error | 74 | 89 |
| evidence_good_decode_bad | 35 | 28 |
| semantic_miss | 105 | 84 |

The biggest positive movement is the reduction of `semantic_miss` from 105 to 84.

## Transition Matrix

Rows are v1 categories. Columns are v2 categories.

| v1 / v2 | good | candidate_exists | boundary_error | evidence_good_decode_bad | semantic_miss |
|---|---:|---:|---:|---:|---:|
| good | 65 | 17 | 1 | 0 | 0 |
| candidate_exists | 11 | 33 | 11 | 0 | 0 |
| boundary_error | 9 | 13 | 49 | 1 | 2 |
| evidence_good_decode_bad | 0 | 1 | 6 | 26 | 2 |
| semantic_miss | 1 | 1 | 22 | 1 | 80 |

## Key Finding

v1 had 105 semantic misses. v2 recovers 25 of them:

- 1 becomes `good`
- 1 becomes `candidate_exists`
- 22 become `boundary_error`
- 1 becomes `evidence_good_decode_bad`

This is the most important result.

Hard-negative evidence training usually does not turn a semantic miss directly into a perfect prediction. Instead, it often moves the error from:

`wrong event / wrong semantic region`

to:

`right-ish event, but boundary not accurate`

That is a useful migration. Boundary errors are more tractable than semantic misses because the model has at least found the relevant temporal neighborhood.

## Regression Pattern

There are 18 cases where v1 was `good` but v2 is not:

- 17 become `candidate_exists`
- 1 becomes `boundary_error`

This means v2 often still keeps a good candidate in the list but changes top-1 ranking. That points to candidate ranking instability rather than complete semantic collapse.

## Research Interpretation

The current evidence supports this refined hypothesis:

> Hard-negative learning reduces complete semantic misses, but it exposes a second bottleneck: boundary-sensitive candidate ranking.

So the next system should not only add stronger negatives. It should distinguish negative types:

1. Semantic false peaks: fully wrong temporal regions.
2. Boundary distractors: partially overlapping regions with poor boundaries.
3. Over-wide or under-wide windows around the correct event.
4. Repeated-event distractors inside the same audio.

Each type should have a different training pressure. Semantic false peaks should be suppressed. Boundary distractors should teach sharper temporal transitions instead of being treated as fully wrong.

## Next Experiment

The recommended v3 is type-aware hard-negative training:

- Mine v2 errors from the training split.
- Label each hard negative by error type.
- Use separate ranking losses for semantic false peaks and boundary distractors.
- Evaluate migration again, especially:
  - `semantic_miss -> boundary_error`
  - `boundary_error -> good`
  - `good -> candidate_exists`

This keeps the project aligned with the central research question: learning temporally grounded semantic evidence, then decoding it into accurate time boundaries.

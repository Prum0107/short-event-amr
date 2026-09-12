# Feature Ceiling Diagnosis v1

## Question

We are still using extracted MS-CLAP features. This raises an important question:

> Are we limited by MS-CLAP representation, or are we still failing to decode the evidence that MS-CLAP already provides?

This diagnosis separates the two cases.

## Method

Script:

`src/analyze_feature_ceiling.py`

For each validation query, the script compares evidence scores inside the ground-truth window against evidence scores outside the ground-truth window.

It computes:

- GT evidence mean
- outside-GT evidence mean
- GT top-25% evidence
- outside-GT top-25% evidence
- whether the global evidence peak is inside GT
- fraction of top evidence frames inside GT
- decoder category from learned case rows

Then it assigns a signal bucket:

| Signal | Meaning |
|---|---|
| strong | GT evidence is clearly higher and the evidence peak is inside GT |
| partial | GT has usable evidence, but it is not dominant |
| weak | GT has little usable evidence |
| misleading | outside-GT evidence is clearly stronger than GT evidence |

For `semantic_miss` cases:

- `weak` or `misleading` means likely representation ceiling.
- `strong` or `partial` means likely decoder/candidate ceiling.

## Input

Evidence source:

`results/evidence_baseline_v31_semantic_only_lsem025/full_val_eval/predictions_evidence_samples.json`

Decoder rows:

- basic learned decoder:
  `results/learned_evidence_decoder_v31_semantic_only_lsem025/learned_case_rows.json`
- shape-aware decoder:
  `results/learned_evidence_decoder_shape_v2_semantic_only_top2/learned_case_rows.json`

Reports:

- `results/feature_ceiling_diagnosis_semantic_only_basic_top10/report.html`
- `results/feature_ceiling_diagnosis_semantic_only_shape_v2_top2/report.html`

## Dataset-Level Evidence Signal

Because both diagnoses use the same evidence source, the dataset-level signal buckets are identical:

| Signal | Count |
|---|---:|
| strong | 137 |
| partial | 164 |
| weak | 10 |
| misleading | 41 |

This means MS-CLAP evidence is not uniformly weak.

In fact, `301 / 352` validation queries have at least partial GT evidence. However, `41 / 352` are misleading: the evidence peak is stronger outside the true event.

## Decoder Comparison

| Decoder | Semantic Miss | Representation Ceiling SM | Decoder/Candidate Ceiling SM |
|---|---:|---:|---:|
| basic learned top10 | 97 | 41 | 56 |
| shape_v2 top2 | 77 | 38 | 39 |

Key observation:

`shape_v2 top2` reduces semantic misses by 20.

Most of that reduction comes from decoder/candidate ceiling:

- decoder/candidate semantic miss drops from 56 to 39
- representation-ceiling semantic miss only drops from 41 to 38

So `shape_v2` mostly recovers cases where MS-CLAP already had usable evidence, but the old decoder failed to exploit it.

## ShapeV2 Top2 Diagnosis

| Category | Count | Avg GT Mean Margin | Avg GT Top25 Margin | Peak In GT |
|---|---:|---:|---:|---:|
| good | 91 | +0.2775 | +0.2174 | 91.21% |
| candidate_exists | 61 | +0.1283 | +0.0865 | 67.21% |
| boundary_error | 97 | +0.0818 | +0.0262 | 32.99% |
| evidence_good_decode_bad | 26 | +0.1409 | +0.0179 | 3.85% |
| semantic_miss | 77 | -0.0098 | -0.0889 | 1.30% |

Interpretation:

- `good` cases have strong, clean evidence: high GT margin and peak mostly inside GT.
- `candidate_exists` cases also have strong evidence, but top1 selection is wrong.
- `boundary_error` cases still have positive GT evidence, but the peak is often not inside GT.
- `semantic_miss` cases have negative GT evidence margins and almost never have the global peak inside GT.

This validates the diagnosis: persistent semantic misses are much closer to the MS-CLAP representation ceiling.

## What This Means

The answer is not simply:

> MS-CLAP is bad.

The better answer is:

> MS-CLAP provides useful but noisy temporal semantic evidence. Some failures are true representation ceiling, but many failures are still decoder/candidate failures.

Our current best system has:

- 38 semantic misses likely caused by representation ceiling
- 39 semantic misses likely caused by decoder/candidate ceiling
- 184 non-good cases that are mainly decoding ceiling

So we should not abandon the decoder line. It is still productive.

## Research Implication

This supports a two-branch research plan:

### Branch A: Continue Decoder Research

For cases with usable evidence, improve:

- source-aware candidate gating
- compact candidate sets
- boundary-aware reranking
- evidence-shape features
- over-wide / under-wide candidate penalties

This branch targets `decoder_candidate_ceiling` and `decoding_ceiling`.

### Branch B: Representation Repair For True Ceiling Cases

For weak/misleading evidence cases, test another representation source only on hard cases:

- BEATs / AudioMAE / AST / PANNs as event-detail features
- Qwen Audio-style semantic rescoring for persistent misleading cases
- hybrid reranking where MS-CLAP proposes evidence and a second model repairs ambiguous cases

This branch targets the 38 current semantic misses that look like true representation ceiling.

## Next Experiment

Recommended immediate next step:

### Source-Aware Candidate Gate

Use shape_v2 features, but control candidate complexity before reranking.

Motivation:

- `shape_v2 top2` works better than `shape_v2 top10`
- top10 introduces too many high-evidence distractors
- many remaining failures are not representation failures

Hypothesis:

> A compact, source-aware candidate set plus evidence-shape reranking will recover more decoder/candidate ceiling cases without changing MS-CLAP features.

After that, we can isolate the remaining representation-ceiling cases and test a second audio representation only where MS-CLAP genuinely fails.

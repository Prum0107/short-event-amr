# Transformer Intervention Case Analysis V5.2

## Question

V5.1 showed that the Transformer can help as a conservative correction signal, but only if we know when to trust it. V5.2 analyzes the validation cases where the Transformer beats the two-mode default.

The goal is not to tune a new score directly. The goal is to explain:

- when the Transformer helps
- when it hurts
- whether simple interpretable rules are enough
- what kind of gate should come next

## Setup

Script:

`src/analyze_transformer_intervention_cases.py`

Inputs:

- Evidence: `results/evidence_baseline_v31_semantic_only_lsem025/full_val_eval/predictions_evidence_samples.json`
- Cached mode records: `results/conservative_transformer_gate_v51/val_mode_records.json`

Output:

`results/transformer_intervention_case_analysis_v52`

Report:

`results/transformer_intervention_case_analysis_v52/report.html`

## Overall Relationship

Validation samples: 352

| Relation | Count |
|---|---:|
| Transformer better than default | 98 |
| Default better than Transformer | 109 |
| Tie | 145 |

Top-1-only relationship:

| Relation | Count |
|---|---:|
| Transformer top1 improved | 63 |
| Transformer top1 regressed | 93 |
| Same top1 IoU | 196 |

This explains why the Transformer is tricky. It has real upside, but the raw direct Transformer decision is dangerous: it improves top1 on 63 cases but regresses top1 on 93.

## What Kind of Changes Happen?

| Change type | Count |
|---|---:|
| small_or_mixed_change | 210 |
| boundary_or_event_regression | 25 |
| semantic_jump_harm | 22 |
| topk_only_gain | 21 |
| good_regression | 19 |
| good_refinement | 16 |
| to_good | 15 |
| boundary_tightening | 9 |
| candidate_to_top1 | 9 |
| semantic_recovery | 6 |

The Transformer's useful behavior is mostly boundary/candidate refinement, not broad semantic rescue.

Semantic recovery exists but is small: only 6 cases. The larger opportunity is:

- moving candidate-exists cases into correct top1
- tightening boundary errors
- refining already-correct windows
- improving top-k candidate quality

## Group Signatures

Transformer-better cases:

- average top1 delta: +0.1342
- average top5 delta: +0.0845
- strict IoU crossings: 15
- loose IoU crossings: 12
- semantic recoveries: 6
- average Transformer agreement with any MLP/default window: 0.7941
- median Transformer agreement: 1.0000
- average center delta: 0.0926
- median center delta: 0.0130

Default-better cases:

- average top1 delta: -0.1932
- average top5 delta: -0.0247
- good regressions: 19
- average Transformer agreement with any MLP/default window: 0.6529
- median Transformer agreement: 0.8571
- average center delta: 0.1546

Important interpretation:

Transformer often helps when it stays near the same temporal neighborhood but changes the boundary. It is dangerous when it shifts too far or turns a correct/default-good case into a different candidate.

## Rule Probe Findings

Best single-feature probes only weakly enrich useful Transformer interventions.

Top examples:

| Rule | Selected | Hits | Precision | Recall |
|---|---:|---:|---:|---:|
| length_delta >= 0.081 | 71 | 30 | 42.25% | 30.61% |
| length_delta >= 0.038 | 107 | 43 | 40.19% | 43.88% |
| score_advantage <= -0.056 | 71 | 26 | 36.62% | 26.53% |
| center_delta >= 0.048 | 106 | 36 | 33.96% | 36.73% |
| cov_pre_iou <= 0.331 | 106 | 34 | 32.08% | 34.69% |

Best two-condition probes:

| Rule | Selected | Hits | Precision | Recall |
|---|---:|---:|---:|---:|
| uncertainty >= 0.4 and cov_pre_iou <= 0.6 | 110 | 40 | 36.36% | 40.82% |
| cov_pre_iou <= 0.6 and center_delta <= 0.4 | 117 | 42 | 35.90% | 42.86% |
| tr_best_agreement >= 0.2 and cov_pre_iou <= 0.6 | 118 | 42 | 35.59% | 42.86% |

Baseline Transformer-better rate is 98 / 352 = 27.84%. The best simple rules raise this to roughly 35%-42%, but not enough to be a reliable gate by themselves.

## Main Conclusion

Simple handcrafted rules are not enough.

The useful Transformer signal is real, but it is distributed across several weak cues:

- default uncertainty
- coverage/precision disagreement
- Transformer staying near the same temporal region
- boundary length change
- top-k local support
- avoidance of semantic jumps

This explains why V5.1 learned conservative gate was healthier than pure rules. It was not just tuning for score; it was combining weak evidence sources.

## Research Interpretation

The Transformer is not acting as a better semantic retriever. It is acting more like a temporal boundary alternative generator.

That means the right framing is:

> The stable MLP decoder finds the semantic neighborhood; the Transformer proposes boundary/candidate corrections; a calibrated gate decides whether the correction is trustworthy.

This fits the project direction well: temporal semantic evidence needs not only a better decoder, but also a reliability model.

## Next Direction

V5.3 should not simply enlarge the gate.

The better next step is a utility-aware conservative gate:

- positive labels only for meaningful interventions, such as strict/loose IoU crossings or large top1 gains
- negative labels for semantic jumps and good regressions
- neutral labels for top-k-only or tiny boundary changes
- asymmetric loss that penalizes damaging a good/default-correct case more than missing a small gain

This should reduce the main weakness of V5.1: it sometimes intervenes on cases where Transformer is only marginally different or where the default was already good.

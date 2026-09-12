# Two-Mode Decoder Selector v1

## Question

`shape_v2 top2` and `source_gate_v2` behave differently:

- `shape_v2 top2` has better semantic coverage and fewer semantic misses.
- `source_gate_v2` has better strict top1 localization.

This experiment asks:

> Can a small selector decide which decoder mode to trust for each query?

## Modes

Coverage mode:

`results/learned_evidence_decoder_shape_v2_semantic_only_top2/best.pt`

Precision mode:

`results/learned_evidence_decoder_shape_v2_source_gate_v2_tight/best.pt`

Selector script:

`src/train_two_mode_decoder_selector.py`

The selector is trained on train evidence and evaluated on val evidence. It does not use validation labels for training.

## Selector Features

The selector uses inference-available features:

- global evidence mean/std/max/percentiles
- top evidence peakiness
- coverage-mode top score and margin
- precision-mode top score and margin
- top candidate length and center
- source identity of each mode's top candidate
- overlap between the two modes' top windows
- score difference between the two modes

It does not use GT windows as input features.

## Results

Report:

`results/two_mode_decoder_selector_v1/report.html`

| Decoder | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 | Semantic Miss | Good |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coverage: shape_v2 top2 | 34.94% | 25.85% | 38.35% | 43.18% | 0.3316 | 0.4959 | 77 | 91 |
| precision: source_gate_v2 | 37.22% | 26.42% | 39.49% | 43.75% | 0.3425 | 0.4905 | 102 | 93 |
| two-mode selector | 37.50% | 26.14% | 38.07% | 43.18% | 0.3437 | 0.4921 | 84 | 92 |
| two-mode oracle | 41.19% | 30.97% | 41.19% | 44.89% | 0.3864 | 0.5128 | 77 | 109 |

Selector usage:

| Split | Coverage | Precision |
|---|---:|---:|
| train | 1460 | 722 |
| val | 238 | 114 |

The selector learned a conservative policy: use coverage most of the time, switch to precision for about one third of validation queries.

## Interpretation

The learned selector is not yet better than precision mode on strict `R1@0.7`, but it improves the trade-off:

- keeps most of precision mode's strict top1 gain
- recovers many semantic misses compared with precision mode
- gives the highest current `R1@0.5`
- gives the highest current mean top1 IoU

The oracle is important:

> If we could choose the better mode per query, R1@0.7 would reach 30.97%.

So the mode-selection problem is real and has a large upper bound.

## Migration From Coverage Mode

Report:

`results/decoder_migration_shape_v2_top2_to_two_mode_selector_v1/report.html`

| Metric | Value |
|---|---:|
| Stable samples | 300 |
| Category improved | 15 |
| Category regressed | 22 |
| IoU improved | 12 |
| IoU regressed | 3 |
| Semantic misses recovered | 0 |
| Good regressed | 7 |
| Avg top1 IoU delta | +0.0121 |
| Avg top5 IoU delta | -0.0038 |

Category shift:

| Category | Coverage | Selector |
|---|---:|---:|
| good | 91 | 92 |
| candidate_exists | 61 | 60 |
| boundary_error | 97 | 87 |
| evidence_good_decode_bad | 26 | 29 |
| semantic_miss | 77 | 84 |

Interpretation:

Compared with coverage mode, the selector improves top1 IoU and reduces boundary errors, but introduces some extra semantic misses.

## Migration From Precision Mode

Report:

`results/decoder_migration_source_gate_v2_to_two_mode_selector_v1/report.html`

| Metric | Value |
|---|---:|
| Stable samples | 284 |
| Category improved | 31 |
| Category regressed | 15 |
| IoU improved | 8 |
| IoU regressed | 14 |
| Semantic misses recovered | 18 |
| Good regressed | 10 |
| Avg top1 IoU delta | +0.0012 |
| Avg top5 IoU delta | +0.0016 |

Category shift:

| Category | Precision | Selector |
|---|---:|---:|
| good | 93 | 92 |
| candidate_exists | 61 | 60 |
| boundary_error | 65 | 87 |
| evidence_good_decode_bad | 31 | 29 |
| semantic_miss | 102 | 84 |

Interpretation:

Compared with precision mode, the selector restores semantic coverage while preserving almost the same strict top1 performance.

## Feature Ceiling Diagnosis

Report:

`results/feature_ceiling_diagnosis_two_mode_selector_v1/report.html`

| Decoder | Representation Ceiling SM | Decoder/Candidate Ceiling SM | Total SM |
|---|---:|---:|---:|
| coverage top2 | 38 | 39 | 77 |
| precision gate v2 | 47 | 55 | 102 |
| two-mode selector | 42 | 42 | 84 |

Interpretation:

The selector lands between the two modes. It does not solve representation ceiling, but it reduces the precision mode's decoder/candidate failures.

## Research Conclusion

This experiment supports a stronger claim:

> Candidate policy should be query-adaptive.

One fixed policy cannot optimize both semantic coverage and strict boundary precision.

The learned selector is only v1, but the oracle gap shows the direction is promising:

- learned selector R1@0.7: 26.14%
- oracle selector R1@0.7: 30.97%

## Next Direction

The next selector should be trained with better supervision:

1. Use candidate-level uncertainty, not only mode-level top1 features.
2. Add source-agreement features across all candidate sources.
3. Train the selector with a pairwise objective between coverage and precision outputs.
4. Add a semantic-safety constraint: avoid switching to precision when coverage evidence is strong and precision top window is far away.

This keeps the research story coherent:

`semantic evidence -> compact/precision candidate policies -> query-adaptive decoder selection`

Transformer follow-up:

`docs/candidate_context_transformer_v1.md`

The first candidate Transformer does not beat the MLP as a top1 selector, but it improves top-k candidate quality. This suggests candidate interaction is useful, but should likely feed a conservative selector rather than directly replace the MLP decoder.

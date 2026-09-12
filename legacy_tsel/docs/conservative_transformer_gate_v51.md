# Conservative Transformer Gate V5.1

## Question

V5 showed that the candidate-context Transformer contains complementary information, but allowing it to become a free third decoder made top-1 decisions less stable. V5.1 tests a more conservative hypothesis:

Use the existing two-mode selector as the default answer, and allow the Transformer to intervene only under a small budget.

This reframes the Transformer from a replacement decoder into a high-confidence correction module.

## Setup

Script:

`src/train_conservative_transformer_gate.py`

Output:

`results/conservative_transformer_gate_v51`

Default system:

`results/two_mode_decoder_selector_v1/selector.pt`

Auxiliary Transformer:

`results/candidate_transformer_source_gate_v2_v1/best.pt`

The experiment evaluates two gates:

- `rule_conservative_gate_v51`: explicit rule-based intervention
- `learned_conservative_gate_v51`: small binary gate deciding whether the Transformer may replace the default prediction

Both gates use only inference-available features:

- default two-mode confidence
- coverage/precision disagreement
- Transformer agreement with coverage, precision, or default windows
- Transformer top1/top2 margin
- Transformer local top-k support
- score, center, and length differences

## Results

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 | Semantic Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Two-mode default | 37.50 | 26.14 | 38.07 | 43.18 | 0.3437 | 0.4921 | 84 |
| Transformer direct | 34.66 | 25.00 | 38.92 | 44.89 | 0.3212 | 0.5080 | 100 |
| Rule conservative gate | 37.22 | 26.42 | 37.78 | 43.18 | 0.3425 | 0.4948 | 86 |
| Learned conservative gate | 37.50 | 26.42 | 38.35 | 42.90 | 0.3461 | 0.4966 | 84 |
| Default-vs-Transformer oracle | 40.91 | 30.40 | 40.06 | 45.17 | 0.3811 | 0.5156 | 78 |
| Oracle 15% budget | 40.91 | 30.40 | 39.20 | 43.47 | 0.3807 | 0.4968 | 81 |

Report:

`results/conservative_transformer_gate_v51/report.html`

Migration reports:

- `results/decoder_migration_two_mode_to_conservative_learned_v51/report.html`
- `results/decoder_migration_two_mode_to_conservative_rule_v51/report.html`

Feature ceiling:

`results/feature_ceiling_diagnosis_conservative_learned_v51/report.html`

## Intervention Behavior

Learned conservative gate on validation:

- changed 43 / 352 cases
- improved 14 cases
- regressed 6 cases
- unchanged 23 cases
- recovered 1 semantic miss
- regressed 3 good cases
- intervention precision: 32.56%

Rule conservative gate on validation:

- changed 36 / 352 cases
- improved 2 cases
- regressed 6 cases
- unchanged 28 cases
- recovered 0 semantic misses
- regressed 0 good cases
- intervention precision: 5.56%

The rule gate is too cautious and not discriminative enough. The learned gate is better: it improves strict R1@0.7, Top1 IoU, and Best IoU@5 while keeping semantic miss unchanged from the default.

## Interpretation

This is the cleanest Transformer result so far.

The Transformer still should not replace the MLP decoder:

- direct Transformer has lower R1@0.7 and many more semantic misses
- free three-way selection in V5 improved top-k but hurt stability

But V5.1 shows that Transformer can be useful as a conservative correction signal:

- default two-mode selector remains the backbone
- Transformer is only allowed to alter a small subset
- learned gate recovers some top1 quality without increasing semantic misses

The oracle is important. If we could perfectly identify when the Transformer is better than the default, R1@0.7 would rise from 26.14 to 30.40. With only a 15% intervention budget, the oracle still reaches 30.40. This means the opportunity is real; the unsolved part is reliable intervention detection.

## Research Takeaway

The key research question is no longer "Should we use a larger Transformer?"

The better question is:

When should a temporally contextual model be trusted to override a stable semantic-boundary decoder?

This matches the broader project direction: temporal semantic evidence is useful, but the model must also learn when evidence is reliable enough to justify a boundary decision.

## Next Step

The next useful step is to analyze the 98 validation cases where the Transformer beats the default. We should describe their common signatures:

- Does the default fail because coverage and precision disagree?
- Does the Transformer win mostly on boundary tightening?
- Does it help more when top-k Transformer candidates cluster around the same region?
- Does it hurt when it jumps to a different semantic event?

This analysis can produce a better V5.2 gate with fewer learned degrees of freedom and stronger interpretability.

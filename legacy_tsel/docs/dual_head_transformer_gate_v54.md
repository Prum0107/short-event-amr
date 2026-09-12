# Dual-Head Transformer Gate V5.4

## Motivation

V5.3 showed that a single utility-aware binary gate is too sparse and conservative. V5.4 separates the intervention decision into two interpretable predictions:

- gain head: predicts whether the Transformer intervention may produce a meaningful improvement
- risk head: predicts whether the Transformer intervention may cause damage

The intervention score is:

`gain_prob - lambda * risk_prob`

This keeps the system reusable. If the alternative decoder is later replaced by T-CLAP, raw-audio temporal features, or another boundary model, the same gain/risk gate idea still applies.

## Setup

Script:

`src/train_dual_head_transformer_gate.py`

Output:

`results/dual_head_transformer_gate_v54`

Inputs:

- Train mode records: `results/conservative_transformer_gate_v51/train_mode_records.json`
- Validation mode records: `results/conservative_transformer_gate_v51/val_mode_records.json`

Selected parameters:

```json
{
  "lambda_risk": 0.7,
  "score_threshold": 0.3,
  "min_agreement": 0.0,
  "max_center_delta": 0.7,
  "budget": 0.15,
  "protect_good": true
}
```

## Gain/Risk Labels

| Split | Gain | Risk | Neutral |
|---|---:|---:|---:|
| Train | 163 | 326 | 1693 |
| Val | 39 | 64 | 249 |

The important structural fact remains: useful interventions are sparse, and risky interventions are not rare.

## Validation Results

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 | Semantic Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Two-mode default | 37.50 | 26.14 | 38.07 | 43.18 | 0.3437 | 0.4921 | 84 |
| Transformer direct | 34.66 | 25.00 | 38.92 | 44.89 | 0.3212 | 0.5080 | 100 |
| Dual-head gate V5.4 | 37.22 | 26.99 | 38.35 | 42.90 | 0.3401 | 0.4899 | 89 |
| Gain/risk oracle 15% budget | 40.91 | 30.40 | 39.49 | 44.89 | 0.3797 | 0.5079 | 78 |

Report:

`results/dual_head_transformer_gate_v54/report.html`

Migration:

`results/decoder_migration_two_mode_to_dual_head_v54/report.html`

Feature ceiling:

`results/feature_ceiling_diagnosis_dual_head_v54/report.html`

## Intervention Behavior

Dual-head gate on validation:

- changed 36 cases
- improved 9
- regressed 16
- unchanged 11
- recovered 2 semantic misses
- good regressions: 0
- intervention precision: 25.00%

Migration from two-mode default:

- stable: 330
- category improved: 6
- category regressed: 9
- IoU improved: 2
- IoU regressed: 5
- average top1 IoU delta: -0.0036
- average top5 IoU delta: -0.0022

Category transitions:

- candidate_exists -> good: 3
- semantic_miss -> boundary_error: 2
- boundary_error -> semantic_miss: 7
- good -> good: 92

## Interpretation

V5.4 is the highest strict top1 gate result so far:

`R1@0.7 = 26.99%`

But it is not the best overall decoder:

- semantic miss increases from 84 to 89
- top1 IoU decreases from 0.3437 to 0.3401
- top5 quality decreases slightly

This means the gain head is more willing to capture high-strict-IoU corrections, but the current risk head does not yet prevent all damaging interventions. It successfully prevents good regressions, but it does not prevent enough boundary-error-to-semantic-miss transitions.

That is a useful mechanism-level result:

Risk is not one thing. "Good regression" and "semantic jump from an already imperfect boundary case" need separate treatment.

## Comparison Across Gates

| Gate | R1@0.7 | Top1 IoU | Semantic Miss | Good Regressed |
|---|---:|---:|---:|---:|
| Two-mode default | 26.14 | 0.3437 | 84 | 0 |
| V5.1 learned conservative | 26.42 | 0.3461 | 84 | 3 |
| V5.3 utility-aware | 26.14 | 0.3427 | 85 | 0 |
| V5.4 dual-head | 26.99 | 0.3401 | 89 | 0 |

V5.1 remains the best balanced system.

V5.4 is the best strict-IoU system, but it sacrifices coverage stability.

## Research Takeaway

The reusable system is working:

`default decoder -> alternative decoder -> intervention utility/gain/risk model -> migration analysis`

The important discovery is that intervention risk has multiple modes. A single risk head is too coarse.

## Next Decision

We should not keep blindly tuning V5.x.

At this point, the framework has done its job:

- it shows Transformer has complementary boundary information
- it shows direct replacement is unsafe
- it shows conservative intervention is better than free selection
- it exposes the specific unsolved risk: boundary-error-to-semantic-miss transitions

Recommended next step:

Freeze this gate framework as our analysis/evaluation system, and use it to test better audio features such as T-CLAP or raw-audio temporal features. The model side can improve later, but the next major bottleneck is likely representation quality, not another small gate variant.

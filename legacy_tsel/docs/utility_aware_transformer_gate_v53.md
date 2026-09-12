# Utility-Aware Transformer Gate V5.3

## Motivation

V5.1 showed that a conservative learned gate can use Transformer predictions as a correction signal. V5.2 showed why this is hard: the Transformer is sometimes useful, but simple rules only weakly enrich the useful cases.

V5.3 tests a more reusable formulation:

Instead of training on "Transformer better than default" directly, train on intervention utility.

This matters because the framework should survive future swaps:

- MS-CLAP to T-CLAP
- candidate Transformer to raw-audio temporal decoder
- MLP decoder to another boundary decoder

The reusable abstraction is:

`default prediction + alternative prediction + utility/risk model -> intervention decision`

## Setup

Script:

`src/train_utility_aware_transformer_gate.py`

Output:

`results/utility_aware_transformer_gate_v53`

Inputs:

- Train mode records: `results/conservative_transformer_gate_v51/train_mode_records.json`
- Validation mode records: `results/conservative_transformer_gate_v51/val_mode_records.json`

The gate trains only on non-neutral intervention examples.

Positive intervention labels include:

- strict IoU crossing
- loose IoU crossing
- semantic recovery
- moving a non-good case to good
- large top1 improvement

Negative intervention labels include:

- semantic jump harm
- good regression
- large top1 loss

Neutral examples are excluded from gate training.

## Label Distribution

| Split | Positive | Neutral | Negative |
|---|---:|---:|---:|
| Train | 163 | 1693 | 326 |
| Val | 39 | 249 | 64 |

This is the key finding before any metric: high-utility interventions are sparse. The gate only has 489 non-neutral train examples, and only 163 are positive.

## Validation Results

| System | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Best IoU@5 | Semantic Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Two-mode default | 37.50 | 26.14 | 38.07 | 43.18 | 0.3437 | 0.4921 | 84 |
| Transformer direct | 34.66 | 25.00 | 38.92 | 44.89 | 0.3212 | 0.5080 | 100 |
| Utility-aware gate V5.3 | 37.22 | 26.14 | 38.35 | 43.18 | 0.3427 | 0.4933 | 85 |
| Utility oracle | 40.91 | 30.40 | 40.06 | 45.74 | 0.3809 | 0.5191 | 78 |
| Utility oracle 15% budget | 40.91 | 30.40 | 39.49 | 44.89 | 0.3797 | 0.5079 | 78 |

Report:

`results/utility_aware_transformer_gate_v53/report.html`

Migration:

`results/decoder_migration_two_mode_to_utility_aware_v53/report.html`

Feature ceiling:

`results/feature_ceiling_diagnosis_utility_aware_v53/report.html`

## Intervention Behavior

Utility-aware gate on validation:

- changed 37 cases
- improved 4
- regressed 6
- unchanged 27
- recovered 1 semantic miss
- good regressions: 0
- intervention precision: 10.81%

Utility oracle with 15% budget:

- changed 53 cases
- improved 48
- regressed 1
- unchanged 4
- recovered 6 semantic misses
- good regressions: 0
- intervention precision: 90.57%

## Interpretation

V5.3 is a useful negative result.

The utility formulation is conceptually better and more reusable, but the current learned gate does not identify positive utility interventions reliably. It avoids damaging already-good predictions, but it misses most high-value opportunities.

This tells us the problem is not only "how should we define utility?" It is also:

How do we learn utility from sparse and noisy intervention examples?

## Comparison With V5.1

V5.1 learned conservative gate:

- R1@0.7: 26.42
- Top1 IoU: 0.3461
- semantic miss: 84
- changed 43 cases
- improved 14
- regressed 6

V5.3 utility-aware gate:

- R1@0.7: 26.14
- Top1 IoU: 0.3427
- semantic miss: 85
- changed 37 cases
- improved 4
- regressed 6

So V5.3 is more risk-aware but less useful. It becomes conservative in the wrong way: it avoids some dangerous cases, but does not find enough beneficial interventions.

## Research Takeaway

The reusable system direction is still correct:

`semantic evidence -> default decoder -> alternative decoder -> intervention utility model -> error migration report`

But the utility model should not be a plain binary classifier trained on sparse strong labels.

The next version should use a ranking or pairwise utility objective:

- rank interventions by expected utility instead of classifying them
- train on relative utility between candidate interventions
- keep neutral cases but give them small weights instead of removing them
- calibrate a "do no harm" score separately from an "expected gain" score

This is a better fit for the data because most cases are neutral or small-change, and the important question is which small subset deserves the intervention budget.

## Next Step

V5.4 should split the gate into two interpretable heads:

1. Risk head: predicts semantic jump / good regression risk.
2. Gain head: predicts strict crossing / large top1 improvement.

Then intervention score becomes:

`expected_gain - lambda * expected_risk`

This keeps the reusable framework while making the mechanism more explainable than a single binary gate.

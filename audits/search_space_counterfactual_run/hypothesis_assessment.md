# Hypothesis assessment

Scientific decision: **HARD_NEGATIVE_COMPETITION_SUPPORTED**

The runner used the pre-registered decoder-only memory-key mask. The full encoder, full-coordinate memory, positional embeddings, query embeddings, initial references, model parameters, query count, and post-processing remained fixed.

| Hypothesis | Status |
|---|---|
| `H1_GLOBAL_COMPETITION_CAUSAL_COMPONENT` | **NOT_SUPPORTED** |
| `H2_HARD_DISTRACTOR_EFFECT` | **SUPPORTED** |
| `H3_SEARCH_SPACE_DOSE_RESPONSE` | **NOT_SUPPORTED** |
| `H4_LONG_RECORDING_RESCUE` | **NOT_SUPPORTED** |
| `H5_SCALE_ERROR_REMAINS_AFTER_COMPETITION_RESCUE` | **SUPPORTED** |

## Predeclared assessment rules

- H1 requires GT_ONLY to improve both CenterHit@10≤2s and positive-overlap Top-10 availability in both primary duration bins.
- H2 requires HARD to be lower than matched RANDOM at the informative 25 and 50 levels in both primary bins for CenterHit@10≤2s; level 100 is a structural identity control because both masks expose the same complete non-GT set.
- H3 requires the RANDOM 25→50→100 means to be non-increasing between GT_ONLY and FULL_ACCESS in both primary bins.
- H4 requires the GT_ONLY rescue delta to be positive and larger for the long-recording half in both primary bins.
- H5 records persistent scale error when GT_ONLY has median width/GT > 2 and R1@0.7 < 50% in a primary bin.

These are diagnostic decision rules, not claims that the intervention is deployable. GT_ONLY uses ground-truth access and is a lower-access endpoint only.

## Validation

- Counterfactual validation: **PASS**
- Full-access reference reproduction: **PASS**
- Memory/position/reference invariants: **PASS**

## Claim boundary

A positive result supports only a decoder-level full-coordinate temporal competition mechanism. It does not show that global competition is the only cause, that the encoder representation is perfect, that normalized coordinates are irrelevant, or that GT-based masking is a deployable method.

Baseline source revision: `45ef471ee47ea75a2141d75bd9cfdb8c45dfc101`.

# GT vs hard-negative mechanism audit

Status: **COMPLETE_WITH_NATIVE_AND_HUMAN_COVERAGE_LIMITS**

This report is diagnostic only. No model weights, training procedure, ranking rule, or AMR method was changed.

## 1. Frozen cohorts

HARMFUL-HARD is defined before mechanism analysis as HARD_25 CenterHit@10≤2s lower than the mean of ten deterministic RANDOM_25 replicates. The resulting counts are 44/46 harmful/control in 0–2 s and 127/249 harmful/control in 2–5 s. P3 location-failure counts are 46 and 135, respectively.

## 2. Hard-negative geometry

The strongest selected HARD_25 peak is the contiguous component containing the highest-saliency selected token. Saliency is selection-conditioned, not independent evidence.

| Bin | Cohort | Adjacent | Moderately near | Remote | Near or GT±2s expanded | Median interval gap (s) |
|---|---|---:|---:|---:|---:|---:|
| 0-2s | harmful | 10 | 2 | 32 | 12/44 | 29.50 |
| 0-2s | control | 10 | 2 | 34 | 12/46 | 30.50 |
| 2-5s | harmful | 27 | 6 | 94 | 33/127 | 27.00 |
| 2-5s | control | 78 | 26 | 145 | 104/249 | 9.00 |

## 3. Native MS-CLAP GT vs hard

Scores use the verified native MS-CLAP shared projection and official logit-scaled normalized dot product on equal-duration raw-audio windows. Missing WAVs are BLOCKED.

| Bin | Cohort | Valid/total | GT>HARD | HARD>GT | Median S_GT−S_HARD |
|---|---|---:|---:|---:|---:|
| 0-2s | harmful | 8/44 | 7 | 1 | 0.5436763167381287 |
| 0-2s | control | 9/46 | 4 | 5 | -0.4266638867557049 |
| 2-5s | harmful | 11/127 | 9 | 2 | 1.8993163704872131 |
| 2-5s | control | 26/249 | 21 | 5 | 1.1430590152740479 |

On the available harmful rows, GT scored above hard in 7/8 (0–2 s) and 9/11 (2–5 s); this does not support a dominant native semantic-competitor explanation. Coverage is only 8/44 and 11/127, so the conclusion is limited rather than population-complete.

## 4. QD internal vs independent semantic evidence

The QD table reports final candidate-center ranks/densities and decoder attention, with selection-conditioned quantities explicitly named. Unsupported query-conditioned memory similarity is BLOCKED.

| Bin | Cohort | Native-valid N | Direction agreement | Direction disagreement |
|---|---|---:|---:|---:|
| 0-2s | harmful | 8 | 3 | 5 |
| 0-2s | control | 9 | 4 | 5 |
| 2-5s | harmful | 11 | 5 | 6 |
| 2-5s | control | 26 | 9 | 17 |

Agreement is mixed, not a proof of equivalence. The QD direction is a center-location diagnostic and is not the native semantic score.

## 5. Decoder reference attraction

Using the full-access baseline trace and its final top-1 query, attraction means the final reference is closer to the hard peak than its initial reference and closer to the hard peak than to the GT center.

| Bin | Harmful N | Begin hard/remain | Begin GT/move hard | Approach GT | Approach neither | Attraction flag |
|---|---:|---:|---:|---:|---:|---:|
| 0-2s | 44 | 23 | 6 | 9 | 6 | 26/44 |
| 2-5s | 127 | 61 | 17 | 34 | 15 | 66/127 |

The attraction trace supports a decoder-competition component in the harmful cohort, but it is still a post-hoc trajectory measurement of the frozen decoder, not an intervention that establishes necessity.

## 6. Matched replacement probe

The one-token replacement was valid for 171 harmful queries. Replacing the first deterministic RANDOM_25 token with the strongest selected hard token worsened the GT-center rank in 36 cases and changed mean Top-10 GT-center density by -0.0292. This is a bounded token-level probe; it is not a peak-level intervention.

## 7. Human review and repeats

The review UI was prepared from the strongest harmful cases with available WAVs: 8 cases in 0–2 s and 11 cases in 2–5 s. Labels were intentionally left blank. Possible repeated or incompletely annotated occurrences are therefore unresolved, not supported or rejected.

## 8. Taxonomy and scale separation

Taxonomy labels are allowed to overlap. `NEAR_GT_SCALE_LEAKAGE` is a geometry label, `SEMANTIC_COMPETITOR` requires native HARD>GT, `MODEL_SPECIFIC_FALSE_POSITIVE` requires native GT>HARD plus QD hard-favored location evidence, and `DECODER_ATTRACTION` comes from the layerwise trace. Unlabeled cases remain `UNRESOLVED`.

Scale is reported separately in `scale_relation.csv`; among harmful rows the median full-access width/GT ratio is 4.00 and HARD_25 is 4.67. A hard-negative location effect does not imply that span extent is accurate.

## 9. Hypotheses and branch

| Hypothesis | Status |
|---|---|
| H1_SEMANTIC_HARD_NEGATIVES | **NOT_SUPPORTED** |
| H2_MODEL_SPECIFIC_CALIBRATION_FAILURE | **PARTIALLY_SUPPORTED** |
| H3_NEAR_GT_SCALE_LEAKAGE | **PARTIALLY_SUPPORTED** |
| H4_REPEAT_OR_ANNOTATION_AMBIGUITY | **INCONCLUSIVE** |
| H5_DECODER_ATTRACTION | **SUPPORTED** |
| H6_HARD_NEGATIVE_AND_SCALE_ARE_SEPARATE | **INCONCLUSIVE** |
| selected next branch | **DECODER_COMPETITION** |

## 10. Claim boundary

This audit establishes that harmful HARD_25 selections often coincide with a frozen-decoder reference-attraction pattern, while native MS-CLAP evidence is mostly GT-favoring on the small raw-audio subset. It does not establish that every hard negative is semantically unrelated, does not resolve repeats or annotation misses, does not prove decoder attraction is necessary, and does not show that the stored QD-DETR temporal representation preserves native local MS-CLAP evidence.

No method design or next experiment was implemented.

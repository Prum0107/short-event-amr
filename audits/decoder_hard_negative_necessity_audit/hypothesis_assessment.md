# Hypothesis assessment

Status rules were fixed before inspecting results: H1 uses positive paired REMOVE_HARD−REMOVE_RANDOM CenterHit in both bins and >50% positive query-level effects; H2 requires both more GT-closer references and fewer HARD-closer references after removal; H3 compares harmful and control paired effects; H4 requires a positive layer-specific mean effect with >50% positive query-level effects in at least one named layer; H5 requires at least five rescued queries per bin with median width/GT >2.

| Hypothesis | Status |
|---|---|
| H1_HARD_REGION_NECESSITY | **INCONCLUSIVE** |
| H2_DECODER_ATTRACTION_CAUSAL | **NOT_SUPPORTED** |
| H3_HARMFUL_COHORT_SPECIFICITY | **SUPPORTED** |
| H4_DECODER_LAYER_LOCALIZATION | **NOT_SUPPORTED** |
| H5_HARD_COMPETITION_AND_SCALE_DISSOCIATE | **SUPPORTED** |

H2 raw redirect statistics: `{'redirect_rate_by_bin': {'0-2s': 0.03409090909090909, '2-5s': 0.03543307086614173}, 'N_pairs_by_bin': {'0-2s': 88, '2-5s': 254}}`.
Rescued scale statistics: `{'0-2s': {'N': 6, 'median_width_gt_ratio': 8.5, 'median_abs_log_width_gt_ratio': 2.1242476210246797, 'oracle10_iou05_pct': 0.0, 'oracle10_iou07_pct': 0.0, 'r1_iou07_pct': 0.0}, '2-5s': {'N': 17, 'median_width_gt_ratio': 3.5, 'median_abs_log_width_gt_ratio': 1.252762968495368, 'oracle10_iou05_pct': 17.647058823529413, 'oracle10_iou07_pct': 5.88235294117647, 'r1_iou07_pct': 0.0}}`.
Previous DECODER_ATTRACTION subset intervention statistics: `{'source_rows': 466, 'flagged_qids': 194, '0-2s': {'N': 26, 'mean_full_pct': 38.46153846153847, 'mean_remove_hard_pct': 50.0, 'mean_remove_random_pct': 40.38461538461539, 'mean_hard_minus_random_pp': 9.615384615384615, 'median_hard_minus_random_pp': 0.0, 'positive_rate': 0.23076923076923078}, '2-5s': {'N': 66, 'mean_full_pct': 45.45454545454545, 'mean_remove_hard_pct': 56.060606060606055, 'mean_remove_random_pct': 44.84848484848485, 'mean_hard_minus_random_pp': 11.212121212121213, 'median_hard_minus_random_pp': 0.0, 'positive_rate': 0.16666666666666666}}`.

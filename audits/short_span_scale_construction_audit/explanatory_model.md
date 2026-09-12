# Explanatory model

Outcome: `log(predicted_width / GT_width)` among final proposals with center error ≤1 s. Models are bounded descriptive OLS summaries; coefficients are associations, not causal effects.

| Model | N | R² |
|---|---:|---:|
| gt_only | 855 | 0.3827547874690975 |
| audio_only | 855 | 0.023515440363938933 |
| slot_only | 855 | 0.11897001494731618 |
| gt_audio | 855 | 0.5265234744173382 |
| gt_audio_slot_center | 855 | 0.623518884402529 |

Query-text token-count check: `{'N_queries': 528, 'token_count_vs_gt_duration_correlation': -0.06786981982331268, 'token_count_vs_centered_log_width_gt_ratio_correlation': 0.08536903019683462}`. This uses only the official query string and no external duration labels or uncontrolled clustering.

Audio-duration conditional summaries: `{'GT_1-2s': {'unadjusted': {'N': 24, 'slope': 0.8688014131842545, 'intercept': -2.935730642273366, 'correlation': 0.5340013837523461, 'residual_q25': -0.39760645734846733, 'residual_median': -0.013480597086804713, 'residual_q75': 0.4512620006879768}, 'adjusted': {'N': 24, 'R2': 0.5591365289542403, 'coefficients': {'intercept': -2.8923352858139553, 'log_gt_duration': -2.886579864025407e-15, 'log_audio_duration': 0.8698928138987013, 'center_error_sec': -0.5812206683862113, 'slot_2': -0.9700301175343431, 'slot_3': 0.6086125303044143, 'slot_4': 0.06590029496975809, 'slot_5': 0.5664806772273692, 'slot_7': 0.0842251551707724, 'slot_8': 0.5175575890092661, 'slot_9': 0.1233363428882227}}}, 'GT_2-3s': {'unadjusted': {'N': 87, 'slope': 0.8705467639011825, 'intercept': -3.502214623803229, 'correlation': 0.6292069045125859, 'residual_q25': -0.23582079952221457, 'residual_median': 0.035836729759413544, 'residual_q75': 0.21296378118642667}, 'adjusted': {'N': 87, 'R2': 0.5120568251227633, 'coefficients': {'intercept': -3.0045363983020406, 'log_gt_duration': -2.0825859333727883, 'log_audio_duration': 0.9902742780531688, 'center_error_sec': 0.514582095551731, 'slot_1': 0.019311358184370003, 'slot_2': -0.19717542635431912, 'slot_3': 0.11224479914519267, 'slot_4': -0.15584576839442607, 'slot_5': 0.49094242540697264, 'slot_6': -0.014282161287815336, 'slot_7': 0.2800693862613493, 'slot_8': 0.43858202121631656, 'slot_9': 0.15351908756735833}}}, 'GT_3-5s': {'unadjusted': {'N': 178, 'slope': 0.7093585479652287, 'intercept': -2.970440136504436, 'correlation': 0.5325315045837477, 'residual_q25': -0.33220204904125084, 'residual_median': -0.02584146720801056, 'residual_q75': 0.280336547874885}, 'adjusted': {'N': 178, 'R2': 0.4730816113356534, 'coefficients': {'intercept': -2.7879556595037362, 'log_gt_duration': -0.8471341308335296, 'log_audio_duration': 0.8061916807589314, 'center_error_sec': 0.18914257492292982, 'slot_1': 0.02298468310852053, 'slot_2': -0.7515008354674678, 'slot_3': 0.2656997112247283, 'slot_4': 0.28826946696529426, 'slot_5': 0.5610485880735164, 'slot_6': 0.7140727578856182, 'slot_7': 0.343174246804902, 'slot_8': 0.5317406602751936, 'slot_9': 0.21249507786369037}}}}`.

The strongest attribution is the factor combination shown by the largest descriptive R² together with the layerwise and width-head numerical summaries. This is not a trained predictive model and is not evidence that the factor causes the scale error.

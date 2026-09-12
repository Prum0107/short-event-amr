# GT vs hard-negative mechanism audit: cohort definition

The cohort was frozen from the saved coordinate-preserving counterfactual output before mechanism measurements.

- Harmful-hard rule: `HARD_25 center_hit10_2s < mean(RANDOM_25 replicate 0..9 center_hit10_2s)`.
- Control rule: the paired HARD_25 value is greater than or equal to the query's 10-replicate RANDOM_25 mean.
- Primary duration bins: `0-2s` and `2-5s`; no manual query selection was used.
- P3 location failure: no baseline Top-10 candidate center within ±2 s of any GT center.
- Geometry bins: adjacent = interval gap ≤ 1 s; moderately near = > 1 and ≤ 5 s; remote = > 5 s.
- Expanded GT neighborhood: any strongest HARD_25 peak overlap with a GT interval expanded by ±2 s.
- The strongest hard peak is the contiguous temporal-token component containing the highest-saliency selected HARD_25 token. Saliency is recorded only as the selection variable; it is not treated as independent explanatory evidence.
- Native audio-text comparison, when available, uses equal absolute window duration equal to max(1 s, the longest annotated GT interval), with the GT window centered on the longest GT interval and the hard window centered on the strongest hard peak.
- Matched replacement is one pre-specified token-level replacement in RANDOM_25 replicate 0: remove the first deterministic random token and add the strongest selected hard token not already present. A peak-level replacement is not claimed.

## Cohort counts

| Duration bin | Harmful | Control | P3 failure total |
|---|---:|---:|---:|
| 0-2s | 44 | 46 | 46 |
| 2-5s | 127 | 249 | 135 |

## Measurement boundaries

QD-DETR quantities are measured in the official model's internal temporal path. Decoder cross-attention weights are returned by the existing attention implementation and captured without changing its inputs or parameters. Query-conditioned memory response is BLOCKED because the audit has no independently justified common query-memory scoring rule.

Native MS-CLAP is scored only for qids whose original WAV is present. Missing WAVs are retained as BLOCKED rows; stored QD temporal features are never substituted.

Counterfactual source: `saved search-space counterfactual output/query_level_counterfactual.csv`; baseline checkpoint SHA-256 `9cdc18a14e906689484f1dde055b42cdcc4b77f0d850f6a0174ff9ef42063d35`; QD baseline commit `45ef471ee47ea75a2141d75bd9cfdb8c45dfc101`.

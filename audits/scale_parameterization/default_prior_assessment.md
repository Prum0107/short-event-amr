# GPT/default and literature-prior assessment

| Candidate | Default-prior label | Why |
|---|---|---|
| `WIDTH_REPRESENTATION_LOG_SECONDS` | `MEDIUM_DEFAULT_PRIOR` | Log-scale regression is a common coordinate-system choice; the diagnosis-specific part is testing absolute-time duration against this dataset's full-audio normalization failure |
| `INITIALIZATION_LOG_SCALE_COVERAGE` | `HIGH_DEFAULT_PRIOR` | Multi-scale anchors, query priors, and scale-distributed initialization are common defaults |
| `MULTIPLICATIVE_LOG_REFINEMENT` | `HIGH_DEFAULT_PRIOR` | Iterative box refinement and log-scale updates are established design patterns |
| `RELATIVE_SCALE_OPTIMIZATION` | `HIGH_DEFAULT_PRIOR` | Relative, IoU-aware, and duration-balanced localization objectives are common |
| `AUDIO_LENGTH_INVARIANT_EFFECTIVE_CLOCK` | `MEDIUM_DEFAULT_PRIOR` | Duration-aware and context-aware temporal modeling is known, but the explicit invariance condition is more specific |
| `DENSE_BOUNDARY_NON_DETR_CONTROL` | `HIGH_DEFAULT_PRIOR` | Dense boundary, boundary-matching, and anchor-free point-distance formulations are established |

Common prior is not a reason to reject a candidate. It raises the required diagnosis-specific comparison and lowers novelty confidence.

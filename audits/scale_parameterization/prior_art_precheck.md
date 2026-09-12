# Prior-art risk precheck

This is a conservative precheck, not a systematic literature review and not a novelty claim.

| Candidate family | Risk | Obvious overlap to check before implementation |
|---|---|---|
| Width representation in absolute/log seconds | MODERATE–HIGH | Generic box/segment log-scale regression, center-width temporal grounding, and dynamic reference-box updates |
| Initialization geometry | HIGH | Dynamic anchor boxes and multi-scale query/anchor priors |
| Refinement dynamics | HIGH | Iterative box refinement and multiplicative/log-scale updates |
| Optimization geometry | HIGH | IoU/GIoU, scale-balanced, relative-duration, and boundary-aware localization losses |
| Audio-length-invariant clock | MODERATE | Duration-aware temporal grounding and models that normalize or condition temporal context on recording length |
| Dense boundary/non-DETR control | HIGH | Boundary-matching proposals, anchor-free point-to-boundary distances, and dense temporal boundary prediction |

Primary references for the obvious overlaps:

- [DAB-DETR](https://arxiv.org/abs/2201.12329) explicitly uses dynamic anchor boxes as queries and updates them layer by layer, so initialization/reference and iterative scale refinement are high-risk overlap areas.
- [BMN](https://arxiv.org/abs/1907.09702) densely evaluates start/end boundary pairs through a boundary-matching confidence map, making dense boundary controls high-risk.
- [ActionFormer](https://arxiv.org/abs/2202.07925) is an anchor-free temporal localization model that predicts action boundaries at temporal locations, so a dense boundary control is not a novelty-safe direction.
- [A2Net / Revisiting Anchor Mechanisms](https://arxiv.org/abs/2008.09837) uses point-level temporal localization with distances to the start and end boundaries, directly overlapping the point-plus-distance control.

The first intervention should therefore be framed as a diagnosis-driven matched ablation of the current QD-DETR width coordinate, not as a claimed new representation or architecture.

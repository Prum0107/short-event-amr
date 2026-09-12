# Hypothesis assessment

This is a mechanism audit, not a method recommendation. The status rules were fixed before inspecting the resulting tables: a supported rate is at least 50%, a partial rate is 25–49.9%, and fewer than 10 independent native comparisons makes H1/H2 inconclusive. Human annotation ambiguity is inconclusive without supplied labels.

| Hypothesis | Status | Evidence summary |
|---|---|---|
| H1 semantic hard negatives | **NOT_SUPPORTED** | Native HARD>GT rate among valid harmful comparisons: 0.15789473684210525 |
| H2 model-specific calibration failure | **PARTIALLY_SUPPORTED** | Requires native GT>HARD plus QD hard-favored evidence; selection-conditioned QD evidence alone is insufficient |
| H3 near-GT scale leakage | **PARTIALLY_SUPPORTED** | Harmful peaks classified as near/expanded-neighborhood: 0.2631578947368421 |
| H4 repeat or annotation ambiguity | **INCONCLUSIVE** | Human review rows contain no labels |
| H5 decoder attraction | **SUPPORTED** | Harmful full-access top-1 traces flagged as attracted: 0.5380116959064327 |
| H6 hard-negative and scale are separate | **INCONCLUSIVE** | Requires stronger independent semantic coverage and matched scale interpretation |

## Single branch: `DECODER_COMPETITION`

The branch is deliberately INCONCLUSIVE unless the independent evidence is sufficient for a cleaner decision. No method is designed or selected by this audit.

Scale and location are reported separately in `scale_relation.csv`. A hard-negative location effect is not interpreted as proof that width/extent is solved.

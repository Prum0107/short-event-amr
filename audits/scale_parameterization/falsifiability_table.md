# Falsifiability table

| Candidate | If mechanism is correct, first metric expected to improve | Minimal falsification / stop condition |
|---|---|---|
| `WIDTH_REPRESENTATION_LOG_SECONDS` | For proposals whose initial center is within 1 s, median final `abs(log(width/GT))` and final width/GT fall in both 0–2 s and 2–5 s | No reduction in either short bin after a matched pilot, or long-event R1@0.7 drops by ≥5 percentage points without reducing the short width residual |
| `INITIALIZATION_LOG_SCALE_COVERAGE` | Initial width/GT and first-layer contraction burden decrease for short GTs | Short initial width/GT remains unchanged, or 20 s+ initial coverage/Oracle@10 degrades materially |
| `MULTIPLICATIVE_LOG_REFINEMENT` | Decoder-layer width contraction ratio and final residual improve before classification/ranking metrics | Layerwise width trajectory is unchanged, or width improves only through a ranking change with no conditioned span improvement |
| `RELATIVE_SCALE_OPTIMIZATION` | Short-event matched width cost becomes more discriminative and `abs(log(width/GT))` falls | GIoU already explains the change and the relative objective adds no width improvement, or 20 s+ width error increases without a short gain |
| `AUDIO_LENGTH_INVARIANT_EFFECTIVE_CLOCK` | Within a GT-duration bin, the slope of final width/GT and R1@0.7 versus audio duration approaches zero | Duration slope remains, or performance becomes dependent on an implicit short/long gate |
| `DENSE_BOUNDARY_NON_DETR_CONTROL` | Boundary error and duration-conditioned IoU improve without a query-width contraction trajectory | Boundary distributions improve but strict IoU and boundary error do not, or the result cannot be compared without changing semantic inputs/capacity |

These are future pilot criteria. No candidate has been implemented or tested in this derivation.

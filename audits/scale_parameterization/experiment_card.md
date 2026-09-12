# Experiment card: Stage 2 scale mechanism to method-space derivation

- Gate: H — mechanism-to-intervention derivation.
- Primary decision: choose one first mechanism-isolating intervention family; do not implement a method.
- Observed problem: short absolute durations are mapped to extreme small normalized widths while learned initial widths scale with full audio duration; decoder contraction is directionally correct but insufficient.
- Evidence: matched 1,347-query audit; 0–2 s normalized GT width median 0.00469; M1 correctly centered trajectory 43.42→15.24→12.39→4.36 s; longer-audio short-event performance is worse.
- Candidates: six orthogonal families spanning width variable, initialization, refinement, objective, temporal clock, and non-DETR control.
- Primary comparison in a future pilot: `Δ = M(candidate) − M(current geometry)` with shared data, model inputs, capacity, optimizer exposure, evaluation, and long-event side-effect metrics.
- Mechanism metrics: conditioned decoder log-width residual, width/GT, contraction ratio, audio-duration slope, and duration-conditioned IoU.
- Evidence status: exploratory derivation; no candidate has been implemented or trained.
- Kill criterion: candidates without a unique mechanism metric, or requiring an oracle short/long threshold, do not enter the first-line pilot.
- Conclusion ceiling: recommend a first intervention category only; no final architecture, novelty claim, or paper contribution claim.

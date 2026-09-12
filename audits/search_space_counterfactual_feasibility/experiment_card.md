# Coordinate-preserving search-space counterfactual — experiment card

## Primary decision

Decide whether increasing the number and semantic competitiveness of decoder-accessible full-coordinate audio memory positions degrades short-event retrieval, with encoder representations and temporal coordinates held fixed.

## Current Gate and problem

- Gate: H/C — mechanism feasibility leading to a bounded causal comparison.
- Observed problem: short-event failure increases with recording duration; full-audio competition is supported observationally, while coordinate normalization remains confounded.
- Falsifier: if decoder-access restriction changes no retrieval behavior, or if the supposed fixed encoder memory/positions are not invariant, this intervention does not identify the proposed mechanism.

## Treatment and control

- Control: `FULL_ACCESS`, all original valid decoder memory positions.
- Treatments: `GT_ONLY`, `RANDOM_25`, `RANDOM_50`, `RANDOM_100`, `HARD_25`, `HARD_50`, `HARD_100`.
- Estimand: paired within-query metric difference treatment minus `FULL_ACCESS`, with random-subset replicate averaging.

## Fixed variables

Full audio features, duration, TEF, position embeddings, encoder/memory vectors, query features, model weights, initial references, query count, output conversion, query population, and evaluation code.

## Intended change and side effects

The intended change is the decoder key set and its attention normalization. Decoder hidden states, later reference updates, class logits, and spans may change as downstream responses. Encoder memory, positions, and initial references must not change.

## Primary and secondary metrics

- Primary: paired `CenterHit@10 <= 2 s` relative to full access.
- Secondary: positive-overlap candidate availability, `Oracle@10 IoU>=0.5/0.7`, final center error, final width/GT, R1@0.5/0.7, and fixed full-access saliency rank.

## Directional decision rule

Support decoder competition if short-event retrieval metrics decline monotonically as random access grows in both short-duration strata, and the hard-versus-random contrast at matched counts is non-null in the same direction. A null or non-monotonic result is inconclusive about encoder-level competition, not evidence that global search-space effects are absent.

## Kill and stop rules

- Kill the run if positions, encoder memory, initial references, or output conversion differ for reasons other than the decoder access mask.
- Stop after the registered conditions and ten random replicates; no condition or threshold tuning is permitted.
- Do not train, crop, re-normalize, change model parameters, alter encoder context, or turn GT masks into a deployable method.

## Claim ceiling

Even a positive result supports only a decoder-level full-coordinate temporal competition mechanism. It does not prove normalized width is irrelevant, representation is perfect, or global competition is the sole cause of failure.

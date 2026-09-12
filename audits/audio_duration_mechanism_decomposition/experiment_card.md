# Audio-Duration Mechanism Decomposition — Experiment Card

## Objective

Determine which descriptive mechanisms are consistent with the observed degradation of short-event QD-DETR localization as recording duration increases, using the unchanged official baseline checkpoint and inference-only measurements.

## Fixed population and provenance

- Population: the official CASTELLA test JSONL shipped with the migrated baseline, without resampling or query removal.
- Model: the existing official baseline checkpoint; no retraining, weight editing, query editing, method change, or hyperparameter search.
- Audio/text features: the existing baseline features loaded through the official `StartEndDataset` path.
- Primary short-event groups: longest annotated GT interval in `0–2 s` and `2–5 s`.
- GT duration for ratios: longest annotated GT interval; IoU and local evidence use the union of all annotated intervals.
- Audio token duration: the official `clip_length=1 s`.

## Predeclared analyses

1. Stratify short-event queries by fine GT-duration band, GT-count bin, and query-word-length bin; compare shorter-vs-longer recording-duration strata and retain `vid` as the audio cluster identifier.
2. Measure within-audio query-conditioned saliency: GT-token fraction, best GT saliency, GT percentile/rank, strongest false peak, false peaks above GT, local nearby-negative margin, and high-saliency distractor peak count.
3. Within narrow absolute GT-duration bands, compare descriptive explanatory strength of recording duration, normalized GT width, and initial width/GT using fixed linear summaries; no causal interpretation of coefficients.
4. Compare duration-only, initial-scale-only, and joint descriptive models for final width/GT and R1@0.7.
5. Test whether local GT evidence and global competition move differently with recording duration.
6. Audit, but do not run, a same-event context-length counterfactual unless stored features permit changing context without changing the model's position coordinate system or introducing synthetic audio.

## Fixed definitions

- Final prediction metrics use the official post-processing convention: convert normalized center-width outputs to seconds, rank by foreground probability, clip to `[0, duration]`, and round to the official 1-second grid.
- Saliency token `i` covers `[i, i+1)` seconds. A token is GT-associated when it has positive overlap with any GT interval.
- GT percentile is the fraction of valid tokens with saliency no greater than the best GT-associated token, with ties retained in the numerator.
- A false peak is a strict local maximum outside every GT interval. A high-saliency distractor peak is a false peak at or above the within-audio 90th saliency percentile.
- Random-negative difference is best GT saliency minus the mean best saliency of 100 deterministic, same-token-count random non-GT sets when available.

## Decision rules

- H1 normalized-coordinate effect: supported only if normalized GT width remains associated with failure in narrow absolute-GT bands and adds descriptive explanatory value beyond duration/initial scale; otherwise partial or inconclusive.
- H2 search-space competition: supported only if false-peak burden or GT rank worsens with audio duration while local GT evidence is not comparably weakened.
- H3 initial-scale mediation: supported only if adding initial width/GT materially reduces the duration association for short-event failure; report the observed reduction, not a causal mediation claim.
- H4 local evidence degradation: supported only if GT-local percentile/margin declines with recording duration within GT-duration strata.
- H5 dataset composition: supported only if the short-vs-long duration association is substantially attenuated by the declared stratification/matching controls.
- Branch selection: choose one dominant branch only after all fixed summaries are written. If multiple mechanisms remain comparable, use `MULTIPLE_COUPLED_AUDIO_LENGTH_FACTORS`; if the measurements cannot distinguish them, use `INCONCLUSIVE`.

## Stop conditions

Do not train, tune, alter QD-DETR, alter the checkpoint, crop real audio as a proposed solution, design a next method, or run a full-corpus context counterfactual. The context probe is blocked if its implementation would change normalized positional encoding or require zero/synthetic audio in place of omitted context.

## Claim boundary

This is a mechanism decomposition of the existing inference behavior. Associations, rankings, and descriptive model fit do not establish causation. A strong search-space pattern does not prove that coordinate normalization is irrelevant, and a weak local evidence pattern does not by itself identify temporal resolution as the cause.

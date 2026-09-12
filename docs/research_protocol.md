# Diagnostic research protocol

The project follows a problem-first, evidence-graded workflow.

## Measurement before intervention

First verify that the measured signal is scientifically valid and belongs to
the claimed representation space. Dimensional equality is not evidence that
two tensors support cosine similarity. If an official model or API cannot be
traced, the experiment is blocked.

## Fixed comparisons

Use within-audio comparisons whenever the question concerns temporal
localization. Keep audio, query, checkpoint, preprocessing, candidate grid,
random seed, and metric definitions fixed across conditions. A negative window
must be defined independently of the positive result and must respect the
specified overlap exclusion rule.

## Validation gate

Start with a deterministic small subset. Inspect score degeneracy, duration
dependence, preprocessing instability, and rank behavior. Do not select a new
score or window size after viewing the validation results. A failed or
inconclusive gate stops the corresponding full audit.

## Interpretation boundary

An observation about native local audio-text similarity is an observation about
that native API path. It is not automatically an observation about stored
frame-wise features, a learned projection, QD-DETR encoder memory, decoder
attention, or final AMR output.

## Reporting

Every report should distinguish exact observations, documented conclusions,
inferences, and unknowns. Negative results and blocked experiments remain
versioned because they prevent invalid follow-up claims.

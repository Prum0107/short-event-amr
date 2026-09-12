# DCASE 2026 Task 6 Research Direction

## 1. Direction

Our future research direction is:

```text
Query-Guided Temporal Semantic Evidence Learning for Audio Moment Retrieval
```

中文可以表述为：

```text
面向音频片段检索的查询引导时序语义证据学习
```

The core idea is not only to predict a temporal window for a text query, but also to learn and expose the semantic evidence on the audio timeline that supports the prediction.

In short:

```text
Baseline learns the answer.
We want to learn the temporal semantic evidence behind the answer.
```

## 2. Core Research Question

Given a long audio recording and a natural-language query, can an audio-language model learn where the query is semantically supported on the time axis?

More specifically:

```text
For each query q and each time position t,
can the model estimate S(t, q),
the degree to which the audio at time t supports the query?
```

This changes the task from direct boundary prediction:

```text
audio + query -> [start, end]
```

to evidence-based temporal grounding:

```text
audio + query
  -> temporal semantic evidence curve S(t, q)
  -> boundary evidence
  -> final [start, end]
```

## 3. Why This Direction

DCASE 2026 Task 6 is not ordinary audio-text retrieval. It is temporal grounding in long audio.

A model must know three things at the same time:

```text
1. What semantic content the query describes.
2. Where this semantic content is supported in the audio timeline.
3. Where the corresponding moment should start and end.
```

Strong semantic understanding alone is not enough. A clip-level audio-language model may know that an audio recording contains cheering, speech, music, or engine sounds, but it may not know exactly when the query becomes true and when it stops being true.

Therefore, the key target is:

```text
time-localized semantics
```

or:

```text
semantic evidence with temporal coordinates
```

## 4. Lessons From Previous Systems

Previous work has already explored several competition-oriented systems:

```text
Binary Hamming hash retrieval
Continuous audio-text retrieval with CSD-style auxiliary loss
Text-to-audio feature prediction
QD-DETR proposal fusion
Multi-source reranking
Conservative gates
Event decode boundary correction
```

These systems produced useful results, but they also revealed important lessons.

### 4.1 Proposal generation is not the only bottleneck

Multi-source proposal systems often contain a correct or near-correct candidate in the top-k list. Oracle analysis showed that many failures are not because the system cannot find any relevant segment, but because it cannot reliably rank the correct segment as top-1.

This suggests that the key problem is not only:

```text
Can we generate more candidates?
```

but:

```text
Can the model understand why one candidate truly supports the query better than other similar candidates?
```

### 4.2 Event-first localization is not sufficient

Event proposals can produce natural audio boundaries, but audio changes are not the same as query-relevant events.

For example, in a query such as:

```text
A man talks while an engine is running.
```

a pure event-first model may detect speech, engine noise, or a sound change, but the correct moment may be the time interval where both semantic conditions are satisfied.

Thus, event boundaries are useful, but they should not be the main decision maker. They are better treated as boundary evidence or diagnostic signals.

### 4.3 Rerankers and gates can become engineering tricks

Previous systems found that rerankers, conservative gates, and ensembles can improve validation performance, but they may also become unstable or over-tuned.

For future research, these components should not be the main contribution. They can be used for analysis, hard negative mining, or diagnostic comparison, but the main research story should be cleaner.

### 4.4 CLAP score curves are useful but under-explained

Previous retrieval systems already produced clip-level similarity curves:

```text
query embedding vs audio clip embeddings -> score over time
```

However, these curves were mainly used as proposal generators. The new direction is to study whether such curves can become meaningful temporal semantic evidence:

```text
Does a high score at time t really mean that the query is semantically supported there?
```

## 5. Difference From The Official Baseline

The official baseline can be viewed roughly as:

```text
MS-CLAP sliding-window features
  + text features
  -> QD-DETR-style temporal localization
  -> predicted [start, end]
```

Its semantic alignment is largely implicit. The model is trained to output the right temporal window, but it is not explicitly required to show which parts of the audio timeline support the query.

Our direction is:

```text
audio temporal features + query
  -> explicit evidence curve S(t, q)
  -> boundary evidence
  -> evidence-aware moment prediction
```

The conceptual difference is:

```text
Baseline: answer supervision
Ours: answer supervision + temporal evidence learning
```

The goal is not only to improve the final metric, but also to make the model's temporal grounding behavior analyzable.

## 6. Proposed Modeling Framework

The basic framework should be simple and research-focused:

```text
Long audio
  -> frame-level or segment-level audio features

Text query
  -> query embedding

Audio features + query embedding
  -> query-guided temporal interaction
  -> evidence curve S(t, q)
  -> start probability
  -> end probability
  -> final moment prediction
```

The model may use official CLAP features at the beginning. Raw audio and stronger encoders can be considered later, but they should not distract from the research question.

## 7. Training Signals

The training objective should encourage the model to learn evidence, not only boundaries.

### 7.1 Dense temporal relevance supervision

For each query, create a soft relevance label over time:

```text
Inside the ground-truth window: high relevance
Outside the ground-truth window: low relevance
Near boundaries: soft transition
```

This supervises the evidence curve directly.

### 7.2 Boundary-aware supervision

The model should also learn where the moment starts and ends:

```text
start probability
end probability
boundary quality
IoU-aware score
```

This is important because DCASE metrics, especially R1@0.7, strongly depend on boundary accuracy.

### 7.3 Hard negative contrastive learning

Hard negatives should come from previous systems and from the same audio:

```text
Semantically similar but temporally wrong segments
Segments that contain only part of the query semantics
Nearby segments with shifted boundaries
Event-like segments that have good boundaries but wrong semantics
Retrieval candidates that score high but miss the ground truth
```

The model should learn not only what is relevant, but also why similar alternatives are not correct.

## 8. Analysis Goals

Since this is a research-oriented project, analysis is as important as final score.

Important analysis questions:

```text
1. Does S(t, q) peak inside the ground-truth window?
2. Does the evidence curve change when the query changes?
3. Can the model distinguish full semantic matches from partial matches?
4. Are boundary errors caused by weak semantics or weak temporal modeling?
5. Do hard negatives improve the quality of the evidence curve?
```

Useful visualization examples:

```text
Same audio, different queries:
  query A: a man talking
  query B: wind blowing
  query C: a man talking while wind is blowing

Expected behavior:
  S(t, query C) should be high mainly where speech and wind overlap.
```

## 9. What We Will Not Make The Main Direction

To avoid repeating previous mistakes, the new project should not center on:

```text
More multi-branch fusion
Validation-tuned gates
Pure event-first segmentation
Large audio LLM reranking as the main system
Simple encoder ensemble
Leaderboard-only optimization
```

These ideas can still be used as tools, baselines, or diagnostic modules, but they should not define the research contribution.

## 10. What We Keep From Previous Work

The previous systems are still valuable. We keep:

```text
Official baseline comparison
Existing CLAP feature pipeline
Retrieval score curve generation
Proposal decoder utilities
Oracle analysis tools
Candidate pools for hard negative mining
Event decode outputs for boundary diagnostics
```

The old systems should become supporting infrastructure for the new research question.

## 11. First-Stage Experimental Plan

### Stage 1: Build the evidence baseline

Use official CLAP features and train a simple model that outputs:

```text
S(t, q)
start probability
end probability
```

Evaluate both localization metrics and evidence quality.

### Stage 2: Add dense relevance supervision

Compare:

```text
boundary-only training
vs
boundary + dense evidence supervision
```

Check whether the evidence curve becomes more aligned with the ground-truth moment.

### Stage 3: Add hard negative learning

Use candidates from old hash, continuous retrieval, QD-DETR, T2A, and event_decode systems as hard negatives.

Study whether the model becomes better at rejecting:

```text
partial semantic matches
near-boundary mistakes
high-score but wrong retrieval candidates
```

### Stage 4: Interpretability and error analysis

Create visual cases showing:

```text
correct grounding
semantic confusion
partial match errors
boundary errors
background interference
query-dependent evidence changes
```

This stage is central to the research story.

## 12. Possible Paper/Report Framing

A possible title:

```text
Learning Query-Guided Temporal Semantic Evidence for Audio Moment Retrieval
```

A possible one-sentence abstract:

```text
We study audio moment retrieval from the perspective of temporal semantic evidence, asking whether a model can not only predict the query-relevant moment, but also identify where and how the query is supported along the audio timeline.
```

A possible contribution statement:

```text
1. We formulate audio moment retrieval as query-guided temporal semantic evidence learning.
2. We introduce an explicit evidence curve S(t, q) supervised by temporal relevance labels.
3. We use hard negative segments from previous retrieval and localization systems to improve evidence discrimination.
4. We analyze how temporal semantic evidence explains correct predictions, partial matches, and boundary errors.
```

## 13. Final Summary

The new direction is a shift from competition engineering to research.

Previous systems showed that multi-source proposals and reranking can improve performance, but they also exposed a deeper weakness:

```text
The model often lacks a clear understanding of why a specific time segment semantically supports the query.
```

Therefore, the future work should focus on:

```text
learning, supervising, and analyzing query-specific temporal semantic evidence.
```

This gives the project a cleaner research identity:

```text
not just finding the answer,
but understanding the time-localized semantic evidence that makes the answer valid.
```

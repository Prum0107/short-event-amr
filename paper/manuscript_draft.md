# Diagnosing Short-Moment Failures in Audio Moment Retrieval with Evidence-Guided Candidate Generation and Ranking

## Abstract

**Problem.** Audio Moment Retrieval (AMR) requires a system to locate the temporal interval in an audio recording that corresponds to a natural-language query. Aggregate retrieval scores do not reveal whether short-moment failures arise from the representation, from missing candidates, or from ranking.

**Gap.** We investigate the short-moment failure regime through duration-stratified evaluation, candidate recall, oracle candidate quality, and the gap between candidate-pool quality and the selected prediction.

**Diagnosis.** On the reproduced CASTELLA QD-DETR evaluation, short moments are substantially harder. The staged audit indicates that query-conditioned temporal evidence can support regions absent from the original proposal pool, while adding candidates without a compatible ranker can increase candidate recall but reduce final retrieval.

**Method.** We evaluate Evidence-Guided Candidate Generation and Ranking (EGCG), which keeps the QD-DETR encoder, backbone, and localization components fixed while adding a learned temporal Evidence Head, evidence-derived candidate proposals, candidate fusion, and an evidence-aware ranker.

**Results.** On the official test split, full EGCG improves QD-DETR R1@0.7 from 18.93% to 26.80%. In the 0–2 s bin, the result changes from 1.11% to 21.11% (N=90). The UVCOM comparison is mixed, and proposal-level evidence is not treated as semantic proof. These results support a benchmark-scoped, architecture-conditional interpretation of explicit temporal evidence modeling.

# 1 Introduction

Audio Moment Retrieval (AMR) studies the problem of locating, within a long audio recording, the interval that matches a natural-language description. The task combines audio-text alignment with temporal localization: a system must identify not only whether a queried acoustic event is present, but also where that event occurs and how long the relevant moment lasts. This temporal requirement makes AMR evaluation sensitive to the quality and ranking of candidate intervals, especially when the target moment is short relative to the recording.



Progress measured only by an overall retrieval score can leave the temporal failure mechanism unresolved. A system may obtain a usable aggregate score while failing disproportionately on a particular duration regime, and a final Top-1 error does not by itself reveal whether no relevant candidate was generated or whether a relevant candidate was generated but ranked incorrectly. We therefore organize the analysis around duration-stratified performance, candidate recall, oracle candidate quality, and the gap between candidate-pool quality and the selected prediction.



The archived CASTELLA test evaluation exposes a pronounced short-moment failure for the reproduced QD-DETR baseline. QD-DETR reaches 18.93% R1@0.7 over the full test set, but only 1.11% for 0–2 s moments and 6.91% for 2–5 s moments. Thus, the overall score does not represent the behavior of the shortest targets. The 0–2 s result is reported with its archived denominator, N=90, and is interpreted as a benchmark-scoped failure regime rather than as a universal property of all AMR data.



We next test competing explanations in sequence. The representation control shows that improving the representation helps overall AMR but does not eliminate the short-scale degradation. The geometry and candidate decomposition then indicate that missing or unsuitable temporal candidates account for a major part of short-moment failure. The evidence-candidate audit further reports cases in which query-conditioned temporal evidence supports relevant regions that are not represented by the original QD-DETR candidates. However, the naive augmentation result shows that candidate availability alone is insufficient: adding evidence candidates without adapting selection can increase candidate recall while reducing final retrieval. The resulting diagnosis contains two linked bottlenecks—candidate generation and candidate ranking—without claiming that they are independent causes for every architecture.



This diagnosis motivates Evidence-Guided Candidate Generation and ranking (EGCG). The intervention is designed to test whether an explicit temporal evidence pathway can make query-relevant regions available to the candidate pool and whether evidence-aware selection can convert that recovered availability into final retrieval. EGCG therefore has two linked roles: it generates additional evidence-supported candidate moments and provides a ranking signal compatible with the fused candidate pool. The baseline encoder, QD-DETR backbone, and localization components remain fixed in the archived implementation, so the intervention is tied to the diagnosed evidence-to-candidate and candidate-selection problem.



This paper makes four scoped contributions. First, it provides a duration-stratified diagnosis of short-moment AMR failure on CASTELLA, separating representation, candidate availability, and ranking/selection evidence. Second, it audits whether query-conditioned temporal evidence can be converted into relevant candidate moments, including the negative result that naive candidate augmentation does not guarantee final retrieval improvement. Third, it evaluates the clean EGCG component ablation in which candidate generation and evidence-aware ranking are separated under frozen implementation semantics. Fourth, it reports the official test result, efficiency measurements, and a mixed cross-model boundary, making explicit what the evidence does not establish.

# 2 Related Work

This first integrated draft organizes the related-work discussion around the interfaces that are evaluated in this study. The supplied archive contains experiment and implementation records, but not a verified external bibliography; citation placeholders should therefore be populated only from an independently checked reference set during a later revision.

## Audio Moment Retrieval

AMR combines audio-text matching with temporal localization. In contrast to clip-level retrieval, the output must identify an interval whose temporal boundaries agree with an annotated moment. The present work focuses on the diagnostic consequence of that requirement: the same semantic query can be difficult to retrieve when its target interval is short, even when the recording contains relevant acoustic information.

## Temporal proposal generation and localization

The archived QD-DETR baseline provides the reference proposal and localization pipeline. Our analysis separates the generation of candidate intervals from their final selection. Candidate recall and oracle metrics quantify whether the proposal pool contains a sufficiently overlapping interval, while R1 metrics quantify the selected Top-1 result. This separation is central to the diagnosis and is not intended as a replacement taxonomy for the broader AMR literature.

## Evidence-guided selection

EGCG is positioned as a lightweight evidence pathway attached to a fixed AMR system. It uses query-conditioned temporal scores to add candidates and then uses evidence-compatible ranking to select from the fused pool. The contribution is evaluated as a controlled intervention on the diagnosed candidate-generation and ranking interfaces, rather than as a claim of a new backbone or a universal AMR architecture.

# 3 Understanding Short-Moment AMR Failures

## 3.1 Short moments remain challenging


We first examine the reproduced QD-DETR baseline by target duration rather than relying on a single aggregate score. On the official CASTELLA test split, QD-DETR obtains 18.93% R1@0.7 overall. The rate falls to 6.91% for 2–5 s moments and to 1.11% for 0–2 s moments. The archived audit counts 90 queries in the 0–2 s bin, so the shortest-bin result is reported with its denominator and treated as a benchmark-scoped observation.



The duration effect establishes where the system fails, but not why. A low Top-1 IoU can result from a proposal pool that does not contain a sufficiently relevant interval, from a relevant interval that is ranked below competing candidates, or from a candidate whose boundaries are too coarse for a short target. We therefore treat Top-1 retrieval as the endpoint of a candidate-generation and selection process and use candidate-level diagnostics to distinguish these possibilities.


## 3.2 Temporal evidence exists but is under-utilized


We next ask whether short-moment failure can be attributed primarily to the representation. The M2D feature-only control improves overall AMR performance under the QD-DETR framework, but the short-scale degradation remains. This result shifts the analysis from an encoder-only explanation toward the temporal interface between representations and candidate moments. It does not imply that representation quality is unimportant; it shows that the archived improvement in representation is not sufficient to remove the observed short-moment failure.



The subsequent evidence-candidate analysis evaluates whether the available representation contains query-conditioned temporal information even when the original proposal set fails. Across the archived evidence audit and oracle study, evidence-derived temporal regions recover relevant candidate intervals that are missing from the original QD-DETR pool in a non-trivial subset of cases. We use “evidence” here as an operational proposal score: the result supports candidate recovery, but does not establish that the score is a complete semantic explanation of the target acoustic event.


## 3.3 Candidate generation bottleneck


The geometry analysis provides the first direct candidate-generation diagnosis. For short moments, positive-overlap predictions often cover part of the target region while extending beyond it, and the 0–5 s decomposition identifies missing relevant candidates as the dominant failure component. This pattern is compatible with a proposal system that has coarse temporal coverage or insufficiently targeted intervals, rather than with a failure caused only by the final score assigned to an otherwise suitable candidate.



Evidence proposal oracle and prototype studies then test the candidate-generation hypothesis without changing the AMR model. Adding query-conditioned evidence proposals raises candidate-level recall relative to the original proposal pool in the archived analyses, including the short-duration regime. This result is an upper-bound and candidate-quality observation: it shows that evidence-derived intervals can recover regions useful for retrieval, but it does not yet show that the existing ranker will select them.


## 3.4 Ranking mismatch


The next experiment separates candidate availability from candidate selection. When evidence-generated candidates are inserted into the original QD-DETR ranking pipeline, candidate recall increases but final R1 decreases. The discrepancy shows that the original ranking scores are not necessarily calibrated for the expanded candidate pool. Consequently, a candidate-generation intervention can expose useful temporal regions without improving the selected moment if the scoring mechanism does not recognize those regions.



The oracle analysis makes the remaining gap explicit. After evidence-guided candidate recovery, the candidate pool can contain intervals with substantially better overlap than the Top-1 prediction, especially for short moments. The gap therefore motivates a second intervention directed at ranking and selection rather than another change to the audio or text encoder. Because the oracle is an upper bound, this observation is diagnostic evidence for a selection bottleneck, not a claim that every recovered candidate is semantically correct.


## 3.5 Summary of diagnosis


The archived diagnosis yields a two-stage failure picture for the tested QD-DETR setting. First, short moments are frequently absent from or poorly represented in the original candidate pool even when query-conditioned temporal evidence can support relevant regions. Second, adding candidates without compatible scoring does not reliably improve final retrieval. The next section therefore evaluates a single linked intervention: explicit evidence-guided candidate generation followed by evidence-aware candidate ranking, while holding the baseline encoder and localization backbone fixed. The intervention is tested through clean candidate-pool semantics and an official test evaluation rather than through the diagnostic oracle alone.

# 4 Method

## 4.1 Problem Formulation

**Purpose.**

We formulate AMR as query-conditioned temporal retrieval and make the candidate-generation/selection interface explicit. Given an audio recording x with duration D and a text query q, the system predicts a temporal window g_hat = [s_hat, e_hat] corresponding to one or more annotated ground-truth windows G = {g_j = [s_j, e_j]}. The frozen QD-DETR baseline produces a proposal set C_QD(q) and associated scores.

**Formulation.**

For a candidate c = [s, e], temporal overlap is defined as:

```text
IoU(c, g) = |c intersection g| / |c union g|.
```

The method separates two operations:

1. **Candidate availability:** construct a pool C(q) that contains intervals with high overlap to at least one g_j.
2. **Candidate selection:** assign a ranking score R_phi(c, q) and select the highest-scoring candidate for the final prediction.

The temporal evidence pathway estimates a query-conditioned score E(t, q) over one-second temporal tokens. EGCG augments the baseline pool with intervals derived from this score and then applies evidence-aware candidate ranking. The baseline audio/text encoders, QD-DETR backbone, and localization components remain fixed in the frozen implementation.

**Implementation details.**

The final implementation uses frozen QD-DETR projected audio states of width 256 and projected query/text states of width 256 for temporal evidence. The ranker separately uses mean M2D audio features and mean normalized M2D text features of width 768, as specified in Sections 4.3–4.4.

**Connection to the diagnosis.**

R3/R4 identify short-moment candidate and geometry failures, while R16 shows a gap between candidate-pool quality and Top-1 selection. This formulation makes those two diagnostic quantities explicit without treating an oracle score as an achieved prediction. [R3, R4, R16; R32 diagnosis_chain.md; R33 claims C3/C5.]

## 4.2 Temporal Evidence Modeling

**Purpose.**

The Evidence Head provides an explicit, query-conditioned temporal score that can be converted into candidate intervals. It is added to the frozen QD-DETR projected states; it does not replace the audio encoder, text encoder, backbone, or localization head.

**Formulation.**

Let a_t in R^256 be the projected audio state for one-second token interval I_t = [t, t+1), and let q_p in R^256 be the projected query state. The recovered Evidence Head computes:

```text
u_t = W_a a_t + b_a
v   = W_q q_p + b_q
z_t = concat(u_t, v, u_t elementwise_product v) in R^384
l_t = W_2 ReLU(W_1 z_t + b_1) + b_2
E_t = sigmoid(l_t)
```

The audio and query projections are 256 -> 128. The classifier is 384 -> 128 -> 1 with a ReLU between its linear layers. E_t is the evidence probability for token t.

For a ground-truth set G, the token target is:

```text
y_t = 1 if there exists g_j in G such that I_t has non-empty overlap with g_j;
      0 otherwise.
```

Only valid audio-mask positions contribute to the masked binary cross-entropy. The positive class uses weight 5.0. The evidence loss is denoted L_evidence.

**Implementation details.**

Evidence Head training uses AdamW with learning rate 1e-3, weight decay 1e-4, three epochs, batch size 16, gradient clipping at 0.1, and seed 2023. The archived training relation is:

```text
L_total = L_QD + lambda * L_evidence,     lambda = 0.5.
```

QD-DETR is frozen during this training. The final evidence checkpoint is the frozen artifact associated with EGCG v1.0.

**Connection to the diagnosis.**

R5–R7 indicate that query-conditioned temporal evidence can support candidate regions missing from the original pool. R14 shows that the minimal Evidence Head can learn temporal alignment, while R15 connects the learned evidence to proposal generation. The Evidence Head therefore operationalizes the diagnosed evidence-to-candidate interface; it is not presented as a semantic ground-truth detector. [R5–R7, R14–R15; R32 diagnosis_chain.md; R33 claims C4/C11.]

## 4.3 Evidence-guided Candidate Generation

**Purpose.**

This component converts the token-level evidence probabilities into temporal proposal windows and fuses them with the original QD-DETR proposals. The goal is candidate recall, not final ranking.

**Formulation.**

For a query, let V(q) be the valid audio-mask token positions. The per-query threshold is the 90th percentile of the evidence probabilities on valid positions:

```text
tau_q = P90({E_t : t in V(q)})
T_q   = {t in V(q) : E_t >= tau_q}.
```

Contiguous selected tokens are grouped into regions. A run from token t_start through t_end produces:

```text
c_E = [t_start, t_end + 1]
s_E(c_E) = mean({E_t : t belongs to the region}).
```

Each region is clipped to [0, D], and the evidence proposal set is capped at 100 intervals. The fused pool is:

```text
C(q) = C_QD(q) union C_E(q).
```

QD and evidence proposals are merged using the exact (start, end) interval key. Duplicate intervals retain the stronger stored evidence score. The stored fusion score is:

```text
s_fuse(c) = 0.5 * s_QD(c) + 0.5 * s_E(c).
```

The score fields and source handling are those of the frozen implementation.

**Implementation details.**

The evidence threshold is per query, regions use one-second token boundaries, clipping uses the dataset/video duration, and the maximum number of evidence proposals is 100. The proposal generator and fusion implementation are frozen at the stated commit. No raw audio is required by this candidate-generation description; it operates on the available temporal states and evidence probabilities.

**Connection to the diagnosis.**

R6/R7/R9 show that evidence-derived intervals can increase candidate-level recall, motivating this conversion rule. R10 also shows why this component must not be interpreted as a complete AMR solution: a larger candidate pool can degrade final retrieval under an incompatible ranker. [R6–R7, R9–R10, R15; R32 diagnosis_chain.md; R33 claims C4/C5.]

## 4.4 Evidence-aware Ranking

**Purpose.**

The ranker scores the fused candidate pool using candidate audio content, query content, QD-DETR score, evidence score, and candidate geometry. It addresses the mismatch exposed when evidence candidates are added without a compatible selection mechanism.

**Formulation.**

For candidate c = [s, e], let v_c in R^768 be the mean M2D audio feature over the candidate interval, and let u_q in R^768 be the mean normalized M2D text embedding for the query. The scalar feature vector is:

```text
r_c = [s_QD(c), s_E(c), s / D, e / D, (e - s) / D].
```

The candidate and query projections are:

```text
p_c = W_c v_c + b_c in R^64
p_q = W_q u_q + b_q in R^64
h_c = concat(p_c, p_q, p_c elementwise_product p_q, r_c) in R^197.
```

The ranker is:

```text
R_phi(c, q) = W_o ReLU(W_h h_c + b_h) + b_o,
```

with layers 197 -> 128 -> 1. During ranker training, a positive candidate satisfies IoU(c, g_j) >= 0.5 for the relevant matched ground truth, while a negative candidate has IoU below 0.5. For a positive/negative pair, the pairwise hinge loss is:

```text
L_rank = max(0, margin - R_phi(c_plus, q) + R_phi(c_minus, q)),
margin = 0.1.
```

**Implementation details.**

Ranker training uses AdamW with learning rate 1e-3, weight decay 1e-4, five epochs, batch size 8, gradient clipping at 0.1, and seed 2023. The ranker uses the 768-dimensional mean M2D audio feature, the 768-dimensional mean normalized M2D text embedding, the two score fields, and the three normalized temporal geometry fields exactly as specified above.

**Connection to the diagnosis.**

R10 identifies a mismatch between the original QD-DETR ranking and evidence-generated candidates. R11 shows that evidence-aware score fusion can recover part of the candidate gain, R16 shows a remaining oracle-to-Top-1 gap, and R17 motivates a trainable ranker. The ranker is therefore the selection-side counterpart to the Evidence Head and proposal generator. [R10–R11, R16–R17; R32 diagnosis_chain.md; R33 claim C5.]

## 4.5 Training Objective and Inference

**Purpose.**

This subsection states how the frozen components are trained and how the final inference path uses the learned evidence and ranking artifacts. It records the implementation protocol without reporting experimental outcomes.

**Formulation.**

The Evidence Head is trained with the archived auxiliary objective:

```text
L_total = L_QD + 0.5 * L_evidence,
```

where the evidence term is masked BCE-with-logits with positive weight 5.0. QD-DETR is frozen during this training. The ranker is then trained with L_rank from Section 4.4 using train-split candidate IoUs. At inference, evidence probabilities generate C_E, the pool is fused with C_QD, and the evidence-aware ranker assigns R_phi(c, q) to fused candidates before the frozen downstream moment-selection/localization path produces the final prediction.

**Implementation details.**

The frozen EGCG v1.0 implementation is identified by commit `07d52c8d8fc2508f6899cd43b3f404add24db719`. The QD-DETR baseline remains fixed; Evidence Head and Ranker checkpoints are loaded without test-time parameter updates. The final clean inference semantics distinguish: original QD proposals/ranking; QD-only proposals with a no-evidence ranker; QD plus evidence proposals with the same no-evidence ranker; and full EGCG with the evidence-aware ranker. Their numerical comparison belongs to the Experiments section, not this Method draft.

**Connection to the diagnosis.**

The staged design follows the archived failure chain: R14–R15 establish learnable evidence and proposal connection, R16–R17 identify selection as the remaining bottleneck, and R29–R31 enforce clean candidate-pool semantics and frozen test evaluation. This makes the method a direct implementation of the diagnosed two-stage interface rather than a replacement of the baseline architecture. [R14–R17, R29–R31; R32 diagnosis_chain.md; R33 claims C3–C6.]

## Implementation provenance

The implementation authority is the recovered R14/R15/R17 source and forensic metadata recorded in the R32 archive. The frozen clean EGCG v1.0 commit is `07d52c8d8fc2508f6899cd43b3f404add24db719`; the QD-DETR baseline is frozen separately. No implementation details from the R13 design document are substituted for recovered source-level choices.

# 5 Experiments

## 5.1 Experimental Setup

### Dataset and task

We evaluate Audio Moment Retrieval on the official CASTELLA test split used by the final archived evaluation. The task is to retrieve a temporal audio interval corresponding to a natural-language query. The final runtime table records 1,347 queries. Duration-stratified analysis uses the archived 0–2 s and 2–5 s target-duration bins; the 0–2 s bin contains 90 queries and the 2–5 s bin contains 376 queries in the archived audit.


### Systems and baselines

The primary comparison is between the original QD-DETR baseline and full EGCG v1.0. The clean ablation defines four systems:

- **A — QD-DETR:** original QD-DETR proposals and original QD-DETR ranking.
- **B — No-evidence ranker:** original QD-DETR proposals only, with the frozen no-evidence ranker and no evidence-derived candidates or evidence features.
- **C — Evidence candidate generation:** QD-DETR proposals fused with Evidence Head proposals, ranked with the same no-evidence mechanism as B and without evidence features.
- **D — Full EGCG:** QD-DETR plus Evidence Head proposals, ranked with the full evidence-aware ranker.

The A–D definitions follow the R29 semantics audit and the clean R30/R31 protocol. The earlier R28 shared-pool ranking-input comparison is not used as the causal ablation.


### Metrics

We report final retrieval with R1@0.5, R1@0.7, and mAP. Candidate diagnostics include CandidateRecall@10@0.5, CandidateRecall@100@0.5, Oracle@10@0.7, and the Top1–Oracle@10@0.7 gap. R1@τ denotes the proportion of queries whose selected Top-1 moment has temporal IoU at least τ with the matched ground truth. Candidate recall and oracle metrics are reported as diagnostics and upper-bound indicators, not as substitutes for final Top-1 retrieval.


### Implementation and data hygiene

The QD-DETR baseline and EGCG v1.0 are frozen. The final test evaluation uses frozen Evidence Head and Ranker checkpoints; no test-split parameter updates or hyperparameter search are performed. The EGCG implementation is identified by commit `07d52c8d8fc2508f6899cd43b3f404add24db719`. The archive records QD-DETR parameters of 7,123,237, Evidence Head parameters of 115,201, and Ranker parameters of 123,905. Detailed runtime and training-cost accounting appears in Section 5.5.


## 5.2 Main Results
# Table 1. Main final test results


| System | Candidate source | Ranking | R1@0.5 | R1@0.7 | mAP |
|---|---|---|---:|---:|---:|
| QD-DETR | Original QD-DETR proposals | Original QD-DETR ranking | 38.75% | 18.93% | 16.26% |
| Full EGCG | QD-DETR + Evidence Head proposals | Evidence-aware ranker | 47.51% | 26.80% | 24.43% |

Scope: frozen EGCG v1.0 checkpoints, official test split, no test-time parameter updates.


Table 1 compares the original QD-DETR system with full EGCG on the official test split. Full EGCG raises R1@0.5 from 38.75% to 47.51%, R1@0.7 from 18.93% to 26.80%, and mAP from 16.26% to 24.43%. These values are the final frozen test results, not a new evaluation performed for this draft.

The comparison answers the primary evaluation question within the tested CASTELLA/QD-DETR setting: the complete evidence pathway is associated with higher final retrieval metrics than the original baseline. Component attribution is deferred to the clean A–D ablation in Section 5.3.


## 5.3 Ablation Study
# Table 2. Clean R31 A–D ablation


| System | R1@0.5 | R1@0.7 | mAP | CandidateRecall@10@0.5 | CandidateRecall@100@0.5 | Oracle@10@0.7 | Top1–Oracle@10@0.7 gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| A — QD-DETR | 38.75% | 18.93% | 16.26% | 64.37% | 64.37% | 37.56% | 18.63% |
| B — No-evidence ranker | 41.13% | 20.64% | 16.78% | 64.37% | 64.37% | 37.56% | 16.93% |
| C — Evidence candidate generation | 40.98% | 21.08% | 20.38% | 76.69% | 83.96% | 54.42% | 33.33% |
| D — Full EGCG | 47.51% | 26.80% | 24.43% | 79.29% | 83.96% | 57.31% | 30.51% |

A/B share the original QD-DETR candidate pool. C adds evidence candidates but uses the no-evidence ranking mechanism. D is the full evidence proposal plus evidence-aware ranking system.


Table 2 separates candidate generation from ranking. A and B use the same original QD-DETR candidate pool, and their CandidateRecall@10@0.5 and CandidateRecall@100@0.5 are both 64.37%. B nevertheless raises R1@0.7 from 18.93% to 20.64%, which records a ranking-only change without a candidate-pool change.

System C adds Evidence Head proposals while retaining the no-evidence ranking mechanism. CandidateRecall@10@0.5 rises to 76.69% and CandidateRecall@100@0.5 to 83.96%; Oracle@10@0.7 rises from 37.56% for A/B to 54.42%. Its Top-1 R1@0.7 is 21.08%, and the Top1–Oracle gap remains 33.33%. Thus, evidence-guided proposals address the candidate-availability side of the diagnosed failure, but the candidate gain is not sufficient by itself to close the selection gap.

System D adds the evidence-aware ranker to the fused candidate pool. It reaches CandidateRecall@10@0.5 of 79.29%, CandidateRecall@100@0.5 of 83.96%, Oracle@10@0.7 of 57.31%, and R1@0.7 of 26.80%, with a Top1–Oracle gap of 30.51%. In the clean comparison, C isolates candidate generation and D adds the ranking component; the table therefore supports a linked interpretation rather than attributing the full result to candidate availability alone.


## 5.4 Short-Moment Analysis
# Table 3. Short-duration final test results


| System | GT-duration bin | N | R1@0.5 | R1@0.7 | mAP | CandidateRecall@10@0.5 | CandidateRecall@100@0.5 | Oracle@10@0.7 | Top1–Oracle@10@0.7 gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — QD-DETR | 0-2s | 90 | 14.44% | 1.11% | 2.34% | 16.67% | 16.67% | 1.11% | 0.00% |
| A — QD-DETR | 2-5s | 376 | 26.33% | 6.91% | 9.38% | 43.88% | 43.88% | 14.10% | 7.18% |
| B — No-evidence ranker | 0-2s | 90 | 13.33% | 1.11% | 2.29% | 16.67% | 16.67% | 1.11% | 0.00% |
| B — No-evidence ranker | 2-5s | 376 | 26.33% | 7.18% | 9.70% | 43.88% | 43.88% | 14.10% | 6.91% |
| C — Evidence candidate generation | 0-2s | 90 | 17.78% | 5.56% | 11.62% | 54.44% | 72.22% | 36.67% | 31.11% |
| C — Evidence candidate generation | 2-5s | 376 | 33.24% | 11.97% | 18.43% | 73.14% | 83.24% | 48.67% | 36.70% |
| D — Full EGCG | 0-2s | 90 | 35.56% | 21.11% | 26.81% | 67.78% | 72.22% | 46.67% | 25.56% |
| D — Full EGCG | 2-5s | 376 | 43.88% | 21.01% | 24.11% | 76.33% | 83.24% | 52.13% | 31.12% |

N=90 for 0–2 s and N=376 for 2–5 s are the archived query denominators. Percentages are not pooled across bins.


Table 3 evaluates the duration regime identified in the diagnosis. For 0–2 s moments, QD-DETR A obtains R1@0.7 of 1.11%. Evidence candidate generation C raises this value to 5.56%, while full EGCG D reaches 21.11%. For 2–5 s moments, the corresponding values are 6.91% for A, 11.97% for C, and 21.01% for D. The short-bin results follow the same staged pattern as the overall ablation: proposal augmentation improves candidate-level availability, while the full pipeline produces the largest selected-moment result.

The 0–2 s result should be read with its archived N=90 denominator. The table is evidence that the diagnosed short regime improves under the frozen EGCG pipeline on this test split; it is not evidence for a universal short-event law or for architecture-independent generalization.


## 5.5 Efficiency Analysis
# Table 4. Complexity, runtime, memory, and training cost

Sources: R32 `final_tables/efficiency.csv`, `egcg_specification.md`, and R27 runtime records in `metrics_database.csv`.

## Parameter accounting

| Component | Parameters | Additional parameters | Additional percentage over QD-DETR |
|---|---:|---:|---:|
| QD-DETR | 7,123,237 | 0 | 0.000000% |
| Evidence Head | 115,201 | 115,201 | 1.617256% |
| Ranker | 123,905 | 123,905 | 1.739448% |
| Total EGCG additional | — | 239,106 | 3.356704% |

## Inference runtime

| System | Queries | Batch size | Runs | Mean latency (s) | Std. latency (s) | Mean latency (ms/query) | Peak GPU memory (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|
| QD-DETR | 1,347 | 64 | 3 | 7.1116 | 0.1139 | 5.2796 | 749,876,224 |
| Full EGCG | 1,347 | 64 | 3 | 11.1116 | 0.2252 | 8.2491 | 750,203,904 |

## Archived training cost

| Component | Training time (s) | Provenance |
|---|---:|---|
| Evidence Head | 50.4959 | R27 clean-reproduction runtime record retained in R32 metrics database |
| Ranker | 21.9285 | R27 clean-reproduction runtime record retained in R32 metrics database |

The R31 test evaluation used frozen checkpoints and did not retrain on test data. Runtime is hardware-, batch-, and implementation-dependent; the archived benchmark used identical settings for QD-DETR and Full EGCG.


Table 4 reports complexity, runtime, memory, and archived training cost. QD-DETR has 7,123,237 parameters. EGCG adds 115,201 Evidence Head parameters and 123,905 Ranker parameters, for 239,106 additional parameters, corresponding to a 3.356704% increase over QD-DETR.

Under the same archived hardware and runtime conditions, both systems process 1,347 queries with batch size 64 over three runs. QD-DETR has mean latency 7.1116 s (standard deviation 0.1139 s; 5.2796 ms/query) and peak GPU memory 749,876,224 bytes. Full EGCG has mean latency 11.1116 s (standard deviation 0.2252 s; 8.2491 ms/query) and peak GPU memory 750,203,904 bytes. Thus, the method has measurable inference and memory overhead; the draft does not describe it as free or negligible.

The archived training records report 50.4959 s for Evidence Head training and 21.9285 s for Ranker training. These are the R27 clean-reproduction runtime records retained in the R32 metrics database; the R31 final test evaluation reused frozen checkpoints and did not retrain on the test split.


## 5.6 Cross-model Analysis
# Table 5. Cross-model analysis


| Model | System | R1@0.5 | R1@0.7 | mAP |
|---|---|---:|---:|---:|
| QD-DETR | Original QD-DETR | 38.75% | 18.93% | 16.26% |
| QD-DETR | Full EGCG | 47.51% | 26.80% | 24.43% |
| UVCOM | B0 original | 31.48% | 20.12% | 15.90% |
| UVCOM | B3 full EGCG | 31.92% | 19.97% | 16.33% |

Interpretation boundary: UVCOM is one additional architecture. Its final R1@0.7 result is mixed, so this table does not support universal architecture-independent improvement.


Table 5 summarizes the archived UVCOM comparison as a boundary analysis. For UVCOM, the original B0 system reports R1@0.5 of 31.48%, R1@0.7 of 20.12%, and mAP of 15.90%. The UVCOM full-EGCG result reports 31.92%, 19.97%, and 16.33%, respectively. Candidate-level diagnostics may improve in this comparison, but the final R1@0.7 value does not show the QD-DETR-sized gain.

The cross-model result limits the scope of the method claim. The archive contains one additional architecture, and the UVCOM outcome is mixed. It is therefore reported as a generalization boundary rather than as evidence of universal improvement across AMR architectures.

# 6 Discussion

## 6.1 What EGCG reveals about AMR failure

The results support an evidence-to-candidate-to-ranking view of short-moment AMR failure. The diagnostic chain first separated temporal evidence from the proposal set: query-conditioned temporal evidence was often available in the representation, while the original QD-DETR proposals did not consistently cover the relevant region. It then separated candidate availability from candidate selection: adding evidence-derived proposals increased oracle candidate quality, but inserting them into the original ranking pipeline did not automatically improve final retrieval. This sequence is consistent with a multi-stage bottleneck rather than with a single score or encoder failure. The evidence-candidate studies and the R16 oracle analysis provide the supporting diagnostic evidence.

The final test ablation makes this separation explicit. The baseline A uses original QD-DETR proposals and ranking. The no-evidence ranker B uses the same original proposal source but replaces the original selection rule with a trainable ranker without evidence features; its R1@0.7 is 20.64%, compared with 18.93% for A. This is a ranking-only comparison. System C adds Evidence Head proposals while retaining the no-evidence ranking input; CandidateRecall@10 rises from 64.37% to 76.69%, CandidateRecall@100 reaches 83.96%, and Oracle@10 R1@0.7 rises from 37.56% to 54.42%. The remaining Top1--Oracle gap is 33.33 percentage points, showing that candidate recovery alone does not ensure correct selection.

Full EGCG combines the expanded candidate source with evidence-aware ranking. It reaches 26.80% R1@0.7, with CandidateRecall@10 of 79.29%, Oracle@10 R1@0.7 of 57.31%, and a Top1--Oracle gap of 30.51 percentage points. Thus, the ablation supports the narrower claim that candidate generation and ranking are distinct, coupled failure points in this pipeline: evidence proposals improve the available candidate set, and evidence-aware ranking helps select from that set. The result should not be read as proof that the evidence score is a complete semantic explanation of every prediction. In the recovered implementation, evidence supervision is an operational proposal-level signal derived from annotated temporal overlap.

## 6.2 Why short moments benefit most

Short moments impose a stricter temporal precision requirement: a small absolute boundary error occupies a larger fraction of the target interval and can reduce IoU sharply. In the final test split, baseline R1@0.7 is 1.11% for 0--2s and 6.91% for 2--5s, compared with 18.93% overall. These values identify the short-duration regime as the failure regime under study; they do not establish a duration law outside this benchmark.

The short-duration ablation shows that candidate recovery is especially consequential in this regime. For 0--2s, CandidateRecall@10 changes from 16.67% for A to 54.44% for C and 67.78% for D, while R1@0.7 changes from 1.11% to 5.56% and 21.11%, respectively. For 2--5s, CandidateRecall@10 changes from 43.88% to 73.14% and 76.33%, while R1@0.7 changes from 6.91% to 11.97% and 21.01%. The corresponding CandidateRecall@100 values also increase, from 16.67% to 72.22% for 0--2s and from 43.88% to 83.24% for 2--5s when comparing A with the evidence-candidate system C.

The difference between C and D further indicates why evidence-aware ranking is needed after candidate recovery. For 0--2s, full EGCG improves R1@0.7 from 5.56% to 21.11% and reduces the Top1--Oracle gap from 31.11 to 25.56 percentage points. For 2--5s, it improves R1@0.7 from 11.97% to 21.01% and reduces the gap from 36.70 to 31.12 percentage points. The gap remains substantial, so EGCG does not eliminate short-moment localization difficulty. The defensible interpretation is that the method improves both access to relevant regions and selection among recovered candidates, with the largest absolute benefit in the regime where the baseline candidate set is weakest.

## 6.3 Cross-model implications

The cross-model result places an important boundary on the interpretation. On QD-DETR, the frozen final test comparison changes R1@0.7 from 18.93% to 26.80%. On UVCOM, the archived comparison changes R1@0.7 from 20.12% for the original system to 19.97% with full EGCG, while mAP changes from 15.90% to 16.33%. Candidate-level improvements on UVCOM therefore do not translate into the same final retrieval gain. The result is mixed rather than a universal improvement claim.

This difference is consistent with architecture-dependent evidence utilization. The usefulness of an evidence score depends on how temporal states, proposal confidence, candidate duration, and final ranking are calibrated by the underlying model. The available archive does not isolate which of these interfaces explains the UVCOM result, so these are hypotheses for future analysis rather than conclusions of this study. What the current evidence supports is a conditional statement: an explicit evidence pathway can be effective when its proposal and ranking interfaces are compatible with the base AMR system.

The generalization evidence is consequently limited to the evaluated QD-DETR and UVCOM settings. The cross-model result is valuable as a boundary condition: it argues against presenting EGCG as architecture-independent, and it motivates reporting the candidate and ranking diagnostics alongside final retrieval metrics in future evaluations.

---

# 7 Limitations

The following limitations define the scope of the claims and should remain explicit in the manuscript.

## Dataset limitation

**Limitation.** The diagnosis and final ablation use the CASTELLA task annotations and their temporal-window definitions. Evidence supervision labels a one-second token as positive when it overlaps any annotated ground-truth window; this follows the available annotation structure rather than an independent acoustic event annotation.

**Impact.** The observed short-moment behavior may reflect the event durations, recording conditions, annotation precision, or query distribution of CASTELLA. The results do not establish that the same evidence pattern holds for every audio scene or annotation policy.

**Possible future direction.** Repeat the evidence and candidate diagnostics with independently audited temporal annotations, and test whether conclusions are stable under alternative temporal-label construction rules.

## Single benchmark limitation

**Limitation.** The paper-level test evaluation is on one AMR benchmark. The archive contains extensive within-CASTELLA diagnostics, but these are not a substitute for a second dataset.

**Impact.** Dataset transfer and cross-benchmark robustness remain unverified. In particular, the magnitude of the short-moment gain should not be interpreted as a benchmark-independent constant.

**Possible future direction.** Reproduce the same clean ablation and duration-stratified diagnostic on additional AMR benchmarks with compatible annotations and evaluation protocols.

## Generalization limitation

**Limitation.** Cross-model validation includes QD-DETR and UVCOM, and the result is mixed: EGCG improves the QD-DETR test result but does not improve UVCOM R1@0.7 in the archived comparison.

**Impact.** The study supports architecture-conditional evidence utilization, not universal generalization. The current archive does not identify which UVCOM interface or calibration property limits transfer.

**Possible future direction.** Evaluate more architectures under matched feature, candidate-pool, checkpoint, and ranking protocols, then analyze which temporal-state and proposal interfaces predict successful evidence use.

## Efficiency limitation

**Limitation.** EGCG adds temporal evidence scoring, proposal generation and fusion, and a ranker over the fused candidates. Under the recorded benchmark, mean latency increases from 7.1116 s for QD-DETR to 11.1116 s for full EGCG, while peak GPU memory changes from 749,876,224 to 750,203,904 bytes. The additional parameter count is 239,106, or 3.356704% relative to QD-DETR, but parameter count alone does not determine latency.

**Impact.** The method has a measurable inference cost and may be less attractive in latency-sensitive deployment settings. The reported timing is tied to the archived hardware, batch setting, query count, and implementation, so it is not a universal systems benchmark.

**Possible future direction.** Profile the evidence and fused-candidate stages separately, and investigate implementation-level acceleration or candidate-budget controls while preserving the clean ablation semantics.

## Evidence supervision limitation

**Limitation.** The Evidence Head is trained with a proposal-level target: a valid one-second token is positive if it has non-empty overlap with any GT window, with masked BCE and positive weight 5.0. This supervision identifies annotated temporal overlap; it does not directly encode semantic correctness, acoustic salience, or causal evidence.

**Impact.** The evidence score may inherit annotation boundary noise and can be useful for candidate recall without being a faithful explanation of the model's semantic reasoning. Therefore, evidence activation should not be presented as human-interpretable proof that a sound event is present.

**Possible future direction.** Compare overlap supervision with weak or independently verified acoustic evidence labels, and assess calibration and semantic validity separately from proposal recall.

# 8 Conclusion

This first draft studied why the reproduced AMR baseline fails on short target moments. The archived diagnosis separates three stages that are otherwise conflated by aggregate R1: temporal evidence, candidate generation, and candidate ranking. For the tested QD-DETR setting, the evidence-candidate and clean ablation results support the interpretation that short-moment failure involves both missing or inadequate candidates and a ranking mismatch after candidates are recovered.

The frozen EGCG v1.0 implementation operationalizes this diagnosis with a learned temporal Evidence Head, evidence-guided candidate generation, candidate fusion, and evidence-aware ranking while keeping the baseline encoder and backbone fixed. On the official CASTELLA test split, it improves R1@0.7 from 18.93% to 26.80%, with a larger observed improvement in the 0–2 s regime. The clean ablation shows that candidate generation and ranking contribute different parts of this result.

The evidence is bounded. The method incurs measurable latency overhead, its proposal-level supervision is not a semantic explanation, and the UVCOM comparison is mixed. The appropriate conclusion is therefore benchmark- and architecture-scoped: explicit temporal evidence can be useful when its proposal and ranking interfaces are compatible with the base AMR system, but broader generalization remains open.

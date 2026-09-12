# Diagnosing Short-Moment Failures in Audio Moment Retrieval: Evidence-Guided Candidate and Ranking Analysis

## Abstract

Audio Moment Retrieval (AMR) requires a system to locate the temporal interval in an audio recording that corresponds to a natural-language query. Aggregate retrieval scores, however, do not reveal whether short-moment failures reflect the representation, the availability of suitable temporal candidates, or the selection of candidates that are already present.

We investigate this failure structure through a duration-stratified diagnostic study on the reproduced CASTELLA QD-DETR setting. The analysis separates temporal evidence, candidate generation, and candidate ranking using candidate recall, oracle candidate quality, and the Top1–Oracle gap. The archived diagnosis indicates that query-conditioned temporal scores can support candidate-relevant regions that are absent from the original proposal pool, while adding candidates without a compatible ranking mechanism can increase candidate recall without improving final retrieval.

Guided by this diagnosis, we evaluate Evidence-Guided Candidate Generation and Ranking (EGCG) as a lightweight diagnosis-driven intervention. EGCG adds a learned temporal evidence pathway, evidence-derived candidate proposals, candidate fusion, and evidence-aware ranking while keeping the QD-DETR encoder, backbone, and localization components fixed. The method is evaluated under clean candidate-pool semantics rather than presented as a new backbone architecture.

On the official CASTELLA test split, full EGCG improves QD-DETR R1@0.7 from 18.93% to 26.80%. In the 0–2 s bin, the result changes from 1.11% to 21.11% (N=90), and in the 2–5 s bin from 6.91% to 21.01%. The archived UVCOM comparison is mixed: R1@0.7 changes from 20.12% to 19.97%, while mAP changes from 15.90% to 16.33%. These results support a benchmark- and architecture-conditional interpretation: explicit temporal evidence can improve the tested QD-DETR pipeline, but the evidence score is a proposal-level GT-overlap signal rather than semantic proof, and broader generalization remains open.

Audio Moment Retrieval (AMR) asks a system to locate, within a recording, the temporal interval that matches a natural-language query. The task therefore combines audio–text alignment with temporal localization. A system must identify not only whether a queried acoustic event is present, but also which interval best matches the query and how precisely its boundaries are estimated. This makes the candidate-generation and candidate-selection interfaces central to evaluation, particularly when the target moment is short.

Aggregate retrieval scores are useful but insufficient for diagnosing this interface. A Top-1 miss can arise because the proposal pool contains no sufficiently overlapping candidate, because a relevant candidate is ranked below competing intervals, or because the available boundaries are too coarse for the target duration. We therefore treat AMR evaluation as a pipeline problem and examine duration-stratified retrieval, candidate recall, oracle candidate quality, and the gap between the best available candidate and the selected prediction.

The reproduced CASTELLA evaluation identifies a severe short-moment regime for QD-DETR. QD-DETR reaches 18.93% R1@0.7 overall, but only 1.11% for 0–2 s moments and 6.91% for 2–5 s moments. These values locate the failure regime; they do not by themselves determine its mechanism or imply a duration-independent law.

We consequently perform a staged diagnosis. A representation control improves overall AMR performance but does not remove short-scale degradation. Geometry and candidate decomposition indicate that candidate availability is a major component of the audited short-moment failure. Query-conditioned evidence audits then show that evidence-derived temporal regions can recover candidate-relevant intervals missing from the original proposal pool. At the same time, naive candidate augmentation can increase candidate recall while reducing final retrieval when the original ranking mechanism is retained. Together, these results motivate a diagnosis of two coupled interfaces: evidence-to-candidate conversion and candidate selection.

We evaluate Evidence-Guided Candidate Generation and Ranking (EGCG) as a targeted intervention for those interfaces. EGCG is not introduced here as an architecture-independent solution or as a claim that its small MLP components are individually novel. It provides a controlled way to test whether a learned temporal evidence signal can add useful candidates and whether evidence-aware ranking can select from the expanded pool, while the baseline encoder, QD-DETR backbone, and localization components remain fixed.

The contributions are deliberately scoped:

1. **Failure characterization.** We provide a duration-stratified diagnosis of short-moment AMR failure in the reproduced CASTELLA QD-DETR setting, separating representation, candidate availability, and candidate selection evidence.
2. **Evidence-to-candidate analysis.** We show that query-conditioned temporal scores can improve candidate-level availability in the audited pipeline, while distinguishing candidate recall from final Top-1 retrieval and from semantic interpretation.
3. **Diagnosis-driven intervention.** We evaluate EGCG under clean candidate-pool semantics, separating the original proposal pool, a no-evidence ranking comparison, evidence candidate generation, and the combined evidence-aware pipeline.
4. **Bounded validation.** We report the official test result, short-duration behavior, efficiency overhead, and the mixed UVCOM comparison. We make no claim of cross-dataset generalization, semantic evidence validation, or state-of-the-art performance.

# 2 Related Work

This section is organized around the interfaces relevant to the diagnosis. Verified references are intentionally not inserted in this revision; each required citation is marked explicitly.

## 2.1 Audio Moment Retrieval systems

AMR systems align natural-language queries with audio recordings and predict temporal moments. The final Related Work should cite representative AMR systems and distinguish their proposal generation, temporal localization, and ranking mechanisms. **[CITATION REQUIRED: representative AMR systems and the CASTELLA task.]**

Existing AMR evaluation is commonly summarized with aggregate retrieval metrics. For this paper, the relevant unresolved question is whether a short-moment miss reflects missing candidates, boundary imprecision, or selection of an available candidate. **[CITATION REQUIRED: prior AMR evaluations that report proposal quality, duration-stratified behavior, or reranking.]** Our work contributes a diagnostic decomposition of these interfaces in the reproduced CASTELLA QD-DETR setting; it does not claim to replace AMR systems or define a single evaluation standard.

## 2.2 Audio-language representation learning

Audio-language representation learning studies how audio and natural-language representations can be aligned, including contrastive learning, multimodal embeddings, and temporal audio representations. **[CITATION REQUIRED: representative audio-language representation and audio-text alignment methods.]**

Representation quality can improve overall retrieval without ensuring that query-relevant temporal information is converted into precise candidate intervals. The archived representation control motivates this distinction: representation improvement helps overall AMR but does not remove short-scale degradation in the tested QD-DETR setting. Our intervention therefore keeps the baseline representation and tests the downstream evidence-to-candidate and candidate-selection interfaces. **[CITATION REQUIRED: prior work connecting audio-language representations to temporal localization or proposal generation.]**

## 2.3 Diagnostic evaluation and failure analysis

Diagnostic evaluation and failure analysis examine subgroup behavior, intermediate candidate quality, oracle performance, and the gap between available candidates and selected outputs. **[CITATION REQUIRED: prior retrieval/temporal-grounding/audio-understanding failure analyses using intermediate or subgroup diagnostics.]**

The limitation relevant to this paper is that a failure analysis may identify an error category without testing whether an intermediate signal can be converted into a useful candidate intervention, while a method paper may report final retrieval without showing whether the candidate pool or ranking mechanism changed. Our work links the two: it first characterizes short-moment failure and then evaluates a diagnosis-driven temporal relevance signal, evidence-derived candidates, and evidence-aware ranking under clean candidate-pool semantics.

## 2.4 Positioning of this work

The paper's gap is deliberately narrow. Existing AMR and audio-language work establishes the task and representation setting; diagnostic work motivates separating intermediate candidate quality from final selection. We then evaluate a targeted evidence-to-candidate-to-ranking intervention for the audited short-moment regime. **[CITATION REQUIRED: closest prior method combining query-conditioned temporal scoring, proposal generation, and candidate reranking.]**

The paper does not claim that these building blocks are individually new, that EGCG establishes benchmark-wide superiority, or that the intervention generalizes across all settings. The contribution is the diagnosis-driven characterization, the clean candidate-pool semantics, and the bounded evaluation of the targeted intervention.

# 3 Understanding Short-Moment AMR Failures

## 3.1 Short moments remain challenging

We first examine the reproduced QD-DETR baseline by target duration. QD-DETR obtains 18.93% R1@0.7 overall, but the rate falls to 6.91% for 2–5 s moments and 1.11% for 0–2 s moments. The shortest-bin audit contains N=90 queries. This establishes a benchmark-scoped failure regime and motivates a more detailed analysis; it does not establish that duration alone determines performance across AMR datasets or architectures.

## 3.2 Temporal evidence as an operational proposal signal

The evidence analysis asks whether query-conditioned temporal scores can identify regions relevant to the annotated moment even when the original candidate pool does not. In this paper, “temporal evidence” has an operational meaning: the Evidence Head produces scores over valid one-second temporal tokens, and the training target is positive when a token overlaps any annotated GT window. The score is therefore evaluated as a proposal-level signal. It is not treated as an independently validated semantic explanation of the queried sound.

The archived evidence audit and oracle studies indicate that evidence-derived regions can recover candidate-relevant intervals missing from the original QD-DETR pool in a non-trivial subset of audited cases. This finding supports an evidence-to-candidate hypothesis in the tested pipeline, while leaving open whether the same signal is available or useful for other architectures.

## 3.3 Candidate generation bottleneck

The geometry and candidate decomposition indicate that candidate availability is a major component of short-moment failure in the audited QD-DETR regime. Positive-overlap predictions can cover part of the target while extending beyond it, and the short-moment decomposition identifies missing relevant candidates as a dominant failure component. This pattern is consistent with insufficiently targeted or coarse candidate coverage, but it does not exclude boundary-quality and ranking effects.

Evidence proposal oracle and prototype studies provide a targeted diagnostic intervention. Adding query-conditioned evidence proposals raises candidate-level recall and oracle candidate quality in the archived analyses, including the short-duration regime. These results reveal what the expanded candidate pool makes available; they do not by themselves establish that the existing ranker will select the recovered intervals or that the intervals are semantically correct.

## 3.4 Ranking mismatch after candidate recovery

When evidence-generated candidates are inserted into the original QD-DETR ranking pipeline, candidate recall increases while final R1 can decrease. This result indicates a mismatch between the original selection scores and the expanded candidate pool in the tested pipeline. This does not establish that ranking is the sole remaining source of error. Boundary precision, score calibration, and downstream localization can also affect the Top-1 result.

The oracle analysis makes the selection problem visible: the expanded candidate pool can contain intervals with substantially higher overlap than the selected prediction, especially for short moments. We therefore evaluate evidence-aware ranking as the selection-side part of the intervention. Its purpose is to address the observed pool–selection mismatch, not to claim that every residual error is explained by ranking.

## 3.5 Diagnosis summary

The diagnosis yields a bounded two-interface interpretation for the tested QD-DETR setting. First, short moments are often poorly served by the original candidate pool even when query-conditioned temporal scores can support relevant regions. Second, candidate expansion without compatible selection does not reliably improve final retrieval. EGCG is therefore evaluated as a diagnosis-driven intervention that links evidence-guided candidate generation with evidence-aware ranking. The subsequent results test this link on the official CASTELLA evaluation without changing the baseline encoder or backbone.

# 4 Method

### Positioning note

In this paper, temporal evidence is used as a temporal relevance signal for candidate generation and selection. It is not a semantic event detector or a human-validated semantic explanation. The technical definitions below are retained from the frozen EGCG v1.0 implementation; the positioning change concerns the role and interpretation of those definitions, not their equations or architecture.

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

The 0–2 s result should be read with its archived N=90 denominator. The table is evidence that the diagnosed short regime improves under the frozen EGCG pipeline on this test split; it is not evidence for a duration-independent short-event law or for architecture-independent generalization.


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

Interpretation boundary: UVCOM is one additional architecture. Its final R1@0.7 result is mixed, so this table does not support architecture-independent improvement.


Table 5 summarizes the archived UVCOM comparison as a boundary analysis. For UVCOM, the original B0 system reports R1@0.5 of 31.48%, R1@0.7 of 20.12%, and mAP of 15.90%. The UVCOM full-EGCG result reports 31.92%, 19.97%, and 16.33%, respectively. Candidate-level diagnostics may improve in this comparison, but the final R1@0.7 value does not show the QD-DETR-sized gain.

The cross-model result limits the scope of the method claim. The archive contains one additional architecture, and the UVCOM outcome is mixed. It is therefore reported as a generalization boundary rather than as evidence of architecture-independent improvement across AMR architectures.

# 6 Discussion

## 6.1 What the intervention reveals

The results support an evidence-to-candidate-to-ranking interpretation of the audited short-moment failure, rather than a claim about one architecture-independent AMR mechanism. The clean ablation separates two pipeline changes. The no-evidence ranker B keeps the original QD-DETR candidate pool and raises R1@0.7 from 18.93% to 20.64%, providing a ranking comparison without evidence-derived candidates. System C adds Evidence Head proposals while retaining the no-evidence ranking mechanism; CandidateRecall@10@0.5 rises from 64.37% to 76.69%, and Oracle@10@0.7 rises from 37.56% to 54.42%. These values indicate improved candidate availability under the stated evaluation definitions.

Full EGCG D combines the expanded candidate pool with evidence-aware ranking and reaches 26.80% R1@0.7, with CandidateRecall@10@0.5 of 79.29%, Oracle@10@0.7 of 57.31%, and a Top1–Oracle gap of 30.51 percentage points. The comparison indicates that ranking compatibility matters after candidate recovery. It does not show that the components have independent additive effects, nor that the Evidence Head provides semantic explanations. The evidence target remains a GT-overlap-supervised proposal signal.

## 6.2 Why the observed gain is larger in short bins

Short intervals are more sensitive to absolute boundary error, so improving candidate coverage can have a larger effect when the original candidate pool is weak. In the 0–2 s bin, R1@0.7 changes from 1.11% for QD-DETR to 5.56% with evidence candidate generation and 21.11% with full EGCG. In the 2–5 s bin, the corresponding values are 6.91%, 11.97%, and 21.01%. Candidate-level and final metrics should be read separately: the former describe availability under explicit IoU thresholds, while the latter describe the selected Top-1 interval.

The full system still has substantial residual Top1–Oracle gaps: 25.56 percentage points for 0–2 s and 31.12 percentage points for 2–5 s. The appropriate interpretation is therefore partial improvement of the observed short-duration regime, not resolution of short-moment localization. The effect is strongest where the baseline candidate pool is weakest in this test split, but the archive does not establish a duration-independent short-event rule.

## 6.3 Architecture-dependent evidence utilization

The cross-model result limits the scope of the intervention. On QD-DETR, full EGCG changes R1@0.7 from 18.93% to 26.80%. On UVCOM, the archived comparison changes R1@0.7 from 20.12% to 19.97%, while mAP changes from 15.90% to 16.33%. Thus, the evidence pathway is effective in the tested QD-DETR/CASTELLA setting but does not produce the same final R1 improvement on the additional architecture.

This pattern indicates architecture-dependent evidence utilization. Differences in temporal state interfaces, proposal score calibration, candidate duration distributions, and ranking inputs are plausible explanations, but the current archive does not isolate which factor determines the UVCOM outcome. The paper should therefore claim conditional compatibility with a base AMR pipeline, not architecture-independent generalization.

## 6.4 Efficiency tradeoff

The intervention has a measurable computational cost. Under the archived runtime setting, mean latency increases from 7.1116 s for QD-DETR to 11.1116 s for full EGCG, and peak GPU memory changes from 749,876,224 to 750,203,904 bytes. EGCG adds 239,106 parameters, corresponding to a 3.356704% increase over QD-DETR, but parameter count does not by itself determine latency because evidence scoring, proposal generation and fusion, and ranking over the expanded pool add runtime work.

This tradeoff is part of the result rather than an implementation detail to omit. The reported timing is tied to the archived hardware and measurement setting, so it should be interpreted as a reproducibility record rather than a deployment benchmark. The intervention is best viewed as a targeted diagnostic and retrieval improvement whose cost must be considered in future deployment-oriented work.

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

**Impact.** The study supports architecture-conditional evidence utilization, not cross-dataset generalization. The current archive does not identify which UVCOM interface or calibration property limits transfer.

**Possible future direction.** Evaluate more architectures under matched feature, candidate-pool, checkpoint, and ranking protocols, then analyze which temporal-state and proposal interfaces predict successful evidence use.

## Efficiency limitation

**Limitation.** EGCG adds temporal evidence scoring, proposal generation and fusion, and a ranker over the fused candidates. Under the recorded benchmark, mean latency increases from 7.1116 s for QD-DETR to 11.1116 s for full EGCG, while peak GPU memory changes from 749,876,224 to 750,203,904 bytes. The additional parameter count is 239,106, or 3.356704% relative to QD-DETR, but parameter count alone does not determine latency.

**Impact.** The method has a measurable inference cost and may be less attractive in latency-sensitive deployment settings. The reported timing is tied to the archived hardware, batch setting, query count, and implementation, so it is not a deployment-wide systems benchmark.

**Possible future direction.** Profile the evidence and fused-candidate stages separately, and investigate implementation-level acceleration or candidate-budget controls while preserving the clean ablation semantics.

## Evidence supervision limitation

**Limitation.** The Evidence Head is trained with a proposal-level target: a valid one-second token is positive if it has non-empty overlap with any GT window, with masked BCE and positive weight 5.0. This supervision identifies annotated temporal overlap; it does not directly encode semantic correctness, acoustic salience, or causal evidence.

**Impact.** The evidence score may inherit annotation boundary noise and can be useful for candidate recall without being a faithful explanation of the model's semantic reasoning. Therefore, evidence activation should not be presented as human-interpretable proof that a sound event is present.

**Possible future direction.** Compare overlap supervision with weak or independently verified acoustic evidence labels, and assess calibration and semantic validity separately from proposal recall.

# 8 Conclusion

This paper examines short-moment AMR failure as a pipeline diagnosis problem. In the reproduced CASTELLA QD-DETR setting, duration-stratified evaluation and intermediate diagnostics reveal a distinction between temporal relevance signals, candidate availability, and candidate selection. Query-conditioned signals can support candidate-relevant regions that are absent from the original proposal pool, while candidate expansion without compatible ranking does not reliably improve final retrieval.

We evaluate EGCG as a diagnosis-driven intervention that links temporal relevance scoring, evidence-derived candidate generation, candidate fusion, and evidence-aware ranking while keeping the baseline encoder and backbone fixed. On the official test split, full EGCG changes QD-DETR R1@0.7 from 18.93% to 26.80%, and the 0–2 s result from 1.11% to 21.11% (N=90). The intervention partially improves the observed short-duration regime; it does not eliminate the residual Top1–Oracle gap.

The scope of the result is explicit. UVCOM shows a mixed outcome, with R1@0.7 changing from 20.12% to 19.97%. The temporal relevance signal is a GT-overlap-supervised proposal-level quantity rather than semantic proof, and the method adds measurable latency. The contribution is therefore a failure characterization and targeted intervention for the tested pipeline, with architecture-dependent generalization left open.

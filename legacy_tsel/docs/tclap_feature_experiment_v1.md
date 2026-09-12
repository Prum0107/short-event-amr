# T-CLAP Feature Experiment V1

## Purpose

We will evaluate audio representations as separate research lines:

```text
Line A: MS-CLAP evidence
Line B: T-CLAP evidence
Line C: T-MS-CLAP fusion, only if Line B is useful or complementary
```

The goal is not to tune another decoder first. The goal is to answer whether a temporally stronger audio-language representation reduces semantic-evidence errors and improves boundary decoding reliability.

## Current Status

T-CLAP is the right kind of candidate because it targets the weakness we observed in MS-CLAP: ordinary CLAP can align audio and text semantically, but it is weak at temporal order and event transition structure.

Practical blocker:

```text
As of this setup pass, no verified public T-CLAP checkpoint has been found in the local/Hugging Face search.
```

Evidence for the blocker:

```text
Hugging Face paper page: no linked model/dataset/space.
Papers With Code page: no code implementation listed.
```

References:

```text
https://arxiv.org/abs/2404.17806
https://huggingface.co/papers/2404.17806
https://paperswithcode.com/paper/t-clap-temporal-enhanced-contrastive-language
```

So we should separate two things clearly:

```text
Real T-CLAP line:
requires an official or reproducible T-CLAP checkpoint.

T-CLAP proxy line:
uses an available temporal audio-language model or windowed CLAP strategy.
This can test the pipeline, but it must not be reported as T-CLAP.
```

## Required Feature Format

The current evidence training pipeline expects precomputed `.npz` files.

Audio features:

```text
features/castella/tclap/{vid}.npz
key: features
shape: [T, D]
time step: preferably 1 second, aligned with clip_length=1
```

Text features:

```text
features/castella/tclap_text/qid{qid}.npz
key: last_hidden_state
shape: [L, D]
```

If T-CLAP only exposes pooled text embeddings, we need either:

```text
1. adapt the evidence model to accept pooled query embeddings, or
2. save the pooled embedding as a length-1 sequence [1, D].
```

Option 2 is simpler and compatible with the current `masked_mean` query encoder.

## Code Support Added

The evidence baseline can now switch feature sets without editing `config.yml`.

New command-line overrides:

```text
--a_feat_dir
--t_feat_dir
--a_feat_dim
--t_feat_dim
--ctx_mode
--clip_length
--max_a_l
--feature_name
```

This makes the MS-CLAP and future T-CLAP experiments comparable under the same model/training code.

## MS-CLAP Reference Command

Validate feature files first:

```powershell
E:\anaconda\python.exe src\validate_feature_set.py `
  --data_path data\castella_val_release.jsonl `
  --a_feat_dir features\castella\clap `
  --t_feat_dir features\castella\clap_text `
  --a_feat_dim 768 `
  --t_feat_dim 768
```

```powershell
E:\anaconda\python.exe src\train_evidence_baseline.py `
  --feature_name ms_clap `
  --a_feat_dir features\castella\clap `
  --t_feat_dir features\castella\clap_text `
  --a_feat_dim 768 `
  --t_feat_dim 768 `
  --train_data_path data\castella_train_release.jsonl `
  --val_data_path data\castella_val_release.jsonl `
  --epochs 3 `
  --batch_size 16 `
  --eval_batch_size 16 `
  --results_dir results\evidence_baseline_ms_clap_v1 `
  --save_path results\evidence_baseline_ms_clap_v1\best.pt
```

## Future T-CLAP Command

After real T-CLAP features exist:

```powershell
E:\anaconda\python.exe src\train_evidence_baseline.py `
  --feature_name t_clap `
  --a_feat_dir features\castella\tclap `
  --t_feat_dir features\castella\tclap_text `
  --a_feat_dim <D> `
  --t_feat_dim <D> `
  --train_data_path data\castella_train_release.jsonl `
  --val_data_path data\castella_val_release.jsonl `
  --epochs 3 `
  --batch_size 16 `
  --eval_batch_size 16 `
  --results_dir results\evidence_baseline_tclap_v1 `
  --save_path results\evidence_baseline_tclap_v1\best.pt
```

## Evaluation Protocol

For each representation line, run the same downstream stages:

```text
1. evidence baseline training
2. full train/val evidence dump
3. shape_v2 learned decoder
4. source gate / two-mode selector only after evidence quality is understood
5. feature ceiling analysis
```

Primary comparison should not be only `R1@0.7`.

We should compare:

```text
semantic_miss
representation_ceiling_errors
decoder_candidate_errors
top-k oracle coverage
evidence_gap
boundary_error vs semantic_error migration
```

## Decision Rule

Use T-CLAP further only if it shows at least one of these:

```text
1. lower semantic_miss than MS-CLAP
2. lower representation_ceiling_errors
3. higher top-k oracle coverage
4. complementary successes where MS-CLAP fails
```

If T-CLAP improves strict top1 but increases semantic misses, it should be treated like V5.4: useful signal, not the new default.

If T-CLAP is complementary but not strictly better, move to score-level fusion first. Do not build a full T-MS-CLAP model until simple fusion gives evidence that the two streams help each other.

# T-CLAP-Inspired Temporal Contrastive Adapter V1

## Purpose

T-CLAP has no verified public code or checkpoint, so we will not claim to use T-CLAP directly.

Instead, this line tests the core idea behind T-CLAP:

```text
audio-text representations should distinguish the correct temporal event region
from temporally shifted, expanded, contracted, or unrelated regions in the same audio.
```

This is a controlled representation experiment built on top of the existing MS-CLAP features.

## Method

The adapter takes existing MS-CLAP features:

```text
audio: features/castella/clap/{vid}.npz
text:  features/castella/clap_text/qid{qid}.npz
```

It trains:

```text
audio adapter: MS-CLAP frame features -> temporal Conv adapter -> adapted audio features
text adapter: pooled MS-CLAP text tokens -> adapted query embedding
```

For each query/audio pair, the positive moment is the ground-truth temporal window. Negatives are generated from the same audio:

```text
shifted windows
expanded windows
contracted windows
random low-overlap windows
```

The loss combines:

```text
local temporal contrastive loss:
query should prefer the GT window over same-audio negative windows

batch contrastive loss:
query should prefer its own positive moment over other batch positives
```

## Why This Is The Right Next Step

This directly tests our current hypothesis:

```text
The bottleneck is not only boundary decoding.
The audio-text representation must encode temporally grounded semantics.
```

If this works, we should see fewer semantic misses and better top-k candidate coverage after running the same evidence decoder stack.

If this fails, the conclusion is still useful: simple supervised temporal contrastive adaptation is insufficient, and we need either raw-audio modeling or a stronger pretrained audio-language backbone.

## Script

```text
src/train_temporal_contrastive_adapter.py
```

## Smoke Command

```powershell
E:\anaconda\python.exe src\train_temporal_contrastive_adapter.py `
  --train_data_path data\castella_train_small.jsonl `
  --val_data_path data\castella_val_small.jsonl `
  --output_dim 256 `
  --epochs 1 `
  --batch_size 8 `
  --output_dir results\temporal_contrastive_adapter_smoke `
  --checkpoint_path results\temporal_contrastive_adapter_smoke\best.pt `
  --export_data_paths data\castella_train_small.jsonl data\castella_val_small.jsonl `
  --audio_output_dir features\castella\tclap_inspired_smoke `
  --text_output_dir features\castella\tclap_inspired_text_smoke
```

## Full First-Pass Command

```powershell
E:\anaconda\python.exe src\train_temporal_contrastive_adapter.py `
  --train_data_path data\castella_train_release.jsonl `
  --val_data_path data\castella_val_release.jsonl `
  --output_dim 768 `
  --epochs 5 `
  --batch_size 16 `
  --output_dir results\temporal_contrastive_adapter_v1 `
  --checkpoint_path results\temporal_contrastive_adapter_v1\best.pt `
  --export_data_paths data\castella_train_release.jsonl data\castella_val_release.jsonl data\castella_test_release.jsonl `
  --audio_output_dir features\castella\tclap_inspired `
  --text_output_dir features\castella\tclap_inspired_text
```

## Downstream Evidence Training

After feature export:

```powershell
E:\anaconda\python.exe src\train_evidence_baseline.py `
  --feature_name tclap_inspired_adapter_v1 `
  --a_feat_dir features\castella\tclap_inspired `
  --t_feat_dir features\castella\tclap_inspired_text `
  --a_feat_dim 768 `
  --t_feat_dim 768 `
  --train_data_path data\castella_train_release.jsonl `
  --val_data_path data\castella_val_release.jsonl `
  --epochs 3 `
  --batch_size 16 `
  --eval_batch_size 16 `
  --results_dir results\evidence_baseline_tclap_inspired_v1 `
  --save_path results\evidence_baseline_tclap_inspired_v1\best.pt
```

## Evaluation Question

Do not judge this line only by `R1@0.7`.

Compare against MS-CLAP on:

```text
semantic_miss
representation_ceiling_errors
top-k oracle coverage
evidence_gap
boundary_error versus semantic_error migration
```

## First Results

The full first-pass adapter was trained for 5 epochs with 768-dimensional output features.

Adapter validation signal:

```text
local temporal contrastive accuracy:
epoch 1: 26.14%
epoch 2: 30.11%
epoch 4: 30.40%
epoch 5: 32.10%

random level with 1 positive + 8 negatives: about 11.11%
```

This means the adapter learned a real temporal preference: the query is more likely to select the GT window over shifted, expanded, contracted, or random same-audio negatives.

### Evidence Baseline Comparison

| Evidence source | R1@0.5 | R1@0.7 | R3@0.7 | best IoU top5 | evidence gap |
|---|---:|---:|---:|---:|---:|
| MS-CLAP V1 | 23.01 | 17.33 | 25.85 | 45.78 | 0.2178 |
| T-CLAP-inspired adapter V1 | 28.13 | 21.31 | 32.39 | 49.05 | 0.2800 |

The evidence-level improvement matters because no decoder change was involved here. The representation itself produced a stronger query-conditioned evidence curve.

### Learned Decoder Comparison

Both rows use the same `shape_v2 + top2` learned decoder protocol.

| Evidence source | R1@0.5 | R1@0.7 | R3@0.7 | R5@0.7 | Top1 IoU | Top5 IoU | Semantic miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| MS-CLAP shape_v2 top2 | 34.94 | 25.85 | 38.35 | 43.18 | 0.3316 | 0.4959 | 77 |
| T-CLAP-inspired shape_v2 top2 | 38.07 | 26.70 | 41.48 | 47.73 | 0.3606 | 0.5437 | 64 |

This is the important result:

```text
semantic_miss reduced from 77 to 64
top5 IoU improved from 0.4959 to 0.5437
R5@0.7 improved from 43.18% to 47.73%
```

So the adapter is not only moving top1 scores. It improves candidate coverage and reduces semantic evidence failures.

### Ceiling Diagnosis

For T-CLAP-inspired shape_v2 top2:

```text
semantic_miss: 64
representation_ceiling: 31
decoder_candidate_ceiling: 33
```

Previous MS-CLAP shape_v2 top2:

```text
semantic_miss: 77
representation_ceiling: 38
decoder_candidate_ceiling: 39
```

Both representation and decoder/candidate ceiling errors decreased.

### Migration From MS-CLAP Shape_V2

```text
semantic_miss recovered: 39 / 77
semantic_miss -> good: 5
good regressed: 35
average top1 IoU delta: +0.0290
average top5 IoU delta: +0.0478
average evidence gap delta: +0.0873
```

Interpretation:

```text
The temporal adapter improves semantic evidence and top-k coverage,
but it is not uniformly better. Some originally good MS-CLAP cases regress.
```

This points to the next research step:

```text
Do not replace MS-CLAP blindly.
Compare MS-CLAP and temporal-adapter evidence case by case,
then test score-level fusion or a conservative representation selector.
```

## Reports

```text
results/temporal_contrastive_adapter_v1/summary.json
results/evidence_baseline_tclap_inspired_v1/full_val_eval/metrics.json
results/learned_evidence_decoder_tclap_inspired_shape_v2_top2/report.html
results/feature_ceiling_diagnosis_tclap_inspired_shape_v2_top2/report.html
results/decoder_migration_ms_clap_shape_v2_to_tclap_inspired_shape_v2/report.html
```

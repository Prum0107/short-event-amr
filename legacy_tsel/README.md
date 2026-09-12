# DCASE26 Task 6 Audio Moment Retrieval

This repository contains research code for DCASE 2026 Task 6: retrieving the
time window in a long audio clip that matches a natural-language query.

The current project direction is **Temporal Semantic Evidence Learning (TSEL)**.
Instead of predicting a start/end pair directly, TSEL first learns a
query-conditioned temporal evidence curve `S(t, q)`, then decodes and reranks
candidate windows from that evidence.

## Method Summary

The paper-facing system is organized as:

1. `MS-EB`: MS-CLAP Evidence Baseline.
2. `TSA`: Temporal-Semantic Adapter trained on top of MS-CLAP features.
3. `SBEC`: Semantic-to-Boundary Evidence Curriculum using hard negatives.
4. `TSEL-ECF`: Evidence Candidate Fusion.
5. `TSEL-RAES`: Risk-Aware Evidence Scoring.

The main research question is:

```text
Can query-conditioned temporal evidence explain both semantic matching and
temporal boundary reliability for audio moment retrieval?
```

## Repository Layout

```text
config.yml                         # default CASTELLA/MS-CLAP configuration
requirements.txt                   # Python dependencies
data/                              # tracked JSONL splits
src/                               # training, decoding, evaluation, analysis code
docs/                              # reports, paper assets, submission notes
docs/paper_assets/                 # paper-facing tables, figures, drafts, naming map
```

Large artifacts are intentionally not tracked:

```text
features/                          # downloaded or generated CLAP features
results/                           # checkpoints, predictions, reports
data/raw/                          # optional raw audio
```

## Installation

Use Python 3.10 or 3.11. A CUDA-enabled PyTorch environment is recommended for
training, but CPU is enough for format checks and small smoke tests.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Preparation

This repository tracks the CASTELLA JSONL metadata splits:

```text
data/castella_train_release.jsonl
data/castella_val_release.jsonl
data/castella_test_release.jsonl
```

The MS-CLAP feature files are too large for Git. Place them in:

```text
features/castella/clap/
features/castella/clap_text/
```

Expected feature dimensionality is `768` for both audio and text features. Check
the feature layout before training:

```bash
python src/validate_feature_set.py \
  --data_path data/castella_val_release.jsonl \
  --a_feat_dir features/castella/clap \
  --t_feat_dir features/castella/clap_text \
  --a_feat_dim 768 \
  --t_feat_dim 768
```

For the official DCASE evaluation set, place the metadata and features under:

```text
data/evaluation_release/dcase2026_evaluation.jsonl
features/evaluation/clap/
features/evaluation/clap_text/
```

The official evaluation set has no public ground truth, so it can only be used
to generate submission files.

## Reproducing Core Runs

Train the MS-CLAP evidence baseline:

```bash
python src/train_evidence_baseline.py \
  --config config.yml \
  --train_data_path data/castella_train_release.jsonl \
  --val_data_path data/castella_val_release.jsonl \
  --results_dir results/ms_eb \
  --save_path results/ms_eb/best.pt \
  --batch_size 32 \
  --eval_batch_size 32 \
  --epochs 20 \
  --feature_name ms_clap
```

Export train/validation evidence dumps from the checkpoint:

```bash
python src/evaluate_evidence_baseline.py \
  --config config.yml \
  --data_path data/castella_train_release.jsonl \
  --ckpt_path results/ms_eb/best.pt \
  --output_dir results/ms_eb/train_eval \
  --batch_size 32 \
  --feature_name ms_clap

python src/evaluate_evidence_baseline.py \
  --config config.yml \
  --data_path data/castella_val_release.jsonl \
  --ckpt_path results/ms_eb/best.pt \
  --output_dir results/ms_eb/val_eval \
  --batch_size 32 \
  --feature_name ms_clap
```

Train and export the temporal-semantic adapter features:

```bash
python src/train_temporal_contrastive_adapter.py \
  --train_data_path data/castella_train_release.jsonl \
  --val_data_path data/castella_val_release.jsonl \
  --a_feat_dir features/castella/clap \
  --t_feat_dir features/castella/clap_text \
  --output_dir results/tsa \
  --checkpoint_path results/tsa/best.pt \
  --audio_output_dir features/castella/tclap_inspired \
  --text_output_dir features/castella/tclap_inspired_text \
  --export_data_paths \
    data/castella_train_release.jsonl \
    data/castella_val_release.jsonl \
    data/castella_test_release.jsonl
```

Train a learned evidence decoder from saved evidence dumps:

```bash
python src/train_learned_evidence_decoder.py \
  --train_evidence_path results/ms_eb/train_eval/predictions_evidence_samples.json \
  --val_evidence_path results/ms_eb/val_eval/predictions_evidence_samples.json \
  --output_dir results/learned_decoder \
  --feature_version shape_v2
```

Generate an official-format submission from an evidence checkpoint:

```bash
python src/create_evaluation_submission.py \
  --data_path data/evaluation_release/dcase2026_evaluation.jsonl \
  --ckpt_path results/ms_eb/best.pt \
  --output_path results/evaluation_ms_eb/output.jsonl \
  --evidence_dump_path results/evaluation_ms_eb/evidence_dump.json \
  --a_feat_dir features/evaluation/clap \
  --t_feat_dir features/evaluation/clap_text \
  --feature_name dcase2026_eval_msclap \
  --batch_size 32 \
  --topn 10
```

Apply a learned decoder to an evidence dump:

```bash
python src/apply_learned_evidence_decoder.py \
  --evidence_path results/evaluation_ms_eb/evidence_dump.json \
  --decoder_ckpt results/learned_decoder/best.pt \
  --output_path results/evaluation_ms_eb/output_decoded.jsonl \
  --topn 10
```

Submission rows follow the official JSONL format:

```json
{
  "qid": 1,
  "query": "A natural-language query.",
  "duration": 300,
  "vid": "audio_id",
  "pred_relevant_windows": [[12.0, 18.0], [30.0, 40.0]]
}
```

## Result Provenance

Read all numbers by split:

| Source | Queries | Ground truth | Usage |
|---|---:|---|---|
| CASTELLA validation release | 352 | yes | ablations and model selection |
| CASTELLA frozen release test | 1347 | yes | local generalization estimate |
| DCASE2026 official evaluation set | 177 | no public labels | submission generation only |

Current paper-facing local results:

| System | Split | R1@0.7 | mAP(avg) | Note |
|---|---|---:|---:|---|
| TSEL-ECF | validation, five-seed mean | 32.22 | - | strongest candidate-fusion result |
| TSEL-RAES | validation, five-seed mean | 32.27 | - | interpretable risk-aware scorer |
| TSEL-ECF | frozen release test, five-seed mean | 23.42 | 18.66 | stronger broad fusion |
| TSEL-RAES | frozen release test, five-seed mean | 23.59 | 18.53 | slightly better strict R1, more explainable |

The official evaluation set cannot be scored locally because labels are not
released. The current official-evaluation submission process is documented in
`docs/evaluation_submission_20260603.md`.

## Documentation

Start here for paper and report materials:

- `docs/paper_assets/README.md`: paper-facing asset index.
- `docs/paper_assets/naming_map.md`: mapping from engineering names to paper names.
- `docs/paper_assets/main_tables.md`: current quantitative tables.
- `docs/paper_assets/paper_draft_v2_en.md`: current English paper draft.
- `docs/evaluation_submission_20260603.md`: official evaluation submission notes.

## Notes

- T-CLAP public code/checkpoints were not verified during this project, so the
  temporal branch is described as **T-CLAP-inspired** rather than as an official
  T-CLAP implementation.
- Checkpoints and result files are excluded from Git. To reproduce a reported
  number, first regenerate the required feature and evidence dumps, then run the
  corresponding training or decoding script.

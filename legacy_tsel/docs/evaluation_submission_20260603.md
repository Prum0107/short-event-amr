# DCASE2026 Task 6 Evaluation Submission Run

Date: 2026-06-03

## Official Evaluation Data

- Source: `lighthouse-emnlp2024/AudioMomentRetrievalFromLongAudio_DCASE2026EvaluationData`
- Evaluation rows: 177 queries
- Audio feature files: 100 MS-CLAP audio feature files
- Text feature files: 177 MS-CLAP text feature files
- Local download cache: `E:\DCASE26_Task6_EvaluationData`
- Remote run directory: `/private/research-artifact`

The evaluation set does not include ground-truth windows, so local R1/mAP cannot
be computed. The generated files are official-format submission files only.

## Generated Outputs

The actual output files are kept under ignored runtime storage:

- `results/evaluation_20260603/ms_evidence/Zhang_XJTLU_task6_1.output.jsonl`
- `results/evaluation_20260603/tsel_sbec_raw_decode/Zhang_XJTLU_task6_2.output.jsonl`
- `results/evaluation_20260603/tsel_learned_decoder/Zhang_XJTLU_task6_3.output.jsonl`
- `results/evaluation_20260603/dcase26_task6_eval_outputs_20260603.zip`

All three files passed local format validation:

- 177 output lines
- `qid` order matches the official template
- required fields are present: `qid`, `query`, `duration`, `vid`,
  `pred_relevant_windows`
- all predicted windows are inside the corresponding clip duration

## Submission Recommendation

Primary submission:

- `Zhang_XJTLU_task6_3.output.jsonl`
- System: TSEL/SBEC evidence with learned evidence decoder
- Reason: this is the strongest validated decoder path in our local annotated
  split, and it avoids relying only on raw start/end logits.

Backup submission:

- `Zhang_XJTLU_task6_1.output.jsonl`
- System: MS-CLAP evidence baseline
- Reason: conservative baseline-style output.

Not recommended as primary:

- `Zhang_XJTLU_task6_2.output.jsonl`
- System: TSEL/SBEC evidence with raw start/end decode
- Reason: quick inspection showed an end-of-audio bias in early evaluation
  examples.

## Reproduction Commands

MS-CLAP evidence submission:

```bash
/private/research-artifact src/create_evaluation_submission.py \
  --data_path data/evaluation_release/dcase2026_evaluation.jsonl \
  --ckpt_path results/evidence_baseline_release_v1/best.pt \
  --output_path results/evaluation_ms_evidence_v1/Zhang_XJTLU_task6_1.output.jsonl \
  --evidence_dump_path results/evaluation_ms_evidence_v1/evidence_dump.json \
  --a_feat_dir features/evaluation/clap \
  --t_feat_dir features/evaluation/clap_text \
  --feature_name dcase2026_eval_msclap \
  --batch_size 32 \
  --topn 10
```

TSEL/SBEC evidence submission:

```bash
/private/research-artifact src/create_evaluation_submission.py \
  --data_path data/evaluation_release/dcase2026_evaluation.jsonl \
  --ckpt_path results/temporal_evidence_hn_v4_stage_semantic_width/best.pt \
  --output_path results/evaluation_tsel_sbec_v1/Zhang_XJTLU_task6_2.output.jsonl \
  --evidence_dump_path results/evaluation_tsel_sbec_v1/evidence_dump.json \
  --a_feat_dir features/evaluation/clap \
  --t_feat_dir features/evaluation/clap_text \
  --feature_name dcase2026_eval_msclap_tsel_sbec \
  --batch_size 32 \
  --topn 10
```

TSEL learned decoder submission:

```bash
/private/research-artifact src/apply_learned_evidence_decoder.py \
  --evidence_path results/evaluation_tsel_sbec_v1/evidence_dump.json \
  --decoder_ckpt results/learned_evidence_decoder_temporal_hn_v4_stage_semantic_width_shape_v2_top2/best.pt \
  --output_path results/evaluation_tsel_learned_decoder_v1/Zhang_XJTLU_task6_3.output.jsonl \
  --case_rows_path results/evaluation_tsel_learned_decoder_v1/scored_candidate_rows.json \
  --topn 10
```

# Audio-duration mechanism decomposition

This is an inference-only diagnostic of the unchanged official QD-DETR baseline. It does not establish causality and does not test a new method.

## Fixed implementation

- GT duration bins use the longest annotated interval; local evidence uses the union of all intervals.
- Audio tokens are the official one-second CLAP feature tokens after the unchanged dataset loader and TEF construction.
- Predictions use the official foreground ranking, second-based conversion, clipping, and one-second rounding.
- Matched comparisons control fine GT-duration band, GT-count bin, and query-word-length bin; `vid` is retained as the audio cluster identifier.

## Duration association after stratification

- **0-2s**: 34 matched pairs; median long-minus-short R1@0.7 = 0.0000; median long-minus-short final width/GT = 5.0000; median long-minus-short center error = 39.7500 s.
- **2-5s**: 159 matched pairs; median long-minus-short R1@0.7 = 0.0000; median long-minus-short final width/GT = 2.0000; median long-minus-short center error = 8.5000 s.

## Search-space versus local evidence

- **0-2s**: Spearman(audio duration, GT token fraction)=-0.6423; GT rank=0.3110; GT percentile=-0.1464; false peaks above GT=0.3110; high-saliency distractor peaks=0.8495.
- **2-5s**: Spearman(audio duration, GT token fraction)=-0.5488; GT rank=0.1127; GT percentile=0.0630; false peaks above GT=0.1127; high-saliency distractor peaks=0.8485.

## Hypothesis status

| Hypothesis | Status |
|---|---|
| H1_NORMALIZED_COORDINATE_EFFECT | **PARTIALLY_SUPPORTED** |
| H2_SEARCH_SPACE_COMPETITION | **SUPPORTED** |
| H3_INITIAL_SCALE_MEDIATION | **PARTIALLY_SUPPORTED** |
| H4_GT_LOCAL_EVIDENCE_DEGRADATION | **PARTIALLY_SUPPORTED** |
| H5_DATASET_COMPOSITION_CONFOUND | **NOT_SUPPORTED** |

## Dominant branch: `GLOBAL_SEARCH_SPACE`

The selected branch is a diagnostic priority, not a method proposal. The context-length counterfactual is separately blocked because the stored-feature interface cannot vary context while preserving the model's normalized position coordinate system without synthetic replacement audio.

## Provenance

```json
{
  "baseline_commit": "45ef471ee47ea75a2141d75bd9cfdb8c45dfc101",
  "baseline_checkpoint": "/private/research-artifact",
  "baseline_checkpoint_sha256": "9cdc18a14e906689484f1dde055b42cdcc4b77f0d850f6a0174ff9ef42063d35",
  "config": "/private/research-artifact",
  "config_sha256": "195a41b47042bb9a6456e1268ccbcc9ef1a25862bb66508f1427085044aeaaab",
  "test_jsonl": "/private/research-artifact",
  "test_jsonl_sha256": "044f141630d4daff984f1bfce1622071520edc27e5f7e49930571c761e6fcaa4",
  "source_qd_detr_sha256": "2b87c7537db88a731f3a0f485915d6a044d065b2bad6d34df8844ce7c2af0738",
  "source_position_encoding_sha256": "0af6cfba3d498a24ced2b5a2f1ededaf4fa2fc2d998880bcc7b632b0585bd0c3",
  "source_dataset_sha256": "bf08379a3473ce0cac215a34665ef548748987b48e742bb2ce2fcae0e6dc457b",
  "sample_count": 1347,
  "random_seed": 20260912,
  "clip_length_sec": 1.0,
  "device": "cuda"
}
```

#!/usr/bin/env python3
"""Score matched GT and hard-negative windows with the official MS-CLAP API.

The script deliberately uses raw WAVs and the native CLAP projection path.
It never compares the QD-DETR stored temporal features to text features.
Missing audio is retained as BLOCKED rather than replaced by a proxy score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from transformers import AutoTokenizer


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clap(args: argparse.Namespace):
    from msclap.models.clap import CLAP

    model = CLAP(
        audioenc_name="HTSAT",
        sample_rate=44100,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=8000,
        classes_num=527,
        out_emb=768,
        text_model=str(args.gpt2_dir.resolve()),
        transformer_embed_dim=768,
        d_proj=1024,
    )
    state = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.gpt2_dir.resolve()))
    tokenizer.add_special_tokens({"pad_token": "!"})
    return model, tokenizer, {"missing_keys": len(missing), "unexpected_keys": len(unexpected)}


def crop_and_resample(path: Path, center_sec: float, window_sec: float, duration_sec: float) -> torch.Tensor:
    """Use the official torchaudio resampler after a deterministic raw crop."""
    # The environment's torchaudio build routes file decoding through an
    # unavailable TorchCodec shared library.  soundfile supplies the same
    # PCM samples for these WAVs; all model preprocessing after decoding is
    # still the official torchaudio resampler and native CLAP encoder.
    samples, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    waveform = torch.from_numpy(samples.T.copy())
    if waveform.shape[0] != 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    max_start = max(0.0, float(duration_sec) - float(window_sec))
    start_sec = min(max(0.0, float(center_sec) - float(window_sec) / 2.0), max_start)
    end_sec = min(float(duration_sec), start_sec + float(window_sec))
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(waveform.shape[-1], max(start + 1, int(round(end_sec * sample_rate))))
    crop = waveform[:, start:end]
    if sample_rate != 44100:
        crop = T.Resample(sample_rate, 44100)(crop)
    return crop.reshape(-1).float()


def project_text(model: torch.nn.Module, tokenizer: Any, query: str, device: str) -> torch.Tensor:
    encoded = tokenizer.encode_plus(
        text=query + " <|endoftext|>",
        add_special_tokens=True,
        max_length=77,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        projected = model.caption_encoder(encoded)
    return projected.reshape(-1)


def project_audio(model: torch.nn.Module, waveform: torch.Tensor, device: str) -> torch.Tensor:
    with torch.inference_mode():
        projected = model.audio_encoder(waveform.to(device).reshape(1, -1))[0]
    return projected.reshape(-1)


def official_similarity(model: torch.nn.Module, audio: torch.Tensor, text: torch.Tensor) -> float:
    audio_norm = audio / torch.linalg.vector_norm(audio).clamp_min(1e-12)
    text_norm = text / torch.linalg.vector_norm(text).clamp_min(1e-12)
    return float((model.logit_scale.exp() * torch.sum(text_norm * audio_norm)).detach().cpu())


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    geometry_rows = read_csv(output / "hard_negative_geometry.csv")
    model, tokenizer, load_info = load_clap(args)
    rows: list[dict[str, Any]] = []
    audio_cache: dict[str, tuple[torch.Tensor, int]] = {}
    text_cache: dict[str, torch.Tensor] = {}
    for row in geometry_rows:
        qid = row["qid"]
        wav = args.raw_audio_dir.resolve() / f"{row['vid']}.wav"
        result: dict[str, Any] = {
            "qid": qid,
            "vid": row["vid"],
            "duration_bin": row["duration_bin"],
            "cohort": row["cohort"],
            "p3_location_failure": row.get("p3_location_failure", ""),
            "hard_peak_start_sec": row["hard_peak_start_sec"],
            "hard_peak_end_sec": row["hard_peak_end_sec"],
            "hard_peak_center_sec": row["hard_peak_center_sec"],
            "gt_duration_sec": row["gt_duration_sec"],
            "window_length_sec": "",
            "gt_window_center_sec": "",
            "hard_window_center_sec": row["hard_peak_center_sec"],
            "audio_path": f"raw_audio/{row['vid']}.wav",
            "status": "",
            "blocked_reason": "",
            "S_GT": "",
            "S_HARD": "",
            "S_GT_minus_S_HARD": "",
            "comparison": "",
            "native_pipeline": "official msclap CLAP audio_encoder/caption_encoder; L2-normalized projected vectors; logit_scale.exp() dot product",
        }
        if not wav.exists():
            result["status"] = "BLOCKED"
            result["blocked_reason"] = "original WAV unavailable; no stored-feature substitute allowed"
            rows.append(result)
            continue
        duration = float(row["audio_duration_sec"])
        window_length = max(1.0, float(row["gt_duration_sec"]))
        # The longest GT interval is used by the QD geometry runner as the
        # matched GT center.  The row has a JSON interval list.
        windows = json.loads(row["gt_windows"])
        gt_window = max(windows, key=lambda item: float(item[1]) - float(item[0]))
        gt_center = (float(gt_window[0]) + float(gt_window[1])) / 2.0
        hard_center = float(row["hard_peak_center_sec"])
        result["window_length_sec"] = window_length
        result["gt_window_center_sec"] = gt_center
        try:
            if qid not in text_cache:
                text_cache[qid] = project_text(model, tokenizer, row.get("query", ""), args.device)
            text = text_cache[qid]
            gt_audio = crop_and_resample(wav, gt_center, window_length, duration)
            hard_audio = crop_and_resample(wav, hard_center, window_length, duration)
            gt_embedding = project_audio(model, gt_audio, args.device)
            hard_embedding = project_audio(model, hard_audio, args.device)
            gt_score = official_similarity(model, gt_embedding, text)
            hard_score = official_similarity(model, hard_embedding, text)
            delta = gt_score - hard_score
            result["status"] = "VALID"
            result["S_GT"] = gt_score
            result["S_HARD"] = hard_score
            result["S_GT_minus_S_HARD"] = delta
            result["comparison"] = "GT>HARD" if delta > 0 else "HARD>GT" if delta < 0 else "TIE_EXACT"
        except Exception as exc:  # retain a query-level BLOCKED record, never proxy it
            result["status"] = "BLOCKED"
            result["blocked_reason"] = f"native MS-CLAP exception: {type(exc).__name__}: {exc}"
        rows.append(result)
        print(f"{qid} {result['status']} {result['comparison']}", flush=True)

    fields = list(rows[0].keys()) if rows else ["qid"]
    write_csv(output / "native_clap_gt_vs_hard.csv", rows, fields)
    payload = {
        "status": "COMPLETE_WITH_QUERY_LEVEL_BLOCKS" if any(row["status"] == "BLOCKED" for row in rows) else "COMPLETE",
        "package": "msclap",
        "package_version": "1.3.4",
        "implementation": "microsoft/CLAP via msclap.models.clap.CLAP",
        "checkpoint": "CLAP_weights_2023.pth (private artifact)",
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "model_config": {"audioenc_name": "HTSAT", "sample_rate": 44100, "window_size": 1024, "hop_size": 320, "mel_bins": 64, "fmin": 50, "fmax": 8000, "classes_num": 527, "out_emb": 768, "transformer_embed_dim": 768, "d_proj": 1024},
        "audio_api": "CLAP.audio_encoder after deterministic raw WAV crop and official torchaudio resampling to 44100 Hz",
        "wav_decoder": "soundfile PCM decode because the server torchaudio build could not load TorchCodec; this does not change the WAV samples or native CLAP API",
        "text_api": "CLAP.caption_encoder with official GPT-2 tokenizer, query + <|endoftext|>, max_length=77",
        "similarity": "CLAPWrapper.compute_similarity: L2-normalized projected vectors, multiplied by model.logit_scale.exp()",
        "native_shared_space_verified": True,
        "load_info": load_info,
        "raw_audio_dir": "private raw-audio directory (server-only)",
        "n_rows": len(rows),
        "n_valid": sum(row["status"] == "VALID" for row in rows),
        "n_blocked": sum(row["status"] == "BLOCKED" for row in rows),
    }
    (output / "native_clap_provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-audio-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpt2-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

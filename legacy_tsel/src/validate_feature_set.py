import argparse
import json
import os
from collections import Counter

import torch  # noqa: F401 - keeps this Windows/UNC environment consistent with training scripts.
import numpy as np


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_npz(path, key, expected_dim=None, max_len=None):
    if not os.path.exists(path):
        return {"ok": False, "error": "missing"}
    try:
        array = np.load(path)[key]
    except KeyError:
        return {"ok": False, "error": f"missing_key:{key}"}
    except Exception as exc:
        return {"ok": False, "error": f"load_error:{type(exc).__name__}"}

    if array.ndim != 2:
        return {"ok": False, "error": f"bad_rank:{array.ndim}", "shape": list(array.shape)}
    if expected_dim is not None and int(array.shape[1]) != int(expected_dim):
        return {"ok": False, "error": f"bad_dim:{array.shape[1]}", "shape": list(array.shape)}
    if max_len is not None and int(array.shape[0]) > int(max_len):
        return {"ok": True, "warning": f"longer_than_max:{array.shape[0]}", "shape": list(array.shape)}
    return {"ok": True, "shape": list(array.shape)}


def summarize_checks(rows, args):
    audio_seen = {}
    text_seen = {}
    audio_errors = Counter()
    text_errors = Counter()
    audio_shapes = Counter()
    text_shapes = Counter()
    warnings = Counter()

    for row in rows:
        vid = row["vid"]
        qid = row["qid"]
        if vid not in audio_seen:
            audio_path = os.path.join(args.a_feat_dir, f"{vid}.npz")
            audio_seen[vid] = check_npz(
                path=audio_path,
                key="features",
                expected_dim=args.a_feat_dim,
                max_len=args.max_a_l,
            )
        if qid not in text_seen:
            text_path = os.path.join(args.t_feat_dir, f"qid{qid}.npz")
            text_seen[qid] = check_npz(
                path=text_path,
                key="last_hidden_state",
                expected_dim=args.t_feat_dim,
                max_len=args.max_q_l,
            )

    for result in audio_seen.values():
        if not result["ok"]:
            audio_errors[result["error"]] += 1
        else:
            audio_shapes[tuple(result["shape"])] += 1
            if "warning" in result:
                warnings[f"audio:{result['warning']}"] += 1

    for result in text_seen.values():
        if not result["ok"]:
            text_errors[result["error"]] += 1
        else:
            text_shapes[tuple(result["shape"])] += 1
            if "warning" in result:
                warnings[f"text:{result['warning']}"] += 1

    return {
        "data_path": args.data_path,
        "items": len(rows),
        "unique_audio": len(audio_seen),
        "unique_text": len(text_seen),
        "audio_errors": dict(audio_errors),
        "text_errors": dict(text_errors),
        "warnings": dict(warnings),
        "top_audio_shapes": [{"shape": list(shape), "count": count} for shape, count in audio_shapes.most_common(10)],
        "top_text_shapes": [{"shape": list(shape), "count": count} for shape, count in text_shapes.most_common(10)],
        "ok": not audio_errors and not text_errors,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--a_feat_dir", required=True)
    parser.add_argument("--t_feat_dir", required=True)
    parser.add_argument("--a_feat_dim", type=int, required=True)
    parser.add_argument("--t_feat_dim", type=int, required=True)
    parser.add_argument("--max_a_l", type=int, default=300)
    parser.add_argument("--max_q_l", type=int, default=32)
    parser.add_argument("--output_path", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(args.data_path)
    summary = summarize_checks(rows, args)
    print(json.dumps(summary, indent=2))
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

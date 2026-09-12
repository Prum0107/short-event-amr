import argparse
import json
import os

import torch

from train_learned_evidence_decoder import (
    CandidateDataset,
    CandidateScorer,
    extract_candidate_features,
    generate_candidates,
    load_items,
    make_feature_context,
    parse_source_quotas,
    rows_to_predictions,
    score_rows,
)


def save_json(obj, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(rows, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_unlabeled_rows(items, topn_per_source, feature_version, source_quotas):
    rows = []
    for item in items:
        context = make_feature_context(item)
        for cand in generate_candidates(item, topn_per_source=topn_per_source, source_quotas=source_quotas):
            rows.append(
                {
                    "qid": item["qid"],
                    "query": item.get("query", ""),
                    "vid": item.get("vid", ""),
                    "duration": item.get("duration", len(item.get("evidence_scores", []))),
                    "window": cand["window"],
                    "source": cand["source"],
                    "source_rank": cand["source_rank"],
                    "features": extract_candidate_features(
                        item,
                        cand,
                        feature_version=feature_version,
                        context=context,
                    ),
                    "target_iou": 0.0,
                }
            )
    return rows


def load_decoder(path, input_dim, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = CandidateScorer(
        input_dim=input_dim,
        hidden_dim=int(args.get("hidden_dim", 128)),
        dropout=float(args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def to_submission(items, pred_by_qid, topn):
    rows = []
    for item in items:
        duration = float(item.get("duration", len(item.get("evidence_scores", []))))
        windows = pred_by_qid.get(item["qid"], [])[:topn]
        clean_windows = []
        for window in windows:
            st = max(0.0, min(float(window[0]), duration))
            ed = max(st, min(float(window[1]), duration))
            if ed > st:
                clean_windows.append([st, ed])
        rows.append(
            {
                "qid": item["qid"],
                "query": item.get("query", ""),
                "duration": int(duration) if duration.is_integer() else duration,
                "vid": item.get("vid", ""),
                "pred_relevant_windows": clean_windows,
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_path", required=True)
    parser.add_argument("--decoder_ckpt", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--case_rows_path", default="")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--topn_per_source", type=int, default=None)
    parser.add_argument("--source_quotas", default=None)
    parser.add_argument("--feature_version", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    items = load_items(args.evidence_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    probe_ckpt = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=False)
    ckpt_args = probe_ckpt.get("args", {})
    topn_per_source = int(args.topn_per_source or ckpt_args.get("topn_per_source", 10))
    feature_version = args.feature_version or ckpt_args.get("feature_version", "basic")
    source_quotas = parse_source_quotas(args.source_quotas if args.source_quotas is not None else ckpt_args.get("source_quotas", ""))

    rows = build_unlabeled_rows(
        items,
        topn_per_source=topn_per_source,
        feature_version=feature_version,
        source_quotas=source_quotas,
    )
    feature_mean = probe_ckpt["feature_mean"].cpu()
    feature_std = probe_ckpt["feature_std"].cpu()
    dataset = CandidateDataset(rows, feature_mean=feature_mean, feature_std=feature_std)
    model, ckpt = load_decoder(args.decoder_ckpt, dataset.features.shape[1], device)
    scored_rows = score_rows(model, dataset, rows, device)
    pred_by_qid = rows_to_predictions(scored_rows, topn=args.topn)
    submission = to_submission(items, pred_by_qid, args.topn)
    save_jsonl(submission, args.output_path)
    if args.case_rows_path:
        save_json(scored_rows, args.case_rows_path)
    save_json(
        {
            "num_queries": len(items),
            "num_candidates": len(rows),
            "output_path": args.output_path,
            "decoder_ckpt": args.decoder_ckpt,
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_metrics": ckpt.get("metrics", {}),
            "topn_per_source": topn_per_source,
            "feature_version": feature_version,
            "source_quotas": source_quotas,
        },
        os.path.splitext(args.output_path)[0] + ".summary.json",
    )
    print(json.dumps({"num_queries": len(items), "num_candidates": len(rows), "output_path": args.output_path}, indent=2))


if __name__ == "__main__":
    main()

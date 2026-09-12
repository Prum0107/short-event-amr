import argparse
import json
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from config import BaseOptions
from dataset import StartEndDataset, start_end_collate
from evidence_baseline import EvidenceBaseline, decode_evidence_windows
from pipeline_utils import apply_feature_overrides, feature_config_summary, torch_load_trusted, unpack_batch
from train_evidence_baseline import select_audio_input


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


def make_eval_loader(opt, data_path, batch_size, num_workers):
    dataset = StartEndDataset(
        data_path=data_path,
        a_feat_dir=opt.a_feat_dir,
        q_feat_dir=opt.t_feat_dir,
        max_q_l=opt.max_q_l,
        max_a_l=opt.max_a_l,
        ctx_mode=opt.ctx_mode,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        load_labels=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=start_end_collate,
    )
    return dataset, loader


def load_model(ckpt_path, device):
    ckpt = torch_load_trusted(ckpt_path, map_location=device)
    model_config = ckpt["model_config"]
    model = EvidenceBaseline(**model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict(model, loader, opt, args, device):
    rows = []
    evidence_dump = []
    for batch in loader:
        batch_meta, query_feat, query_mask, audio_feat_full, audio_feat_clap, audio_mask = unpack_batch(
            batch=batch,
            device=device,
            a_feat_dim=opt.a_feat_dim,
        )
        audio_feat = select_audio_input(audio_feat_full, audio_feat_clap, args.use_tef)
        outputs = model(
            audio_feat=audio_feat,
            query_feat=query_feat,
            audio_mask=audio_mask,
            query_mask=query_mask,
        )

        for idx, meta in enumerate(batch_meta):
            valid_len = int(audio_mask[idx].sum().item())
            windows = decode_evidence_windows(
                evidence_logits=outputs["evidence_logits"][idx],
                start_logits=outputs["start_logits"][idx],
                end_logits=outputs["end_logits"][idx],
                valid_len=valid_len,
                clip_length=opt.clip_length,
                topn=args.topn,
                min_len=args.decode_min_len,
                max_len=args.decode_max_len,
                evidence_weight=args.decode_evidence_weight,
                nms_threshold=args.decode_nms,
            )
            duration = float(meta.get("duration", valid_len * opt.clip_length))
            row = {
                "qid": meta["qid"],
                "query": meta.get("query", ""),
                "duration": int(duration) if duration.is_integer() else duration,
                "vid": meta.get("vid", ""),
                "pred_relevant_windows": [[float(w[0]), float(w[1])] for w in windows],
            }
            rows.append(row)
            if len(evidence_dump) < args.max_evidence_dump:
                evidence_probs = torch.sigmoid(outputs["evidence_logits"][idx, :valid_len]).detach().cpu().tolist()
                evidence_dump.append(
                    {
                        **row,
                        "pred_relevant_windows_with_scores": windows,
                        "evidence_scores": evidence_probs,
                    }
                )
    return rows, evidence_dump


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--ckpt_path", default="results/evidence_baseline_release_v1/best.pt")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--evidence_dump_path", default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--a_feat_dir", default=None)
    parser.add_argument("--t_feat_dir", default=None)
    parser.add_argument("--a_feat_dim", type=int, default=None)
    parser.add_argument("--t_feat_dim", type=int, default=None)
    parser.add_argument("--ctx_mode", default=None)
    parser.add_argument("--clip_length", type=float, default=None)
    parser.add_argument("--max_a_l", type=int, default=None)
    parser.add_argument("--feature_name", default="config")
    parser.add_argument("--use_tef", action="store_true", default=True)
    parser.add_argument("--no_tef", action="store_false", dest="use_tef")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--decode_min_len", type=int, default=1)
    parser.add_argument("--decode_max_len", type=int, default=150)
    parser.add_argument("--decode_evidence_weight", type=float, default=1.0)
    parser.add_argument("--decode_nms", type=float, default=0.7)
    parser.add_argument("--max_evidence_dump", type=int, default=1000000)
    return parser.parse_args()


def main():
    args = parse_args()
    option_manager = BaseOptions(args.config)
    option_manager.parse()
    opt = apply_feature_overrides(option_manager.option, args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("feature_name:", args.feature_name)
    print("feature_config:", json.dumps(feature_config_summary(opt), indent=2))

    model, ckpt = load_model(args.ckpt_path, device)
    _, loader = make_eval_loader(opt, args.data_path, args.batch_size, args.num_workers)
    rows, evidence_dump = predict(model, loader, opt, SimpleNamespace(**vars(args)), device)
    save_jsonl(rows, args.output_path)
    if args.evidence_dump_path:
        save_json(evidence_dump, args.evidence_dump_path)

    summary = {
        "num_queries": len(rows),
        "output_path": args.output_path,
        "evidence_dump_path": args.evidence_dump_path,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_metric": ckpt.get("best_metric"),
        "feature_name": args.feature_name,
        "feature_config": feature_config_summary(opt),
    }
    save_json(summary, os.path.splitext(args.output_path)[0] + ".summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

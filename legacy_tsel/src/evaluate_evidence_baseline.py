import argparse
import json
import os
from types import SimpleNamespace

import torch

from config import BaseOptions
from evidence_baseline import EvidenceBaseline
from pipeline_utils import apply_feature_overrides, feature_config_summary, torch_load_trusted
from train_evidence_baseline import evaluate, make_loader


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_model(ckpt_path, device):
    ckpt = torch_load_trusted(ckpt_path, map_location=device)
    model_config = ckpt["model_config"]
    model = EvidenceBaseline(**model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--data_path", default="data/castella_val_release.jsonl")
    parser.add_argument("--ckpt_path", default="results/evidence_baseline_release_v1/best.pt")
    parser.add_argument("--output_dir", default="results/evidence_baseline_release_v1/full_val_eval")
    parser.add_argument("--batch_size", type=int, default=16)
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
    parser.add_argument("--evidence_sigma", type=float, default=2.0)
    parser.add_argument("--boundary_sigma", type=float, default=1.5)
    parser.add_argument("--lambda_boundary", type=float, default=0.5)
    parser.add_argument("--use_pos_weight", action="store_true", default=True)
    parser.add_argument("--no_pos_weight", action="store_false", dest="use_pos_weight")

    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--decode_min_len", type=int, default=1)
    parser.add_argument("--decode_max_len", type=int, default=150)
    parser.add_argument("--decode_evidence_weight", type=float, default=1.0)
    parser.add_argument("--decode_nms", type=float, default=0.7)
    parser.add_argument("--official_eval", action="store_true")
    parser.add_argument("--max_evidence_dump", type=int, default=1000000)
    return parser.parse_args()


def main():
    args = parse_args()
    option_manager = BaseOptions(args.config)
    option_manager.parse()
    opt = option_manager.option
    opt = apply_feature_overrides(opt, args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("feature_name:", args.feature_name)
    print("feature_config:", json.dumps(feature_config_summary(opt), indent=2))
    model, ckpt = load_model(args.ckpt_path, device)
    _, loader = make_loader(
        opt,
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    dataset = loader.dataset

    os.makedirs(args.output_dir, exist_ok=True)
    predictions_path = os.path.join(args.output_dir, "predictions.jsonl")
    official_path = os.path.join(args.output_dir, "official_metrics.json")
    eval_args = SimpleNamespace(**vars(args))
    metrics = evaluate(
        model=model,
        dataset=dataset,
        loader=loader,
        opt=opt,
        args=eval_args,
        device=device,
        predictions_path=predictions_path,
        official_metrics_path=official_path,
    )
    metrics["checkpoint_epoch"] = ckpt.get("epoch")
    metrics["checkpoint_metric"] = ckpt.get("best_metric")
    metrics["feature_name"] = args.feature_name
    metrics["feature_config"] = feature_config_summary(opt)
    save_json(metrics, os.path.join(args.output_dir, "metrics.json"))
    print(json.dumps(metrics, indent=2))
    print("saved predictions to", predictions_path)


if __name__ == "__main__":
    main()

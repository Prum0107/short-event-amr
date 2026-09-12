import argparse
import json
import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from config import BaseOptions
from dataset import StartEndDataset, start_end_collate
from evidence_baseline import (
    EvidenceBaseline,
    build_evidence_targets,
    decode_evidence_windows,
    evidence_curve_stats,
    evidence_loss,
    hard_negative_ranking_loss,
    type_aware_hard_negative_loss,
)
from metrics import best_iou_among_topk, recall_at_1_iou, recall_at_k_iou
from pipeline_utils import apply_feature_overrides, feature_config_summary, make_dataset_config, torch_load_trusted, unpack_batch


def save_jsonl(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_hard_negatives(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "hard_negatives" in payload:
        return payload["hard_negatives"]
    return payload


def hard_negative_is_typed(hard_negative_map):
    if not hard_negative_map:
        return False
    first_value = next(iter(hard_negative_map.values()))
    return isinstance(first_value, dict)


def count_hard_negatives(hard_negative_map):
    total = 0
    for value in hard_negative_map.values():
        if isinstance(value, dict):
            total += sum(len(candidates) for candidates in value.values())
        else:
            total += len(value)
    return total


def configure_trainable_parameters(model, trainable_parts):
    if trainable_parts == "all":
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
        modules = []
        if trainable_parts == "boundary_heads":
            modules = [model.start_head, model.end_head]
        elif trainable_parts == "heads":
            modules = [model.evidence_head, model.start_head, model.end_head]
        elif trainable_parts == "evidence_head":
            modules = [model.evidence_head]
        else:
            raise ValueError(f"unknown trainable_parts: {trainable_parts}")
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def set_frozen_modules_eval(model, trainable_parts):
    if trainable_parts == "all":
        return
    frozen_modules = [model.audio_proj, model.query_proj, model.fuse, model.temporal_layers]
    if trainable_parts != "evidence_head":
        frozen_modules.append(model.evidence_head)
    if trainable_parts not in ["boundary_heads", "heads"]:
        frozen_modules.extend([model.start_head, model.end_head])
    for module in frozen_modules:
        module.eval()


def load_initial_checkpoint(model, path, device):
    if not path:
        return None
    ckpt = torch_load_trusted(path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return ckpt


def make_loader(opt, data_path, batch_size, num_workers, shuffle):
    dataset = StartEndDataset(**make_dataset_config(opt, data_path))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=start_end_collate,
    )
    return dataset, loader


def select_audio_input(audio_feat_full, audio_feat_clap, use_tef):
    return audio_feat_full if use_tef else audio_feat_clap


def train_one_epoch(model, loader, optimizer, opt, args, device, hard_negative_map=None):
    model.train()
    set_frozen_modules_eval(model, args.trainable_parts)
    meters = defaultdict(float)
    num_steps = 0

    for step, batch in enumerate(loader):
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
        targets = build_evidence_targets(
            batch_meta=batch_meta,
            audio_length=outputs["evidence_logits"].shape[1],
            clip_length=opt.clip_length,
            device=device,
            evidence_sigma=args.evidence_sigma,
            boundary_sigma=args.boundary_sigma,
        )
        loss, loss_items = evidence_loss(
            outputs=outputs,
            targets=targets,
            audio_mask=audio_mask,
            lambda_boundary=args.lambda_boundary,
            use_pos_weight=args.use_pos_weight,
        )
        use_type_aware = hard_negative_map and (
            args.hard_negative_mode == "type_aware"
            or (args.hard_negative_mode == "auto" and hard_negative_is_typed(hard_negative_map))
        )
        if use_type_aware and (args.lambda_semantic_hard_negative > 0 or args.lambda_boundary_hard_negative > 0):
            hn_loss, hn_items = type_aware_hard_negative_loss(
                outputs=outputs,
                batch_meta=batch_meta,
                hard_negative_map=hard_negative_map,
                audio_mask=audio_mask,
                clip_length=opt.clip_length,
                semantic_margin=args.semantic_hn_margin,
                boundary_margin=args.boundary_hn_margin,
                semantic_topk=args.semantic_hn_topk,
                boundary_topk=args.boundary_hn_topk,
                lambda_semantic=args.lambda_semantic_hard_negative,
                lambda_boundary=args.lambda_boundary_hard_negative,
            )
            loss = loss + hn_loss
            loss_items["loss"] = loss.detach()
            for key, value in hn_items.items():
                if torch.is_tensor(value):
                    loss_items[key] = value.detach()
                else:
                    loss_items[key] = torch.tensor(float(value))
        elif hard_negative_map and args.lambda_hard_negative > 0:
            hn_loss, hn_items = hard_negative_ranking_loss(
                evidence_logits=outputs["evidence_logits"],
                batch_meta=batch_meta,
                hard_negative_map=hard_negative_map,
                audio_mask=audio_mask,
                clip_length=opt.clip_length,
                margin=args.hn_margin,
                topk=args.hn_topk,
            )
            loss = loss + args.lambda_hard_negative * hn_loss
            loss_items["loss"] = loss.detach()
            loss_items["hard_negative_loss"] = hn_loss.detach()
            loss_items["hard_negative_pairs"] = torch.tensor(float(hn_items["hard_negative_pairs"]))
            loss_items["hard_negative_gap"] = torch.tensor(float(hn_items["hard_negative_gap"]))

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        stats = evidence_curve_stats(outputs["evidence_logits"], targets, audio_mask)
        for key, value in loss_items.items():
            meters[key] += float(value.item())
        for key, value in stats.items():
            meters[key] += float(value)
        num_steps += 1

        if step % args.log_interval == 0:
            print(
                f"[train step {step}] "
                f"loss={loss_items['loss'].item():.4f} "
                f"evidence={loss_items['evidence_loss'].item():.4f} "
                f"boundary={loss_items['boundary_loss'].item():.4f} "
                f"gap={stats['evidence_gap']:.4f} "
                f"hn={loss_items.get('hard_negative_loss', torch.tensor(0.0)).item():.4f} "
                f"typed={loss_items.get('typed_hard_negative_loss', torch.tensor(0.0)).item():.4f}"
            )

    return {key: value / max(num_steps, 1) for key, value in meters.items()}


@torch.no_grad()
def evaluate(model, dataset, loader, opt, args, device, predictions_path=None, official_metrics_path=None):
    model.eval()
    meters = defaultdict(float)
    totals = defaultdict(float)
    num_steps = 0
    num_samples = 0
    submissions = []
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
        targets = build_evidence_targets(
            batch_meta=batch_meta,
            audio_length=outputs["evidence_logits"].shape[1],
            clip_length=opt.clip_length,
            device=device,
            evidence_sigma=args.evidence_sigma,
            boundary_sigma=args.boundary_sigma,
        )
        loss, loss_items = evidence_loss(
            outputs=outputs,
            targets=targets,
            audio_mask=audio_mask,
            lambda_boundary=args.lambda_boundary,
            use_pos_weight=args.use_pos_weight,
        )
        stats = evidence_curve_stats(outputs["evidence_logits"], targets, audio_mask)
        for key, value in loss_items.items():
            meters[key] += float(value.item())
        for key, value in stats.items():
            meters[key] += float(value)
        num_steps += 1

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
            gt_windows = meta["relevant_windows"]
            submissions.append(
                {
                    "qid": meta["qid"],
                    "query": meta.get("query", ""),
                    "vid": meta.get("vid", ""),
                    "pred_relevant_windows": windows,
                }
            )
            evidence_probs = torch.sigmoid(outputs["evidence_logits"][idx, :valid_len]).detach().cpu().tolist()
            evidence_dump.append(
                {
                    "qid": meta["qid"],
                    "query": meta.get("query", ""),
                    "vid": meta.get("vid", ""),
                    "duration": meta.get("duration", valid_len * opt.clip_length),
                    "gt_windows": gt_windows,
                    "pred_relevant_windows": windows,
                    "evidence_scores": evidence_probs,
                }
            )
            totals["R1@0.5"] += recall_at_1_iou(windows, gt_windows, threshold=0.5)
            totals["R1@0.7"] += recall_at_1_iou(windows, gt_windows, threshold=0.7)
            totals["R3@0.5"] += recall_at_k_iou(windows, gt_windows, threshold=0.5, k=3)
            totals["R3@0.7"] += recall_at_k_iou(windows, gt_windows, threshold=0.7, k=3)
            totals["best_iou_top5"] += best_iou_among_topk(windows, gt_windows, k=5)
            num_samples += 1

    metrics = {key: value / max(num_steps, 1) for key, value in meters.items()}
    metrics.update({key: value / max(num_samples, 1) for key, value in totals.items()})
    metrics["num_samples"] = num_samples

    if predictions_path:
        save_jsonl(submissions, predictions_path)
        save_json(evidence_dump[: args.max_evidence_dump], predictions_path.replace(".jsonl", "_evidence_samples.json"))

    if args.official_eval and official_metrics_path:
        from standalone_eval.eval import eval_submission

        official = eval_submission(submissions, dataset.data, verbose=False)
        metrics["official_brief"] = official["brief"]
        save_json(official, official_metrics_path)

    return metrics


def build_model(opt, args, device):
    audio_dim = opt.a_feat_dim + 2 if args.use_tef else opt.a_feat_dim
    return EvidenceBaseline(
        audio_in_dim=audio_dim,
        query_in_dim=opt.t_feat_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        temporal_kernel=args.temporal_kernel,
        temporal_layers=args.temporal_layers,
    ).to(device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--train_data_path", default="data/castella_train_small.jsonl")
    parser.add_argument("--val_data_path", default="data/castella_val_small.jsonl")
    parser.add_argument("--results_dir", default="results/evidence_baseline")
    parser.add_argument("--save_path", default="results/evidence_baseline/best.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--init_ckpt_path", default="")
    parser.add_argument("--trainable_parts", choices=["all", "boundary_heads", "heads", "evidence_head"], default="all")

    parser.add_argument("--a_feat_dir", default=None)
    parser.add_argument("--t_feat_dir", default=None)
    parser.add_argument("--a_feat_dim", type=int, default=None)
    parser.add_argument("--t_feat_dim", type=int, default=None)
    parser.add_argument("--ctx_mode", default=None)
    parser.add_argument("--clip_length", type=float, default=None)
    parser.add_argument("--max_a_l", type=int, default=None)
    parser.add_argument("--feature_name", default="config")

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temporal_kernel", type=int, default=3)
    parser.add_argument("--temporal_layers", type=int, default=2)
    parser.add_argument("--use_tef", action="store_true", default=True)
    parser.add_argument("--no_tef", action="store_false", dest="use_tef")

    parser.add_argument("--evidence_sigma", type=float, default=2.0)
    parser.add_argument("--boundary_sigma", type=float, default=1.5)
    parser.add_argument("--lambda_boundary", type=float, default=0.5)
    parser.add_argument("--hard_negative_path", default="")
    parser.add_argument("--hard_negative_mode", choices=["auto", "generic", "type_aware"], default="auto")
    parser.add_argument("--lambda_hard_negative", type=float, default=0.0)
    parser.add_argument("--hn_margin", type=float, default=0.1)
    parser.add_argument("--hn_topk", type=int, default=3)
    parser.add_argument("--lambda_semantic_hard_negative", type=float, default=0.0)
    parser.add_argument("--lambda_boundary_hard_negative", type=float, default=0.0)
    parser.add_argument("--semantic_hn_margin", type=float, default=0.1)
    parser.add_argument("--boundary_hn_margin", type=float, default=0.2)
    parser.add_argument("--semantic_hn_topk", type=int, default=3)
    parser.add_argument("--boundary_hn_topk", type=int, default=3)
    parser.add_argument("--use_pos_weight", action="store_true", default=True)
    parser.add_argument("--no_pos_weight", action="store_false", dest="use_pos_weight")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--decode_min_len", type=int, default=1)
    parser.add_argument("--decode_max_len", type=int, default=150)
    parser.add_argument("--decode_evidence_weight", type=float, default=1.0)
    parser.add_argument("--decode_nms", type=float, default=0.7)
    parser.add_argument("--save_metric", choices=["R1@0.7", "R1@0.5", "best_iou_top5", "evidence_gap"], default="R1@0.7")
    parser.add_argument("--official_eval", action="store_true")
    parser.add_argument("--max_evidence_dump", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    option_manager = BaseOptions(args.config)
    option_manager.parse()
    opt = option_manager.option
    opt = apply_feature_overrides(opt, args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("train:", args.train_data_path)
    print("val:", args.val_data_path)
    print("results_dir:", args.results_dir)
    print("feature_name:", args.feature_name)
    print("feature_config:", json.dumps(feature_config_summary(opt), indent=2))
    hard_negative_map = load_hard_negatives(args.hard_negative_path)
    if hard_negative_map:
        total_hn = count_hard_negatives(hard_negative_map)
        print(
            "hard negatives:",
            args.hard_negative_path,
            "qid_count:",
            len(hard_negative_map),
            "total:",
            total_hn,
            "typed:",
            hard_negative_is_typed(hard_negative_map),
        )

    os.makedirs(args.results_dir, exist_ok=True)
    train_dataset, train_loader = make_loader(
        opt,
        args.train_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_dataset, val_loader = make_loader(
        opt,
        args.val_data_path,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    print("train size:", len(train_dataset))
    print("val size:", len(val_dataset))

    model = build_model(opt, args, device)
    init_ckpt = load_initial_checkpoint(model, args.init_ckpt_path, device)
    if init_ckpt:
        print("loaded initial checkpoint:", args.init_ckpt_path, "epoch:", init_ckpt.get("epoch"))
    trainable, total = configure_trainable_parameters(model, args.trainable_parts)
    print("trainable parameters:", trainable, "/", total, "mode:", args.trainable_parts)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    best_metric = -1e9

    for epoch in range(args.epochs):
        print(f"\n========== EPOCH {epoch + 1}/{args.epochs} ==========")
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            opt,
            args,
            device,
            hard_negative_map=hard_negative_map,
        )
        val_predictions_path = os.path.join(args.results_dir, f"val_epoch{epoch + 1}.jsonl")
        val_official_path = os.path.join(args.results_dir, f"val_epoch{epoch + 1}_official_metrics.json")
        val_metrics = evaluate(
            model=model,
            dataset=val_dataset,
            loader=val_loader,
            opt=opt,
            args=args,
            device=device,
            predictions_path=val_predictions_path,
            official_metrics_path=val_official_path,
        )

        print("train metrics:", json.dumps(train_metrics, indent=2))
        print("val metrics:", json.dumps(val_metrics, indent=2))
        save_json(
            {
                "epoch": epoch + 1,
                "args": vars(args),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            os.path.join(args.results_dir, f"metrics_epoch{epoch + 1}.json"),
        )

        current = val_metrics[args.save_metric]
        if current > best_metric:
            best_metric = current
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "save_metric": args.save_metric,
                    "args": vars(args),
                    "feature_config": feature_config_summary(opt),
                    "model_config": {
                        "audio_in_dim": opt.a_feat_dim + 2 if args.use_tef else opt.a_feat_dim,
                        "query_in_dim": opt.t_feat_dim,
                        "hidden_dim": args.hidden_dim,
                        "dropout": args.dropout,
                        "temporal_kernel": args.temporal_kernel,
                        "temporal_layers": args.temporal_layers,
                    },
                    "val_metrics": val_metrics,
                },
                args.save_path,
            )
            print(f"saved best checkpoint to {args.save_path} ({args.save_metric}={best_metric:.4f})")


if __name__ == "__main__":
    main()

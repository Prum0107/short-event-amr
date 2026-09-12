import argparse
import json
import math
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def l2_normalize(array, eps=1e-8):
    denom = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(denom, eps)


def safe_window_indices(window, clip_length, valid_len):
    st = int(math.floor(float(window[0]) / clip_length))
    ed = int(math.ceil(float(window[1]) / clip_length))
    st = max(0, min(valid_len - 1, st))
    ed = max(st + 1, min(valid_len, ed))
    return st, ed


def interval_iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / max(union, 1e-6)


def clamp_window(st, ed, valid_len):
    st = max(0, min(valid_len - 1, int(st)))
    ed = max(st + 1, min(valid_len, int(ed)))
    return st, ed


def generate_negative_windows(pos, valid_len, count, max_iou=0.7):
    pos_st, pos_ed = pos
    pos_len = max(1, pos_ed - pos_st)
    center = (pos_st + pos_ed) / 2.0
    candidates = []

    for shift in [-(pos_len // 2), pos_len // 2, -pos_len, pos_len, -2 * pos_len, 2 * pos_len]:
        st, ed = clamp_window(pos_st + shift, pos_ed + shift, valid_len)
        candidates.append((st, ed))

    for ratio in [0.5, 1.5, 2.0]:
        new_len = max(1, int(round(pos_len * ratio)))
        st = int(round(center - new_len / 2.0))
        ed = st + new_len
        candidates.append(clamp_window(st, ed, valid_len))

    tries = 0
    while len(candidates) < count * 4 and tries < count * 20:
        tries += 1
        neg_len = max(1, int(round(pos_len * random.uniform(0.5, 2.0))))
        if neg_len >= valid_len:
            neg_len = max(1, valid_len // 2)
        st = random.randint(0, max(0, valid_len - neg_len))
        candidates.append((st, st + neg_len))

    negatives = []
    seen = set()
    for cand in candidates:
        if cand in seen or cand == pos:
            continue
        seen.add(cand)
        if interval_iou(pos, cand) <= max_iou:
            negatives.append(cand)
        if len(negatives) >= count:
            break

    while len(negatives) < count:
        fallback = (0, min(valid_len, max(1, pos_len)))
        if fallback == pos and valid_len > pos_len:
            fallback = (valid_len - pos_len, valid_len)
        negatives.append(fallback)
    return negatives[:count]


class MomentContrastiveDataset(Dataset):
    def __init__(self, data_path, a_feat_dir, t_feat_dir, clip_length=1.0, max_a_l=300):
        self.rows = read_jsonl(data_path)
        self.a_feat_dir = a_feat_dir
        self.t_feat_dir = t_feat_dir
        self.clip_length = clip_length
        self.max_a_l = max_a_l

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio_path = os.path.join(self.a_feat_dir, f"{row['vid']}.npz")
        text_path = os.path.join(self.t_feat_dir, f"qid{row['qid']}.npz")
        audio = np.load(audio_path)["features"][: self.max_a_l].astype(np.float32)
        text = np.load(text_path)["last_hidden_state"].astype(np.float32)
        audio = l2_normalize(audio)
        text = l2_normalize(text)
        query = text.mean(axis=0).astype(np.float32)
        valid_len = int(audio.shape[0])
        pos = safe_window_indices(row["relevant_windows"][0], self.clip_length, valid_len)
        return {
            "qid": row["qid"],
            "vid": row["vid"],
            "audio": torch.from_numpy(audio),
            "query": torch.from_numpy(query),
            "pos": pos,
            "valid_len": valid_len,
        }


def make_collate_fn(num_negatives, max_neg_iou):
    def collate(batch):
        batch_size = len(batch)
        max_len = max(item["valid_len"] for item in batch)
        feat_dim = int(batch[0]["audio"].shape[1])
        audio = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        query = torch.stack([item["query"] for item in batch], dim=0)
        windows = []
        meta = []

        for idx, item in enumerate(batch):
            valid_len = item["valid_len"]
            audio[idx, :valid_len] = item["audio"]
            mask[idx, :valid_len] = 1
            pos = item["pos"]
            negs = generate_negative_windows(
                pos=pos,
                valid_len=valid_len,
                count=num_negatives,
                max_iou=max_neg_iou,
            )
            windows.append([pos] + negs)
            meta.append({"qid": item["qid"], "vid": item["vid"], "pos": pos})

        return {
            "audio": audio,
            "audio_mask": mask,
            "query": query,
            "windows": torch.tensor(windows, dtype=torch.long),
            "meta": meta,
        }

    return collate


class TemporalContrastiveAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=768, dropout=0.1, temporal_layers=2):
        super().__init__()
        self.audio_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(temporal_layers)
            ]
        )
        self.audio_out = nn.Linear(hidden_dim, output_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_audio(self, audio, audio_mask=None):
        hidden = self.audio_in(audio)
        x = hidden.transpose(1, 2)
        for layer in self.temporal_layers:
            x = x + layer(x)
        hidden = x.transpose(1, 2)
        encoded = self.audio_out(hidden)
        if audio_mask is not None:
            encoded = encoded * audio_mask.unsqueeze(-1).float()
        return F.normalize(encoded, dim=-1)

    def encode_text(self, query):
        return F.normalize(self.text_proj(query), dim=-1)


def pool_windows(audio_encoded, windows):
    pooled = []
    for batch_idx in range(audio_encoded.shape[0]):
        item_pooled = []
        for st, ed in windows[batch_idx].tolist():
            item_pooled.append(audio_encoded[batch_idx, st:ed].mean(dim=0))
        pooled.append(torch.stack(item_pooled, dim=0))
    return F.normalize(torch.stack(pooled, dim=0), dim=-1)


def compute_loss(model, batch, device, temperature, lambda_batch):
    audio = batch["audio"].to(device)
    audio_mask = batch["audio_mask"].to(device)
    query = batch["query"].to(device)
    windows = batch["windows"].to(device)

    audio_encoded = model.encode_audio(audio, audio_mask)
    query_encoded = model.encode_text(query)
    moment_encoded = pool_windows(audio_encoded, windows)

    local_scores = torch.einsum("bd,bnd->bn", query_encoded, moment_encoded) / temperature
    labels = torch.zeros(local_scores.shape[0], dtype=torch.long, device=device)
    local_loss = F.cross_entropy(local_scores, labels)

    pos_moments = moment_encoded[:, 0, :]
    batch_scores = query_encoded @ pos_moments.t() / temperature
    batch_labels = torch.arange(batch_scores.shape[0], dtype=torch.long, device=device)
    batch_loss = 0.5 * (
        F.cross_entropy(batch_scores, batch_labels)
        + F.cross_entropy(batch_scores.t(), batch_labels)
    )

    loss = local_loss + lambda_batch * batch_loss
    with torch.no_grad():
        local_acc = (local_scores.argmax(dim=1) == 0).float().mean()
        pos_score = local_scores[:, 0].mean()
        neg_score = local_scores[:, 1:].mean()
        margin = pos_score - neg_score
    return loss, {
        "loss": float(loss.detach().cpu()),
        "local_loss": float(local_loss.detach().cpu()),
        "batch_loss": float(batch_loss.detach().cpu()),
        "local_acc": float(local_acc.detach().cpu()),
        "pos_score": float(pos_score.detach().cpu()),
        "neg_score": float(neg_score.detach().cpu()),
        "margin": float(margin.detach().cpu()),
    }


def run_epoch(model, loader, device, optimizer, args, train):
    model.train(train)
    totals = defaultdict(float)
    steps = 0
    for step, batch in enumerate(loader):
        if train:
            optimizer.zero_grad()
        loss, metrics = compute_loss(
            model=model,
            batch=batch,
            device=device,
            temperature=args.temperature,
            lambda_batch=args.lambda_batch,
        )
        if train:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        for key, value in metrics.items():
            totals[key] += value
        steps += 1
        if train and step % args.log_interval == 0:
            print(
                f"[train step {step}] loss={metrics['loss']:.4f} "
                f"local_acc={metrics['local_acc']:.3f} margin={metrics['margin']:.3f}"
            )
    return {key: value / max(steps, 1) for key, value in totals.items()}


def build_loader(data_path, args, shuffle):
    dataset = MomentContrastiveDataset(
        data_path=data_path,
        a_feat_dir=args.a_feat_dir,
        t_feat_dir=args.t_feat_dir,
        clip_length=args.clip_length,
        max_a_l=args.max_a_l,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(args.num_negatives, args.max_neg_iou),
    )


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


@torch.no_grad()
def export_features(model, args, device):
    if not args.export_data_paths:
        return
    rows = []
    for data_path in args.export_data_paths:
        rows.extend(read_jsonl(data_path))

    by_vid = {row["vid"]: row for row in rows}
    by_qid = {row["qid"]: row for row in rows}
    os.makedirs(args.audio_output_dir, exist_ok=True)
    os.makedirs(args.text_output_dir, exist_ok=True)

    model.eval()
    for idx, vid in enumerate(sorted(by_vid)):
        audio_path = os.path.join(args.a_feat_dir, f"{vid}.npz")
        audio = np.load(audio_path)["features"][: args.max_a_l].astype(np.float32)
        audio = l2_normalize(audio)
        tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
        mask = torch.ones(1, tensor.shape[1], dtype=torch.long, device=device)
        encoded = model.encode_audio(tensor, mask)[0].detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(os.path.join(args.audio_output_dir, f"{vid}.npz"), features=encoded)
        if idx % 100 == 0:
            print(f"[export audio] {idx}/{len(by_vid)}")

    for idx, qid in enumerate(sorted(by_qid)):
        text_path = os.path.join(args.t_feat_dir, f"qid{qid}.npz")
        text = np.load(text_path)["last_hidden_state"].astype(np.float32)
        text = l2_normalize(text)
        query = torch.from_numpy(text.mean(axis=0).astype(np.float32)).unsqueeze(0).to(device)
        encoded = model.encode_text(query)[0].detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(
            os.path.join(args.text_output_dir, f"qid{qid}.npz"),
            last_hidden_state=encoded[None, :],
        )
        if idx % 500 == 0:
            print(f"[export text] {idx}/{len(by_qid)}")

    save_json(
        {
            "export_data_paths": args.export_data_paths,
            "audio_output_dir": args.audio_output_dir,
            "text_output_dir": args.text_output_dir,
            "unique_audio": len(by_vid),
            "unique_text": len(by_qid),
            "output_dim": args.output_dim,
        },
        os.path.join(args.output_dir, "export_summary.json"),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", default="data/castella_train_release.jsonl")
    parser.add_argument("--val_data_path", default="data/castella_val_release.jsonl")
    parser.add_argument("--a_feat_dir", default="features/castella/clap")
    parser.add_argument("--t_feat_dir", default="features/castella/clap_text")
    parser.add_argument("--input_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--output_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temporal_layers", type=int, default=2)
    parser.add_argument("--clip_length", type=float, default=1.0)
    parser.add_argument("--max_a_l", type=int, default=300)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--max_neg_iou", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_batch", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_dir", default="results/temporal_contrastive_adapter_v1")
    parser.add_argument("--checkpoint_path", default="results/temporal_contrastive_adapter_v1/best.pt")
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--export_data_paths", nargs="*", default=[])
    parser.add_argument("--audio_output_dir", default="features/castella/tclap_inspired")
    parser.add_argument("--text_output_dir", default="features/castella/tclap_inspired_text")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("train:", args.train_data_path)
    print("val:", args.val_data_path)
    print("output_dir:", args.output_dir)

    model = TemporalContrastiveAdapter(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        temporal_layers=args.temporal_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = build_loader(args.train_data_path, args, shuffle=True)
    val_loader = build_loader(args.val_data_path, args, shuffle=False)

    best_acc = -1.0
    history = []
    for epoch in range(args.epochs):
        print(f"\n========== EPOCH {epoch + 1}/{args.epochs} ==========")
        random.seed(args.seed + epoch)
        train_metrics = run_epoch(model, train_loader, device, optimizer, args, train=True)
        val_metrics = run_epoch(model, val_loader, device, optimizer, args, train=False)
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print("train metrics:", json.dumps(train_metrics, indent=2))
        print("val metrics:", json.dumps(val_metrics, indent=2))
        save_json(record, os.path.join(args.output_dir, f"metrics_epoch{epoch + 1}.json"))
        if val_metrics["local_acc"] > best_acc:
            best_acc = val_metrics["local_acc"]
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_local_acc": best_acc,
                    "val_metrics": val_metrics,
                },
                args.checkpoint_path,
            )
            print(f"saved best checkpoint to {args.checkpoint_path} (local_acc={best_acc:.4f})")

    save_json({"history": history, "best_local_acc": best_acc}, os.path.join(args.output_dir, "summary.json"))
    load_checkpoint(model, args.checkpoint_path, device)
    export_features(model, args, device)


if __name__ == "__main__":
    main()

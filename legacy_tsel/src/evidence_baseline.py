import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import iou_1d


class EvidenceBaseline(nn.Module):
    """Query-guided temporal evidence baseline.

    The model predicts a dense evidence curve S(t, q) plus start/end boundary
    logits. It is intentionally compact so the first experiments focus on the
    research signal rather than another large fusion system.
    """

    def __init__(
        self,
        audio_in_dim=770,
        query_in_dim=768,
        hidden_dim=256,
        dropout=0.1,
        temporal_kernel=3,
        temporal_layers=2,
    ):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(query_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        layers = []
        padding = temporal_kernel // 2
        for _ in range(temporal_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=temporal_kernel, padding=padding),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
        self.temporal_layers = nn.ModuleList(layers)

        self.evidence_head = nn.Linear(hidden_dim, 1)
        self.start_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def masked_mean(x, mask):
        if mask is None:
            return x.mean(dim=1)
        mask = mask.float().unsqueeze(-1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(self, audio_feat, query_feat, audio_mask=None, query_mask=None):
        query = self.masked_mean(query_feat, query_mask)
        query = self.query_proj(query)

        audio = self.audio_proj(audio_feat)
        query_expanded = query.unsqueeze(1).expand_as(audio)
        fused = torch.cat(
            [
                audio,
                query_expanded,
                audio * query_expanded,
                torch.abs(audio - query_expanded),
            ],
            dim=-1,
        )
        hidden = self.fuse(fused)

        x = hidden.transpose(1, 2)
        for layer in self.temporal_layers:
            x = x + layer(x)
        hidden = x.transpose(1, 2)

        evidence_logits = self.evidence_head(hidden).squeeze(-1)
        start_logits = self.start_head(hidden).squeeze(-1)
        end_logits = self.end_head(hidden).squeeze(-1)

        if audio_mask is not None:
            invalid = audio_mask == 0
            evidence_logits = evidence_logits.masked_fill(invalid, -1e4)
            start_logits = start_logits.masked_fill(invalid, -1e4)
            end_logits = end_logits.masked_fill(invalid, -1e4)

        return {
            "evidence_logits": evidence_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "hidden": hidden,
        }


def _clip_center(index, clip_length):
    return (float(index) + 0.5) * float(clip_length)


def build_soft_evidence_labels(
    relevant_windows,
    audio_length,
    clip_length=1.0,
    boundary_sigma=2.0,
    outside_decay=0.25,
):
    labels = torch.zeros(audio_length, dtype=torch.float32)
    for idx in range(audio_length):
        center = _clip_center(idx, clip_length)
        value = 0.0
        for st, ed in relevant_windows:
            if st <= center <= ed:
                value = 1.0
            elif boundary_sigma > 0:
                distance = min(abs(center - st), abs(center - ed))
                value = max(value, outside_decay * math.exp(-(distance ** 2) / (2 * boundary_sigma ** 2)))
        labels[idx] = max(labels[idx].item(), value)
    return labels


def build_boundary_distribution(
    relevant_windows,
    audio_length,
    clip_length=1.0,
    kind="start",
    sigma=1.5,
):
    values = torch.zeros(audio_length, dtype=torch.float32)
    for st, ed in relevant_windows:
        if kind == "start":
            center_idx = int(st // clip_length)
        else:
            center_idx = int(math.ceil(ed / clip_length) - 1)
        center_idx = max(0, min(audio_length - 1, center_idx))
        if sigma <= 0:
            values[center_idx] = max(values[center_idx].item(), 1.0)
            continue
        positions = torch.arange(audio_length, dtype=torch.float32)
        values = torch.maximum(values, torch.exp(-((positions - center_idx) ** 2) / (2 * sigma ** 2)))
    total = values.sum().clamp(min=1e-6)
    return values / total


def build_evidence_targets(
    batch_meta,
    audio_length,
    clip_length,
    device,
    evidence_sigma=2.0,
    boundary_sigma=1.5,
):
    evidence = []
    start = []
    end = []
    for meta in batch_meta:
        windows = meta["relevant_windows"]
        evidence.append(
            build_soft_evidence_labels(
                relevant_windows=windows,
                audio_length=audio_length,
                clip_length=clip_length,
                boundary_sigma=evidence_sigma,
            )
        )
        start.append(
            build_boundary_distribution(
                relevant_windows=windows,
                audio_length=audio_length,
                clip_length=clip_length,
                kind="start",
                sigma=boundary_sigma,
            )
        )
        end.append(
            build_boundary_distribution(
                relevant_windows=windows,
                audio_length=audio_length,
                clip_length=clip_length,
                kind="end",
                sigma=boundary_sigma,
            )
        )
    return {
        "evidence": torch.stack(evidence).to(device),
        "start": torch.stack(start).to(device),
        "end": torch.stack(end).to(device),
    }


def masked_bce_with_logits(logits, targets, mask, pos_weight=None):
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    valid = mask.float()
    return (loss * valid).sum() / valid.sum().clamp(min=1.0)


def masked_soft_ce(logits, targets, mask):
    masked_logits = logits.masked_fill(mask == 0, -1e4)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    valid_targets = targets * mask.float()
    valid_targets = valid_targets / valid_targets.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return -(valid_targets * log_probs).sum(dim=-1).mean()


def evidence_loss(outputs, targets, audio_mask, lambda_boundary=0.5, use_pos_weight=True):
    pos_weight = None
    if use_pos_weight:
        binary_pos = (targets["evidence"] > 0.5) & audio_mask.bool()
        num_pos = binary_pos.float().sum()
        num_valid = audio_mask.float().sum()
        num_neg = (num_valid - num_pos).clamp(min=1.0)
        pos_weight = (num_neg / num_pos.clamp(min=1.0)).detach()

    evidence = masked_bce_with_logits(
        outputs["evidence_logits"],
        targets["evidence"],
        audio_mask,
        pos_weight=pos_weight,
    )
    start = masked_soft_ce(outputs["start_logits"], targets["start"], audio_mask)
    end = masked_soft_ce(outputs["end_logits"], targets["end"], audio_mask)
    boundary = 0.5 * (start + end)
    total = evidence + lambda_boundary * boundary
    return total, {
        "loss": total.detach(),
        "evidence_loss": evidence.detach(),
        "boundary_loss": boundary.detach(),
        "start_loss": start.detach(),
        "end_loss": end.detach(),
    }


def _window_to_index_span(window, valid_len, clip_length):
    st = max(0, int(math.floor(float(window[0]) / float(clip_length))))
    ed = min(valid_len, int(math.ceil(float(window[1]) / float(clip_length))))
    if ed <= st:
        return None
    return st, ed


def _window_to_boundary_indices(window, valid_len, clip_length):
    start_idx = max(0, min(valid_len - 1, int(float(window[0]) // float(clip_length))))
    end_idx = max(0, min(valid_len - 1, int(math.ceil(float(window[1]) / float(clip_length)) - 1)))
    return start_idx, end_idx


def _best_gt_window(window, gt_windows):
    best = None
    best_iou = -1.0
    for gt in gt_windows:
        overlap = max(0.0, min(float(window[1]), float(gt[1])) - max(float(window[0]), float(gt[0])))
        union = (float(window[1]) - float(window[0])) + (float(gt[1]) - float(gt[0])) - overlap
        iou = overlap / max(union, 1e-6)
        if iou > best_iou:
            best = gt
            best_iou = iou
    return best


def hard_negative_ranking_loss(
    evidence_logits,
    batch_meta,
    hard_negative_map,
    audio_mask,
    clip_length=1.0,
    margin=0.1,
    topk=3,
):
    """Encourage GT evidence to outrank mined high-scoring wrong windows."""
    probs = torch.sigmoid(evidence_logits)
    terms = []
    gt_scores = []
    hn_scores = []

    for batch_idx, meta in enumerate(batch_meta):
        candidates = hard_negative_map.get(meta["qid"], [])[:topk]
        if not candidates:
            continue

        valid_len = int(audio_mask[batch_idx].sum().item())
        positive_scores = []
        for window in meta.get("relevant_windows", []):
            span = _window_to_index_span(window, valid_len, clip_length)
            if span is None:
                continue
            st, ed = span
            positive_scores.append(probs[batch_idx, st:ed].mean())
        if not positive_scores:
            continue

        gt_score = torch.stack(positive_scores).max()
        for candidate in candidates:
            window = candidate.get("window", candidate)
            span = _window_to_index_span(window, valid_len, clip_length)
            if span is None:
                continue
            st, ed = span
            hn_score = probs[batch_idx, st:ed].mean()
            terms.append(F.relu(margin - gt_score + hn_score))
            gt_scores.append(gt_score.detach())
            hn_scores.append(hn_score.detach())

    if not terms:
        zero = evidence_logits.sum() * 0.0
        return zero, {"hard_negative_pairs": 0, "hard_negative_gap": 0.0}

    gt_mean = torch.stack(gt_scores).mean()
    hn_mean = torch.stack(hn_scores).mean()
    return torch.stack(terms).mean(), {
        "hard_negative_pairs": len(terms),
        "hard_negative_gap": float((gt_mean - hn_mean).item()),
    }


def type_aware_hard_negative_loss(
    outputs,
    batch_meta,
    hard_negative_map,
    audio_mask,
    clip_length=1.0,
    semantic_margin=0.1,
    boundary_margin=0.2,
    semantic_topk=3,
    boundary_topk=3,
    lambda_semantic=0.15,
    lambda_boundary=0.1,
):
    """Apply different pressure to semantic false peaks and boundary distractors."""
    evidence_probs = torch.sigmoid(outputs["evidence_logits"])
    start_logits = outputs["start_logits"]
    end_logits = outputs["end_logits"]
    semantic_terms = []
    semantic_gt_scores = []
    semantic_hn_scores = []
    boundary_terms = []
    boundary_pos_scores = []
    boundary_neg_scores = []
    type_pair_counts = {
        "semantic_false_peak_pairs": 0,
        "boundary_distractor_pairs": 0,
        "over_wide_pairs": 0,
        "under_wide_pairs": 0,
    }

    for batch_idx, meta in enumerate(batch_meta):
        typed_candidates = hard_negative_map.get(meta["qid"], {})
        if not isinstance(typed_candidates, dict):
            continue

        valid_len = int(audio_mask[batch_idx].sum().item())
        positive_scores = []
        for window in meta.get("relevant_windows", []):
            span = _window_to_index_span(window, valid_len, clip_length)
            if span is None:
                continue
            st, ed = span
            positive_scores.append(evidence_probs[batch_idx, st:ed].mean())

        if positive_scores and lambda_semantic > 0:
            gt_score = torch.stack(positive_scores).max()
            for candidate in typed_candidates.get("semantic_false_peak", [])[:semantic_topk]:
                span = _window_to_index_span(candidate.get("window", candidate), valid_len, clip_length)
                if span is None:
                    continue
                st, ed = span
                hn_score = evidence_probs[batch_idx, st:ed].mean()
                semantic_terms.append(F.relu(semantic_margin - gt_score + hn_score))
                semantic_gt_scores.append(gt_score.detach())
                semantic_hn_scores.append(hn_score.detach())
                type_pair_counts["semantic_false_peak_pairs"] += 1

        if lambda_boundary <= 0:
            continue

        for type_name in ["boundary_distractor", "over_wide", "under_wide"]:
            for candidate in typed_candidates.get(type_name, [])[:boundary_topk]:
                window = candidate.get("window", candidate)
                matched_gt = candidate.get("matched_gt") or _best_gt_window(window, meta.get("relevant_windows", []))
                if not matched_gt:
                    continue
                cand_start, cand_end = _window_to_boundary_indices(window, valid_len, clip_length)
                gt_start, gt_end = _window_to_boundary_indices(matched_gt, valid_len, clip_length)

                local_terms = []
                if cand_start != gt_start:
                    local_terms.append(
                        F.relu(boundary_margin - start_logits[batch_idx, gt_start] + start_logits[batch_idx, cand_start])
                    )
                    boundary_pos_scores.append(start_logits[batch_idx, gt_start].detach())
                    boundary_neg_scores.append(start_logits[batch_idx, cand_start].detach())
                if cand_end != gt_end:
                    local_terms.append(
                        F.relu(boundary_margin - end_logits[batch_idx, gt_end] + end_logits[batch_idx, cand_end])
                    )
                    boundary_pos_scores.append(end_logits[batch_idx, gt_end].detach())
                    boundary_neg_scores.append(end_logits[batch_idx, cand_end].detach())
                if local_terms:
                    boundary_terms.append(torch.stack(local_terms).mean())
                    type_pair_counts[f"{type_name}_pairs"] += 1

    zero = outputs["evidence_logits"].sum() * 0.0
    semantic_loss = torch.stack(semantic_terms).mean() if semantic_terms else zero
    boundary_loss = torch.stack(boundary_terms).mean() if boundary_terms else zero
    total = lambda_semantic * semantic_loss + lambda_boundary * boundary_loss

    if semantic_gt_scores:
        semantic_gap = float((torch.stack(semantic_gt_scores).mean() - torch.stack(semantic_hn_scores).mean()).item())
    else:
        semantic_gap = 0.0
    if boundary_pos_scores:
        boundary_gap = float((torch.stack(boundary_pos_scores).mean() - torch.stack(boundary_neg_scores).mean()).item())
    else:
        boundary_gap = 0.0

    return total, {
        "typed_hard_negative_loss": total.detach(),
        "semantic_hard_negative_loss": semantic_loss.detach(),
        "boundary_hard_negative_loss": boundary_loss.detach(),
        "semantic_hard_negative_gap": semantic_gap,
        "boundary_hard_negative_gap": boundary_gap,
        **type_pair_counts,
    }


def masked_log_softmax(logits, mask):
    return F.log_softmax(logits.masked_fill(mask == 0, -1e4), dim=-1)


def decode_evidence_windows(
    evidence_logits,
    start_logits,
    end_logits,
    valid_len,
    clip_length=1.0,
    topn=10,
    min_len=1,
    max_len=150,
    evidence_weight=1.0,
    nms_threshold=0.7,
):
    evidence = torch.sigmoid(evidence_logits[:valid_len])
    start_logp = F.log_softmax(start_logits[:valid_len], dim=-1)
    end_logp = F.log_softmax(end_logits[:valid_len], dim=-1)

    csum = torch.cat([torch.zeros(1, device=evidence.device), evidence.cumsum(dim=0)])
    starts = torch.arange(valid_len, device=evidence.device).view(-1, 1)
    ends = torch.arange(valid_len, device=evidence.device).view(1, -1)
    lengths = ends - starts + 1
    valid = lengths >= max(min_len, 1)
    if max_len > 0:
        valid = valid & (lengths <= max_len)

    segment_sum = csum[ends + 1] - csum[starts]
    mean_ev = segment_sum / lengths.clamp(min=1).float()
    score_matrix = start_logp.view(-1, 1) + end_logp.view(1, -1) + evidence_weight * mean_ev
    score_matrix = score_matrix.masked_fill(~valid, -1e9)

    flat_scores = score_matrix.reshape(-1)
    pre_nms = min(flat_scores.numel(), max(topn * 30, topn))
    values, indices = torch.topk(flat_scores, k=pre_nms)
    candidates = []
    for value, flat_idx in zip(values.tolist(), indices.tolist()):
        if value <= -1e8:
            continue
        st = flat_idx // valid_len
        ed = flat_idx % valid_len
        candidates.append([st, ed, value])

    kept = []
    for st, ed, score in candidates:
        window = [st * clip_length, (ed + 1) * clip_length, score]
        if all(iou_1d(window[0], window[1], old[0], old[1]) < nms_threshold for old in kept):
            kept.append(window)
        if len(kept) >= topn:
            break
    return kept


def evidence_curve_stats(evidence_logits, targets, audio_mask):
    probs = torch.sigmoid(evidence_logits)
    valid = audio_mask.bool()
    pos = (targets["evidence"] > 0.5) & valid
    neg = (targets["evidence"] < 0.05) & valid
    stats = {}
    stats["evidence_pos_mean"] = probs[pos].mean().item() if pos.any() else 0.0
    stats["evidence_neg_mean"] = probs[neg].mean().item() if neg.any() else 0.0
    stats["evidence_gap"] = stats["evidence_pos_mean"] - stats["evidence_neg_mean"]
    return stats

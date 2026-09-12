"""Recovered R14 evidence-head training with artifact persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

import r14_minimal_trainable_egcg as recovered_r14


LearnableEvidenceHead = recovered_r14.LearnableEvidenceHead
FrozenQDWithEvidence = recovered_r14.FrozenQDWithEvidence
evidence_targets = recovered_r14.evidence_targets
evidence_loss = recovered_r14.evidence_loss


def parameter_count(module: nn.Module) -> int:
    """Return the number of parameters in a module."""
    return sum(parameter.numel() for parameter in module.parameters())


def train_evidence_head(
    model: FrozenQDWithEvidence,
    criterion: nn.Module,
    dataset: Any,
    opt: Any,
    lambda_evidence: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    checkpoint_path: Path,
    optimizer_path: Path,
    log_path: Path,
) -> tuple[List[Dict[str, Any]], torch.optim.Optimizer, Dict[str, float]]:
    """Train the recovered R14 evidence head and save model/optimizer artifacts."""
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=recovered_r14.start_end_collate,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.evidence_head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: List[Dict[str, Any]] = []
    model.qd_model.eval()
    model.evidence_head.train()
    started_total = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(1, epochs + 1):
            totals = {"loss_total": 0.0, "loss_qd": 0.0, "loss_evidence": 0.0, "batches": 0}
            started = time.perf_counter()
            for batch in loader:
                metas = batch[0]
                model_inputs, targets = recovered_r14.prepare_batch_inputs(batch[1], opt.device)
                outputs, logits = model(model_inputs)
                labels = evidence_targets(metas, model_inputs["src_aud_mask"])
                with torch.no_grad():
                    loss_qd_value = recovered_r14.qd_loss(criterion, outputs, targets)
                loss_ev = evidence_loss(logits, labels, model_inputs["src_aud_mask"])
                loss_total = loss_qd_value.detach() + lambda_evidence * loss_ev
                optimizer.zero_grad(set_to_none=True)
                if lambda_evidence > 0:
                    (lambda_evidence * loss_ev).backward()
                    nn.utils.clip_grad_norm_(model.evidence_head.parameters(), grad_clip)
                    optimizer.step()
                totals["loss_total"] += float(loss_total.detach().cpu())
                totals["loss_qd"] += float(loss_qd_value.detach().cpu())
                totals["loss_evidence"] += float(loss_ev.detach().cpu())
                totals["batches"] += 1
            batches = max(1, totals["batches"])
            row = {
                "epoch": epoch,
                "lambda_evidence": lambda_evidence,
                "loss_total": totals["loss_total"] / batches,
                "loss_qd": totals["loss_qd"] / batches,
                "loss_evidence": totals["loss_evidence"] / batches,
                "seconds": time.perf_counter() - started,
            }
            history.append(row)
            log_handle.write(json.dumps(row) + "\n")
            log_handle.flush()

    elapsed = {"seconds": time.perf_counter() - started_total}
    torch.save(
        {
            "state_dict": model.evidence_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "configuration": {
                "lambda_evidence": lambda_evidence,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
            },
        },
        checkpoint_path,
    )
    torch.save(optimizer.state_dict(), optimizer_path)
    return history, optimizer, elapsed

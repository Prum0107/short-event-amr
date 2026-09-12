"""Recovered R17 ranker, pairwise hinge objective, and artifact persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

import r17_trainable_evidence_ranker as recovered_r17


EvidenceRanker = recovered_r17.EvidenceRanker
FeatureCache = recovered_r17.FeatureCache
examples_from_records = recovered_r17.examples_from_records
pad_batch = recovered_r17.pad_batch
pairwise_loss = recovered_r17.pairwise_loss
rank_examples = recovered_r17.rank_examples
parameter_count = lambda module: sum(parameter.numel() for parameter in module.parameters())


def train_ranker(
    model: EvidenceRanker,
    examples: List[Dict[str, Any]],
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    margin: float,
    checkpoint_path: Path,
    optimizer_path: Path,
    log_path: Path,
) -> tuple[List[Dict[str, Any]], torch.optim.Optimizer, Dict[str, float]]:
    """Train the recovered R17 ranker with the exact all-pairs hinge construction."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: List[Dict[str, Any]] = []
    started_total = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_values: List[float] = []
            valid_queries = 0
            pairs = 0
            started = time.perf_counter()
            for batch in recovered_r17.batch_iter(examples, batch_size, shuffle=True):
                audio, query, scalar, ious = pad_batch(batch, device)
                scores = model(audio, query, scalar)
                loss, valid, count = pairwise_loss(scores, ious)
                if loss is None:
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                loss_values.append(float(loss.detach().cpu()))
                valid_queries += valid
                pairs += count
            row = {
                "epoch": epoch,
                "pairwise_loss": float(np.mean(loss_values)) if loss_values else None,
                "valid_queries": valid_queries,
                "pairs": pairs,
                "seconds": time.perf_counter() - started,
            }
            history.append(row)
            log_handle.write(json.dumps(row) + "\n")
            log_handle.flush()
    elapsed = {"seconds": time.perf_counter() - started_total}
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "configuration": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "grad_clip": grad_clip,
                "margin": margin,
            },
        },
        checkpoint_path,
    )
    torch.save(optimizer.state_dict(), optimizer_path)
    return history, optimizer, elapsed

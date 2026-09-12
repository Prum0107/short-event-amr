"""End-to-end EGCG inference pipeline assembled from recovered components."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

import r14_minimal_trainable_egcg as recovered_r14
import r15_trainable_egcg as recovered_r15
import r17_trainable_evidence_ranker as recovered_r17


class EGCGInferencePipeline:
    """Run frozen QD-DETR, recovered evidence proposals, fusion, and ranker."""

    def __init__(self, model: Any, criterion: Any, opt: Any) -> None:
        self.model = model
        self.criterion = criterion
        self.opt = opt

    def collect_and_enrich(self, dataset: Any) -> List[Dict[str, Any]]:
        """Collect QD/evidence outputs and build the recovered candidate pools."""
        records = recovered_r14.collect_records(self.model, dataset, self.opt)
        return recovered_r15.enrich(records)

    @staticmethod
    def attach_ranked(
        records: Sequence[Mapping[str, Any]],
        ranked: Sequence[Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Attach ranker scores to records without changing candidate values."""
        result: List[Dict[str, Any]] = []
        for record, candidates in zip(records, ranked):
            current = dict(record)
            current["e2_ranked"] = list(candidates)
            current["e1_egcg"] = list(current["e3_full_egcg"])
            current["e3_ranker_only"] = sorted(
                [candidate for candidate in candidates if candidate["source"] == "qd"],
                key=lambda candidate: (-candidate["ranker_score"], candidate["start"], candidate["end"]),
            )
            result.append(current)
        return result

    @staticmethod
    def system_metrics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Compute the historical E0–E4 system metrics on supplied records."""
        systems = {
            "E0_QD": ("canonical QD-DETR", "e0_qd", "qd_score"),
            "E1_evidence_head_only": ("evidence head only", "e1_head_only", "qd_score"),
            "E2_evidence_candidate_generation": ("QD plus evidence proposals", "e2_qd_plus_evidence", "qd_score"),
            "E3_evidence_ranker_only": ("ranker on QD candidates", "e3_ranker_only", "ranker_score"),
            "E4_full_EGCG": ("full EGCG", "e2_ranked", "ranker_score"),
        }
        return {
            name: recovered_r17.system_metrics(records, description, key, score_key)
            for name, (description, key, score_key) in systems.items()
        }

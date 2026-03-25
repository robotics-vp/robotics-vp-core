"""Append-only store for fill-outcome records.

Each record captures a coverage-loop decision and its measured impact:

    gap edge → fill method chosen → Δcoverage + quality signal

These triples are the training data for the learned gap ranker (Phase 2)
and learned fill-path policy (Phase 3).

Storage format: one JSON object per line (.jsonl), append-only.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass
class FillOutcomeRecord:
    """Single fill-outcome observation."""

    edge_key: str                  # "task:open_drawer -> skill:grasp_handle"
    fill_method: str               # "diffusion" | "real_sim" | "synthetic_branch"
    gap_features: Dict[str, Any]   # edge feature snapshot at decision time
    pre_evidence_count: int        # evidence before fill attempt
    post_evidence_count: int       # evidence after fill attempt
    coverage_delta: float          # Δ coverage_ratio after fill
    wall_time_s: float             # time to generate the fill data
    quality_score: float           # downstream quality (ΔMPL, trust_net score, etc.)
    timestamp: str = ""            # ISO-8601

    # Computed reward used as training target
    @property
    def marginal_value(self) -> float:
        """Value per unit cost: coverage improvement × quality / time."""
        cost = max(self.wall_time_s, 0.1)
        return (self.coverage_delta * self.quality_score) / cost

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["marginal_value"] = self.marginal_value
        return d

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FillOutcomeRecord":
        return cls(
            edge_key=str(payload.get("edge_key", "")),
            fill_method=str(payload.get("fill_method", "")),
            gap_features=dict(payload.get("gap_features", {})),
            pre_evidence_count=int(payload.get("pre_evidence_count", 0)),
            post_evidence_count=int(payload.get("post_evidence_count", 0)),
            coverage_delta=float(payload.get("coverage_delta", 0.0)),
            wall_time_s=float(payload.get("wall_time_s", 0.0)),
            quality_score=float(payload.get("quality_score", 0.0)),
            timestamp=str(payload.get("timestamp", "")),
        )


class FillOutcomeStore:
    """Append-only JSONL store for fill-outcome records.

    Usage::

        store = FillOutcomeStore("data/fill_outcomes.jsonl")
        store.append(record)
        all_records = store.load_all()
    """

    def __init__(self, path: str = "data/fill_outcomes.jsonl") -> None:
        self.path = Path(path)

    def append(self, record: FillOutcomeRecord) -> None:
        """Append a single record."""
        if not record.timestamp:
            record = FillOutcomeRecord(
                **{**asdict(record), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def append_batch(self, records: Sequence[FillOutcomeRecord]) -> None:
        """Append multiple records atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            for record in records:
                if not record.timestamp:
                    record = FillOutcomeRecord(
                        **{**asdict(record), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    )
                f.write(json.dumps(record.to_dict()) + "\n")

    def load_all(self) -> List[FillOutcomeRecord]:
        """Load all records from the store."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(FillOutcomeRecord.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return records

    def load_for_edge(self, edge_key: str) -> List[FillOutcomeRecord]:
        """Load records for a specific edge."""
        return [r for r in self.load_all() if r.edge_key == edge_key]

    def load_for_method(self, fill_method: str) -> List[FillOutcomeRecord]:
        """Load records for a specific fill method."""
        return [r for r in self.load_all() if r.fill_method == fill_method]

    def summary(self) -> Dict[str, Any]:
        """Aggregate summary statistics."""
        records = self.load_all()
        if not records:
            return {
                "total_records": 0,
                "methods": {},
                "avg_coverage_delta": 0.0,
                "avg_quality_score": 0.0,
                "avg_marginal_value": 0.0,
            }

        by_method: Dict[str, List[FillOutcomeRecord]] = {}
        for r in records:
            by_method.setdefault(r.fill_method, []).append(r)

        method_stats = {}
        for method, recs in by_method.items():
            deltas = [r.coverage_delta for r in recs]
            qualities = [r.quality_score for r in recs]
            values = [r.marginal_value for r in recs]
            method_stats[method] = {
                "count": len(recs),
                "avg_coverage_delta": sum(deltas) / len(deltas),
                "avg_quality_score": sum(qualities) / len(qualities),
                "avg_marginal_value": sum(values) / len(values),
            }

        all_deltas = [r.coverage_delta for r in records]
        all_qualities = [r.quality_score for r in records]
        all_values = [r.marginal_value for r in records]

        return {
            "total_records": len(records),
            "methods": method_stats,
            "avg_coverage_delta": sum(all_deltas) / len(all_deltas),
            "avg_quality_score": sum(all_qualities) / len(all_qualities),
            "avg_marginal_value": sum(all_values) / len(all_values),
        }

    def record_count(self) -> int:
        """Count records without loading all."""
        if not self.path.exists():
            return 0
        count = 0
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


__all__ = [
    "FillOutcomeRecord",
    "FillOutcomeStore",
]

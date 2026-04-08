"""Typed benchmark-evidence artifacts for Perception / Grounding promotion.

These artifacts are the persistent promotion inputs for learned perception
subsystems. They replace ad hoc in-memory benchmark dicts with a stable,
serializable contract that can be written alongside annotation exports or
other evaluation artifacts and reloaded later by the compiler.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.utils.config_digest import sha256_file, sha256_json

from .common import clip01, mapping


PERCEPTION_BENCHMARK_EVIDENCE_SCHEMA_VERSION = "perception_benchmark_evidence_v1"


@dataclass(frozen=True)
class PerceptionBenchmarkEvidence:
    """Persistent promotion evidence for a perception subsystem."""

    subsystem_key: str
    benchmark_evidence_present: bool = False
    evidence_source_provisional: bool = True
    evidence_truth_class: str = "heuristic_derived"
    token_source_kind: str = "heuristic_scene_graph"
    source_record_count: int = 0
    annotation_supervision_score: float = 0.0
    held_out_label_agreement: float = 0.0
    downstream_usefulness_score: float = 0.0
    receipt_consistency: float = 0.0
    gate_score: float = 0.0
    promotion_eligible: bool = False
    source_artifact_path: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = PERCEPTION_BENCHMARK_EVIDENCE_SCHEMA_VERSION

    @property
    def evidence_digest(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        effective_eligible = (
            self.promotion_eligible
            and self.benchmark_evidence_present
            and not self.evidence_source_provisional
        )
        return {
            "schema_version": self.schema_version,
            "subsystem_key": self.subsystem_key,
            "benchmark_evidence_present": bool(self.benchmark_evidence_present),
            "evidence_source_provisional": bool(self.evidence_source_provisional),
            "evidence_truth_class": str(self.evidence_truth_class),
            "token_source_kind": str(self.token_source_kind),
            "source_record_count": int(self.source_record_count),
            "annotation_supervision_score": clip01(
                self.annotation_supervision_score
            ),
            "held_out_label_agreement": clip01(self.held_out_label_agreement),
            "downstream_usefulness_score": clip01(
                self.downstream_usefulness_score
            ),
            "receipt_consistency": clip01(self.receipt_consistency),
            "gate_score": clip01(self.gate_score),
            "promotion_eligible": bool(effective_eligible),
            "source_artifact_path": self.source_artifact_path,
            "created_at": self.created_at,
            "metadata": mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "PerceptionBenchmarkEvidence":
        return cls(
            schema_version=str(
                payload.get(
                    "schema_version",
                    PERCEPTION_BENCHMARK_EVIDENCE_SCHEMA_VERSION,
                )
            ),
            subsystem_key=str(payload.get("subsystem_key", "")),
            benchmark_evidence_present=bool(
                payload.get("benchmark_evidence_present", False)
            ),
            evidence_source_provisional=bool(
                payload.get("evidence_source_provisional", True)
            ),
            evidence_truth_class=str(
                payload.get("evidence_truth_class", "heuristic_derived")
            ),
            token_source_kind=str(
                payload.get("token_source_kind", "heuristic_scene_graph")
            ),
            source_record_count=int(payload.get("source_record_count", 0)),
            annotation_supervision_score=float(
                payload.get("annotation_supervision_score", 0.0) or 0.0
            ),
            held_out_label_agreement=float(
                payload.get("held_out_label_agreement", 0.0) or 0.0
            ),
            downstream_usefulness_score=float(
                payload.get("downstream_usefulness_score", 0.0) or 0.0
            ),
            receipt_consistency=float(
                payload.get("receipt_consistency", 0.0) or 0.0
            ),
            gate_score=float(payload.get("gate_score", 0.0) or 0.0),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            source_artifact_path=payload.get("source_artifact_path"),
            created_at=str(payload.get("created_at", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


def build_perception_benchmark_evidence(
    *,
    subsystem_key: str,
    metrics: Mapping[str, Any],
    source_record_count: int = 0,
    source_artifact_path: Optional[str | Path] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> PerceptionBenchmarkEvidence:
    """Build a typed benchmark-evidence artifact from evaluation metrics."""
    payload = dict(metrics or {})
    if "gate_score" in payload:
        gate_score = float(payload.get("gate_score", 0.0) or 0.0)
    else:
        gate_score = (
            0.4 * float(payload.get("annotation_supervision_score", 0.0) or 0.0)
            + 0.3 * float(payload.get("held_out_label_agreement", 0.0) or 0.0)
            + 0.2 * float(payload.get("downstream_usefulness_score", 0.0) or 0.0)
            + 0.1 * float(payload.get("receipt_consistency", 0.0) or 0.0)
        )
    provisional = bool(payload.get("evidence_source_provisional", True))
    benchmark_present = bool(payload.get("benchmark_evidence_present", False))
    promotion_eligible = bool(
        payload.get(
            "promotion_eligible",
            benchmark_present and (not provisional) and gate_score >= 0.6,
        )
    )
    return PerceptionBenchmarkEvidence(
        subsystem_key=subsystem_key,
        benchmark_evidence_present=benchmark_present,
        evidence_source_provisional=provisional,
        evidence_truth_class=str(
            payload.get("evidence_truth_class", "heuristic_derived")
        ),
        token_source_kind=str(
            payload.get("token_source_kind", "heuristic_scene_graph")
        ),
        source_record_count=int(
            payload.get("source_record_count", source_record_count)
        ),
        annotation_supervision_score=float(
            payload.get("annotation_supervision_score", 0.0) or 0.0
        ),
        held_out_label_agreement=float(
            payload.get("held_out_label_agreement", 0.0) or 0.0
        ),
        downstream_usefulness_score=float(
            payload.get("downstream_usefulness_score", 0.0) or 0.0
        ),
        receipt_consistency=float(
            payload.get("receipt_consistency", 0.0) or 0.0
        ),
        gate_score=float(gate_score),
        promotion_eligible=promotion_eligible,
        source_artifact_path=(
            str(Path(source_artifact_path).resolve())
            if source_artifact_path is not None
            else None
        ),
        metadata=dict(metadata or {}),
    )


def write_perception_benchmark_evidence(
    path: str | Path,
    evidence: PerceptionBenchmarkEvidence,
) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sha256_file(output_path)


def load_perception_benchmark_evidence(
    path: str | Path,
) -> PerceptionBenchmarkEvidence:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PerceptionBenchmarkEvidence.from_dict(payload)


def coerce_perception_benchmark_evidence_payload(
    value: Optional[Any],
) -> Dict[str, Any]:
    """Normalize a benchmark-evidence input to a plain mapping.

    Accepts:
    - ``None``
    - ``PerceptionBenchmarkEvidence``
    - a mapping payload
    - a filesystem path to a persisted JSON artifact
    """
    if value is None:
        return {}
    if isinstance(value, PerceptionBenchmarkEvidence):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, Path)):
        evidence = load_perception_benchmark_evidence(value)
        payload = evidence.to_dict()
        payload["metadata"] = {
            **dict(payload.get("metadata", {}) or {}),
            "loaded_from_path": str(Path(value).resolve()),
        }
        return payload
    raise TypeError(
        "Unsupported benchmark evidence value type: "
        f"{type(value)!r}"
    )


__all__ = [
    "PERCEPTION_BENCHMARK_EVIDENCE_SCHEMA_VERSION",
    "PerceptionBenchmarkEvidence",
    "build_perception_benchmark_evidence",
    "coerce_perception_benchmark_evidence_payload",
    "load_perception_benchmark_evidence",
    "write_perception_benchmark_evidence",
]

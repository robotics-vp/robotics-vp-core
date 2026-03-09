"""Belief-state snapshots derived from the evidence bus."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from src.evidence.bus import EvidenceBus
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _strings(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in (values or [])]


@dataclass(frozen=True)
class BeliefState:
    """Aggregated evidence snapshot for routing, planning, and supervision."""

    belief_id: str
    episode_id: str
    timestamp: str
    semantic_tags: list[str]
    state_vector: Dict[str, float] = field(default_factory=dict)
    uncertainty: Dict[str, float] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "belief_state_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "semantic_tags": list(self.semantic_tags),
            "state_vector": _float_mapping(self.state_vector),
            "uncertainty": _float_mapping(self.uncertainty),
            "evidence_refs": list(self.evidence_refs),
            "artifact_refs": _mapping(self.artifact_refs),
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BeliefState":
        return cls(
            belief_id=str(payload.get("belief_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            timestamp=str(payload.get("timestamp", "")),
            semantic_tags=_strings(payload.get("semantic_tags")),
            state_vector=_float_mapping(payload.get("state_vector")),
            uncertainty=_float_mapping(payload.get("uncertainty")),
            evidence_refs=_strings(payload.get("evidence_refs")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "belief_state_v1")),
        )


def belief_state_from_evidence_bus(
    *,
    evidence_bus: EvidenceBus,
    episode_id: str,
    timestamp: str,
    semantic_tags: Optional[Sequence[Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    extra_state: Optional[Mapping[str, Any]] = None,
) -> BeliefState:
    """Aggregate evidence records into a compact planning-facing belief state."""

    records = evidence_bus.for_episode(episode_id)
    summary = evidence_bus.summarize_episode(episode_id)

    metrics: Dict[str, list[float]] = {}
    for record in records:
        for key, value in record.metrics.items():
            metric_key = str(key)
            metrics.setdefault(metric_key, []).append(float(value))

    metric_means = {
        key: sum(values) / float(len(values))
        for key, values in metrics.items()
        if values
    }

    geometry_quality = float(
        metric_means.get("scene_ir_quality")
        or metric_means.get("map_first_quality_score")
        or metric_means.get("semantic_fusion_confidence_mean")
        or summary.get("confidence_mean", 0.0)
    )
    semantic_quality = float(
        metric_means.get("semantic_fusion_confidence_mean")
        or metric_means.get("teacher_confidence_mean")
        or summary.get("confidence_mean", 0.0)
    )
    disagreement_mean = float(summary.get("disagreement_mean", 0.0))
    coverage = float(summary.get("coverage", 0.0))
    teacher_alignment = float(
        metric_means.get("teacher_confidence_mean")
        or metric_means.get("vla_confidence_mean")
        or 0.0
    )

    state_vector = {
        "evidence_confidence_mean": float(summary.get("confidence_mean", 0.0)),
        "evidence_disagreement_mean": disagreement_mean,
        "evidence_coverage": coverage,
        "geometry_quality": geometry_quality,
        "semantic_quality": semantic_quality,
        "teacher_alignment": teacher_alignment,
    }
    for key, value in _float_mapping(extra_state).items():
        state_vector[str(key)] = float(value)

    uncertainty = {
        "epistemic": float(max(0.0, 1.0 - state_vector["evidence_confidence_mean"])),
        "semantic": disagreement_mean,
        "coverage_gap": float(max(0.0, 1.0 - coverage)),
    }

    belief_payload = {
        "episode_id": str(episode_id),
        "timestamp": str(timestamp),
        "semantic_tags": _strings(semantic_tags),
        "state_vector": state_vector,
        "uncertainty": uncertainty,
        "evidence_refs": [record.evidence_id for record in records],
        "artifact_refs": {
            **_mapping(summary.get("artifact_refs")),
            **_mapping(artifact_refs),
        },
        "provenance": _mapping(provenance),
        "metadata": _mapping(metadata),
        "version": "belief_state_v1",
    }
    belief_id = f"belief_{sha256_json(belief_payload)[:16]}"
    return BeliefState(
        belief_id=belief_id,
        episode_id=str(episode_id),
        timestamp=str(timestamp),
        semantic_tags=_strings(semantic_tags),
        state_vector=_float_mapping(state_vector),
        uncertainty=_float_mapping(uncertainty),
        evidence_refs=[record.evidence_id for record in records],
        artifact_refs={
            **_mapping(summary.get("artifact_refs")),
            **_mapping(artifact_refs),
        },
        provenance=_mapping(provenance),
        metadata=_mapping(metadata),
    )


__all__ = ["BeliefState", "belief_state_from_evidence_bus"]

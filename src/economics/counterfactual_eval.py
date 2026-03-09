"""Counterfactual supervision sidecars for routing, collection, and adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

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


@dataclass(frozen=True)
class CounterfactualCandidate:
    """Single counterfactual action/route candidate."""

    candidate_id: str
    label: str
    expected_net_value: float
    deltas: Dict[str, float] = field(default_factory=dict)
    action: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "expected_net_value": float(self.expected_net_value),
            "deltas": _float_mapping(self.deltas),
            "action": _float_mapping(self.action),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterfactualCandidate":
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            label=str(payload.get("label", "")),
            expected_net_value=float(payload.get("expected_net_value", 0.0)),
            deltas=_float_mapping(payload.get("deltas")),
            action=_float_mapping(payload.get("action")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class CounterfactualEval:
    """Evaluation bundle over candidate routes/actions versus a baseline."""

    eval_id: str
    run_id: str
    episode_id: str
    timestamp: str
    runtime_packet_id: Optional[str]
    objective_profile_id: str
    baseline_label: str
    candidates: list[CounterfactualCandidate] = field(default_factory=list)
    recommended_action: str = "noop"
    evidence_refs: Dict[str, Any] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "counterfactual_eval_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "runtime_packet_id": self.runtime_packet_id,
            "objective_profile_id": self.objective_profile_id,
            "baseline_label": self.baseline_label,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "recommended_action": self.recommended_action,
            "evidence_refs": _mapping(self.evidence_refs),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterfactualEval":
        return cls(
            eval_id=str(payload.get("eval_id", "")),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            timestamp=str(payload.get("timestamp", "")),
            runtime_packet_id=payload.get("runtime_packet_id"),
            objective_profile_id=str(payload.get("objective_profile_id", "")),
            baseline_label=str(payload.get("baseline_label", "noop")),
            candidates=[
                CounterfactualCandidate.from_dict(item)
                for item in payload.get("candidates", []) or []
            ],
            recommended_action=str(payload.get("recommended_action", "noop")),
            evidence_refs=_mapping(payload.get("evidence_refs")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "counterfactual_eval_v1")),
        )


def build_counterfactual_eval(
    *,
    run_id: str,
    episode_id: str,
    timestamp: str,
    runtime_packet_id: Optional[str],
    objective_profile_id: str,
    baseline_value: float,
    branch_values: Sequence[Mapping[str, Any]],
    evidence_refs: Optional[Mapping[str, Any]] = None,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> CounterfactualEval:
    candidates: list[CounterfactualCandidate] = [
        CounterfactualCandidate(
            candidate_id=f"cf_{sha256_json({'episode_id': episode_id, 'label': 'noop'})[:12]}",
            label="noop",
            expected_net_value=float(baseline_value),
            deltas={"delta_value_vs_noop": 0.0},
            action={},
            metadata={"baseline": True},
        )
    ]
    for item in branch_values:
        label = str(item.get("label", "candidate"))
        expected_net_value = float(item.get("expected_net_value", baseline_value))
        candidates.append(
            CounterfactualCandidate(
                candidate_id=f"cf_{sha256_json({'episode_id': episode_id, 'label': label})[:12]}",
                label=label,
                expected_net_value=expected_net_value,
                deltas={
                    "delta_value_vs_noop": float(expected_net_value - baseline_value),
                    **_float_mapping(item.get("deltas")),
                },
                action=_float_mapping(item.get("action")),
                artifact_refs=_mapping(item.get("artifact_refs")),
                metadata=_mapping(item.get("metadata")),
            )
        )

    recommended = max(candidates, key=lambda candidate: candidate.expected_net_value).label if candidates else "noop"
    payload = {
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "timestamp": str(timestamp),
        "runtime_packet_id": runtime_packet_id,
        "objective_profile_id": str(objective_profile_id),
        "baseline_label": "noop",
        "candidates": [candidate.to_dict() for candidate in candidates],
        "recommended_action": recommended,
        "evidence_refs": _mapping(evidence_refs),
        "artifact_refs": _mapping(artifact_refs),
        "metadata": _mapping(metadata),
        "version": "counterfactual_eval_v1",
    }
    eval_id = f"cfeval_{sha256_json(payload)[:16]}"
    return CounterfactualEval(
        eval_id=eval_id,
        run_id=str(run_id),
        episode_id=str(episode_id),
        timestamp=str(timestamp),
        runtime_packet_id=runtime_packet_id,
        objective_profile_id=str(objective_profile_id),
        baseline_label="noop",
        candidates=candidates,
        recommended_action=recommended,
        evidence_refs=_mapping(evidence_refs),
        artifact_refs=_mapping(artifact_refs),
        metadata=_mapping(metadata),
    )


__all__ = [
    "CounterfactualCandidate",
    "CounterfactualEval",
    "build_counterfactual_eval",
]

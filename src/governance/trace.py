"""Governance trace sidecars for vetoes, reroutes, and advisory gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _strings(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in (values or [])]


@dataclass(frozen=True)
class GovernanceTraceEntry:
    """Single governance judgment attached to runtime packet and evidence state."""

    trace_id: str
    run_id: str
    episode_id: str
    timestamp: str
    node_id: str
    outcome: str
    reason_codes: list[str]
    runtime_packet_id: Optional[str] = None
    contract_id: Optional[str] = None
    source_event_ids: list[str] = field(default_factory=list)
    decision_id: Optional[str] = None
    evidence_refs: Dict[str, Any] = field(default_factory=dict)
    rule_refs: list[str] = field(default_factory=list)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "governance_trace_entry_v1"

    @classmethod
    def from_components(
        cls,
        *,
        run_id: str,
        episode_id: str,
        timestamp: str,
        node_id: str,
        outcome: str,
        reason_codes: Optional[Sequence[Any]] = None,
        runtime_packet_id: Optional[str] = None,
        contract_id: Optional[str] = None,
        source_event_ids: Optional[Sequence[Any]] = None,
        decision_id: Optional[str] = None,
        evidence_refs: Optional[Mapping[str, Any]] = None,
        rule_refs: Optional[Sequence[Any]] = None,
        artifact_refs: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        trace_id: Optional[str] = None,
        version: str = "governance_trace_entry_v1",
    ) -> "GovernanceTraceEntry":
        resolved_run_id = str(run_id)
        resolved_episode_id = str(episode_id)
        resolved_timestamp = str(timestamp)
        resolved_node_id = str(node_id)
        resolved_outcome = str(outcome)
        resolved_reason_codes = _strings(reason_codes)
        resolved_source_event_ids = _strings(source_event_ids)
        resolved_evidence_refs = _mapping(evidence_refs)
        resolved_rule_refs = _strings(rule_refs)
        resolved_artifact_refs = _mapping(artifact_refs)
        resolved_provenance = _mapping(provenance)
        resolved_metadata = _mapping(metadata)
        resolved_version = str(version)
        payload: Dict[str, Any] = {
            "run_id": resolved_run_id,
            "episode_id": resolved_episode_id,
            "timestamp": resolved_timestamp,
            "node_id": resolved_node_id,
            "outcome": resolved_outcome,
            "reason_codes": resolved_reason_codes,
            "runtime_packet_id": runtime_packet_id,
            "contract_id": contract_id,
            "source_event_ids": resolved_source_event_ids,
            "decision_id": decision_id,
            "evidence_refs": resolved_evidence_refs,
            "rule_refs": resolved_rule_refs,
            "artifact_refs": resolved_artifact_refs,
            "provenance": resolved_provenance,
            "metadata": resolved_metadata,
            "version": resolved_version,
        }
        resolved_id = trace_id or f"governance_{sha256_json(payload)[:16]}"
        return cls(
            trace_id=resolved_id,
            run_id=resolved_run_id,
            episode_id=resolved_episode_id,
            timestamp=resolved_timestamp,
            node_id=resolved_node_id,
            outcome=resolved_outcome,
            reason_codes=resolved_reason_codes,
            runtime_packet_id=runtime_packet_id,
            contract_id=contract_id,
            source_event_ids=resolved_source_event_ids,
            decision_id=decision_id,
            evidence_refs=resolved_evidence_refs,
            rule_refs=resolved_rule_refs,
            artifact_refs=resolved_artifact_refs,
            provenance=resolved_provenance,
            metadata=resolved_metadata,
            version=resolved_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "outcome": self.outcome,
            "reason_codes": list(self.reason_codes),
            "runtime_packet_id": self.runtime_packet_id,
            "contract_id": self.contract_id,
            "source_event_ids": list(self.source_event_ids),
            "decision_id": self.decision_id,
            "evidence_refs": _mapping(self.evidence_refs),
            "rule_refs": list(self.rule_refs),
            "artifact_refs": _mapping(self.artifact_refs),
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GovernanceTraceEntry":
        return cls(
            trace_id=str(payload.get("trace_id", "")),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            timestamp=str(payload.get("timestamp", "")),
            node_id=str(payload.get("node_id", "")),
            outcome=str(payload.get("outcome", "")),
            reason_codes=_strings(payload.get("reason_codes")),
            runtime_packet_id=payload.get("runtime_packet_id"),
            contract_id=payload.get("contract_id"),
            source_event_ids=_strings(payload.get("source_event_ids")),
            decision_id=payload.get("decision_id"),
            evidence_refs=_mapping(payload.get("evidence_refs")),
            rule_refs=_strings(payload.get("rule_refs")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "governance_trace_entry_v1")),
        )


def governance_trace_sidecar_payload(
    *,
    run_id: str,
    traces: Sequence[GovernanceTraceEntry],
    schema_version: str = "governance_trace_sidecar_v1",
) -> Dict[str, Any]:
    ordered = sorted(
        list(traces),
        key=lambda trace: (trace.run_id, trace.episode_id, trace.timestamp, trace.trace_id),
    )
    return {
        "schema_version": str(schema_version),
        "run_id": str(run_id),
        "trace_count": len(ordered),
        "traces": [trace.to_dict() for trace in ordered],
    }


__all__ = ["GovernanceTraceEntry", "governance_trace_sidecar_payload"]

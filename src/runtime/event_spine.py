"""Ordered runtime event and decision sidecars for additive auditability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _strings(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in (values or [])]


def _ordered_events(events: Sequence["RuntimeEvent"]) -> list["RuntimeEvent"]:
    return sorted(
        list(events),
        key=lambda event: (event.run_id, event.episode_id, int(event.sequence_idx), event.event_id),
    )


def _ordered_decisions(decisions: Sequence["DecisionLedgerEntry"]) -> list["DecisionLedgerEntry"]:
    return sorted(
        list(decisions),
        key=lambda decision: (
            decision.run_id,
            decision.episode_id,
            int(decision.sequence_idx),
            decision.decision_id,
        ),
    )


@dataclass(frozen=True)
class RuntimeEvent:
    """Single ordered runtime event used by the EventSpine sidecar."""

    event_id: str
    run_id: str
    episode_id: str
    timestamp: str
    event_kind: str
    sequence_idx: int
    scope: Dict[str, Any]
    runtime_packet_id: Optional[str]
    contract_id: Optional[str]
    receipt_label_refs: list[str] = field(default_factory=list)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "runtime_event_v1"

    @classmethod
    def from_components(
        cls,
        *,
        run_id: str,
        episode_id: str,
        timestamp: str,
        event_kind: str,
        sequence_idx: int,
        scope: Optional[Mapping[str, Any]] = None,
        runtime_packet_id: Optional[str] = None,
        contract_id: Optional[str] = None,
        receipt_label_refs: Optional[Sequence[Any]] = None,
        artifact_refs: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        event_id: Optional[str] = None,
        version: str = "runtime_event_v1",
    ) -> "RuntimeEvent":
        resolved_run_id = str(run_id)
        resolved_episode_id = str(episode_id)
        resolved_timestamp = str(timestamp)
        resolved_event_kind = str(event_kind)
        resolved_sequence_idx = int(sequence_idx)
        resolved_scope = _mapping(scope)
        resolved_receipt_label_refs = _strings(receipt_label_refs)
        resolved_artifact_refs = _mapping(artifact_refs)
        resolved_provenance = _mapping(provenance)
        resolved_metadata = _mapping(metadata)
        resolved_version = str(version)
        payload: Dict[str, Any] = {
            "run_id": resolved_run_id,
            "episode_id": resolved_episode_id,
            "timestamp": resolved_timestamp,
            "event_kind": resolved_event_kind,
            "sequence_idx": resolved_sequence_idx,
            "scope": resolved_scope,
            "runtime_packet_id": runtime_packet_id,
            "contract_id": contract_id,
            "receipt_label_refs": resolved_receipt_label_refs,
            "artifact_refs": resolved_artifact_refs,
            "provenance": resolved_provenance,
            "metadata": resolved_metadata,
            "version": resolved_version,
        }
        resolved_event_id = event_id or f"event_{sha256_json(payload)[:16]}"
        return cls(
            event_id=resolved_event_id,
            run_id=resolved_run_id,
            episode_id=resolved_episode_id,
            timestamp=resolved_timestamp,
            event_kind=resolved_event_kind,
            sequence_idx=resolved_sequence_idx,
            scope=resolved_scope,
            runtime_packet_id=runtime_packet_id,
            contract_id=contract_id,
            receipt_label_refs=resolved_receipt_label_refs,
            artifact_refs=resolved_artifact_refs,
            provenance=resolved_provenance,
            metadata=resolved_metadata,
            version=resolved_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "event_kind": self.event_kind,
            "sequence_idx": int(self.sequence_idx),
            "scope": _mapping(self.scope),
            "runtime_packet_id": self.runtime_packet_id,
            "contract_id": self.contract_id,
            "receipt_label_refs": list(self.receipt_label_refs),
            "artifact_refs": _mapping(self.artifact_refs),
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeEvent":
        return cls(
            event_id=str(payload.get("event_id", "")),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            timestamp=str(payload.get("timestamp", "")),
            event_kind=str(payload.get("event_kind", "")),
            sequence_idx=int(payload.get("sequence_idx", 0)),
            scope=_mapping(payload.get("scope")),
            runtime_packet_id=payload.get("runtime_packet_id"),
            contract_id=payload.get("contract_id"),
            receipt_label_refs=_strings(payload.get("receipt_label_refs")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "runtime_event_v1")),
        )


@dataclass(frozen=True)
class DecisionLedgerEntry:
    """Economically and governance-relevant decision sidecar row."""

    decision_id: str
    run_id: str
    episode_id: str
    timestamp: str
    decision_kind: str
    outcome: str
    sequence_idx: int
    scope: Dict[str, Any]
    reasons: list[str]
    source_event_ids: list[str]
    runtime_packet_id: Optional[str]
    contract_id: Optional[str]
    receipt_label_refs: list[str] = field(default_factory=list)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "decision_ledger_entry_v1"

    @classmethod
    def from_components(
        cls,
        *,
        run_id: str,
        episode_id: str,
        timestamp: str,
        decision_kind: str,
        outcome: str,
        sequence_idx: int,
        scope: Optional[Mapping[str, Any]] = None,
        reasons: Optional[Sequence[Any]] = None,
        source_event_ids: Optional[Sequence[Any]] = None,
        runtime_packet_id: Optional[str] = None,
        contract_id: Optional[str] = None,
        receipt_label_refs: Optional[Sequence[Any]] = None,
        artifact_refs: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        decision_id: Optional[str] = None,
        version: str = "decision_ledger_entry_v1",
    ) -> "DecisionLedgerEntry":
        resolved_run_id = str(run_id)
        resolved_episode_id = str(episode_id)
        resolved_timestamp = str(timestamp)
        resolved_decision_kind = str(decision_kind)
        resolved_outcome = str(outcome)
        resolved_sequence_idx = int(sequence_idx)
        resolved_scope = _mapping(scope)
        resolved_reasons = _strings(reasons)
        resolved_source_event_ids = _strings(source_event_ids)
        resolved_receipt_label_refs = _strings(receipt_label_refs)
        resolved_artifact_refs = _mapping(artifact_refs)
        resolved_provenance = _mapping(provenance)
        resolved_metadata = _mapping(metadata)
        resolved_version = str(version)
        payload: Dict[str, Any] = {
            "run_id": resolved_run_id,
            "episode_id": resolved_episode_id,
            "timestamp": resolved_timestamp,
            "decision_kind": resolved_decision_kind,
            "outcome": resolved_outcome,
            "sequence_idx": resolved_sequence_idx,
            "scope": resolved_scope,
            "reasons": resolved_reasons,
            "source_event_ids": resolved_source_event_ids,
            "runtime_packet_id": runtime_packet_id,
            "contract_id": contract_id,
            "receipt_label_refs": resolved_receipt_label_refs,
            "artifact_refs": resolved_artifact_refs,
            "provenance": resolved_provenance,
            "metadata": resolved_metadata,
            "version": resolved_version,
        }
        resolved_decision_id = decision_id or f"decision_{sha256_json(payload)[:16]}"
        return cls(
            decision_id=resolved_decision_id,
            run_id=resolved_run_id,
            episode_id=resolved_episode_id,
            timestamp=resolved_timestamp,
            decision_kind=resolved_decision_kind,
            outcome=resolved_outcome,
            sequence_idx=resolved_sequence_idx,
            scope=resolved_scope,
            reasons=resolved_reasons,
            source_event_ids=resolved_source_event_ids,
            runtime_packet_id=runtime_packet_id,
            contract_id=contract_id,
            receipt_label_refs=resolved_receipt_label_refs,
            artifact_refs=resolved_artifact_refs,
            provenance=resolved_provenance,
            metadata=resolved_metadata,
            version=resolved_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "decision_kind": self.decision_kind,
            "outcome": self.outcome,
            "sequence_idx": int(self.sequence_idx),
            "scope": _mapping(self.scope),
            "reasons": list(self.reasons),
            "source_event_ids": list(self.source_event_ids),
            "runtime_packet_id": self.runtime_packet_id,
            "contract_id": self.contract_id,
            "receipt_label_refs": list(self.receipt_label_refs),
            "artifact_refs": _mapping(self.artifact_refs),
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DecisionLedgerEntry":
        return cls(
            decision_id=str(payload.get("decision_id", "")),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            timestamp=str(payload.get("timestamp", "")),
            decision_kind=str(payload.get("decision_kind", "")),
            outcome=str(payload.get("outcome", "")),
            sequence_idx=int(payload.get("sequence_idx", 0)),
            scope=_mapping(payload.get("scope")),
            reasons=_strings(payload.get("reasons")),
            source_event_ids=_strings(payload.get("source_event_ids")),
            runtime_packet_id=payload.get("runtime_packet_id"),
            contract_id=payload.get("contract_id"),
            receipt_label_refs=_strings(payload.get("receipt_label_refs")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "decision_ledger_entry_v1")),
        )


def event_spine_sidecar_payload(
    *,
    run_id: str,
    events: Sequence[RuntimeEvent],
    schema_version: str = "event_spine_sidecar_v1",
) -> Dict[str, Any]:
    """Serialize the EventSpine sidecar payload."""

    ordered = _ordered_events(events)
    return {
        "schema_version": str(schema_version),
        "run_id": str(run_id),
        "event_count": len(ordered),
        "events": [event.to_dict() for event in ordered],
    }


def decision_ledger_sidecar_payload(
    *,
    run_id: str,
    decisions: Sequence[DecisionLedgerEntry],
    schema_version: str = "decision_ledger_sidecar_v1",
) -> Dict[str, Any]:
    """Serialize the DecisionLedger sidecar payload."""

    ordered = _ordered_decisions(decisions)
    return {
        "schema_version": str(schema_version),
        "run_id": str(run_id),
        "decision_count": len(ordered),
        "decisions": [decision.to_dict() for decision in ordered],
    }


__all__ = [
    "DecisionLedgerEntry",
    "RuntimeEvent",
    "decision_ledger_sidecar_payload",
    "event_spine_sidecar_payload",
]

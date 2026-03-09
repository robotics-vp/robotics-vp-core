"""Canonical evidence publication envelope and sidecar payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

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
class EvidenceRecord:
    """Single advisory evidence publication from a specialist or sidecar."""

    evidence_id: str
    episode_id: str
    timestamp: str
    source: str
    kind: str
    confidence: float
    disagreement: float = 0.0
    validity: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "evidence_record_v1"

    @classmethod
    def from_components(
        cls,
        *,
        episode_id: str,
        timestamp: str,
        source: str,
        kind: str,
        confidence: float,
        disagreement: float = 0.0,
        validity: Optional[Mapping[str, Any]] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
        artifact_refs: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        evidence_id: Optional[str] = None,
        version: str = "evidence_record_v1",
    ) -> "EvidenceRecord":
        resolved_episode_id = str(episode_id)
        resolved_timestamp = str(timestamp)
        resolved_source = str(source)
        resolved_kind = str(kind)
        resolved_confidence = float(confidence)
        resolved_disagreement = float(disagreement)
        resolved_validity = _mapping(validity)
        resolved_metrics = _float_mapping(metrics)
        resolved_payload = _mapping(payload)
        resolved_artifact_refs = _mapping(artifact_refs)
        resolved_provenance = _mapping(provenance)
        resolved_metadata = _mapping(metadata)
        resolved_version = str(version)
        record_payload: Dict[str, Any] = {
            "episode_id": resolved_episode_id,
            "timestamp": resolved_timestamp,
            "source": resolved_source,
            "kind": resolved_kind,
            "confidence": resolved_confidence,
            "disagreement": resolved_disagreement,
            "validity": resolved_validity,
            "metrics": resolved_metrics,
            "payload": resolved_payload,
            "artifact_refs": resolved_artifact_refs,
            "provenance": resolved_provenance,
            "metadata": resolved_metadata,
            "version": resolved_version,
        }
        resolved_id = evidence_id or f"evidence_{sha256_json(record_payload)[:16]}"
        return cls(
            evidence_id=resolved_id,
            episode_id=resolved_episode_id,
            timestamp=resolved_timestamp,
            source=resolved_source,
            kind=resolved_kind,
            confidence=resolved_confidence,
            disagreement=resolved_disagreement,
            validity=resolved_validity,
            metrics=resolved_metrics,
            payload=resolved_payload,
            artifact_refs=resolved_artifact_refs,
            provenance=resolved_provenance,
            metadata=resolved_metadata,
            version=resolved_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "kind": self.kind,
            "confidence": float(self.confidence),
            "disagreement": float(self.disagreement),
            "validity": _mapping(self.validity),
            "metrics": _float_mapping(self.metrics),
            "payload": _mapping(self.payload),
            "artifact_refs": _mapping(self.artifact_refs),
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceRecord":
        return cls(
            evidence_id=str(payload.get("evidence_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            timestamp=str(payload.get("timestamp", "")),
            source=str(payload.get("source", "")),
            kind=str(payload.get("kind", "")),
            confidence=float(payload.get("confidence", 0.0)),
            disagreement=float(payload.get("disagreement", 0.0)),
            validity=_mapping(payload.get("validity")),
            metrics=_float_mapping(payload.get("metrics")),
            payload=_mapping(payload.get("payload")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "evidence_record_v1")),
        )


class EvidenceBus:
    """Append-only evidence publication bus for sidecar-grade supervision."""

    def __init__(self, records: Optional[Iterable[EvidenceRecord]] = None) -> None:
        self._records: list[EvidenceRecord] = []
        if records:
            self.publish_many(records)

    def publish(self, record: EvidenceRecord) -> EvidenceRecord:
        self._records.append(record)
        return record

    def publish_many(self, records: Iterable[EvidenceRecord]) -> None:
        for record in records:
            self.publish(record)

    def records(self) -> list[EvidenceRecord]:
        return sorted(
            list(self._records),
            key=lambda record: (record.episode_id, record.timestamp, record.source, record.kind, record.evidence_id),
        )

    def for_episode(self, episode_id: str) -> list[EvidenceRecord]:
        return [record for record in self.records() if record.episode_id == str(episode_id)]

    def summarize_episode(self, episode_id: str) -> Dict[str, Any]:
        records = self.for_episode(episode_id)
        if not records:
            return {
                "episode_id": str(episode_id),
                "record_count": 0,
                "confidence_mean": 0.0,
                "disagreement_mean": 0.0,
                "coverage": 0.0,
                "sources": {},
                "kinds": {},
                "artifact_refs": {},
            }

        confidence_mean = sum(record.confidence for record in records) / float(len(records))
        disagreement_mean = sum(record.disagreement for record in records) / float(len(records))
        sources: Dict[str, int] = {}
        kinds: Dict[str, int] = {}
        artifact_refs: Dict[str, Any] = {}
        for record in records:
            sources[record.source] = sources.get(record.source, 0) + 1
            kinds[record.kind] = kinds.get(record.kind, 0) + 1
            artifact_refs.update(record.artifact_refs)

        coverage = min(1.0, float(len(records)) / 4.0)
        return {
            "episode_id": str(episode_id),
            "record_count": len(records),
            "confidence_mean": float(confidence_mean),
            "disagreement_mean": float(disagreement_mean),
            "coverage": float(coverage),
            "sources": sources,
            "kinds": kinds,
            "artifact_refs": artifact_refs,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "evidence_bus_v1",
            "record_count": len(self._records),
            "records": [record.to_dict() for record in self.records()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceBus":
        return cls(
            records=[
                EvidenceRecord.from_dict(item)
                for item in payload.get("records", []) or []
            ]
        )


__all__ = ["EvidenceBus", "EvidenceRecord"]

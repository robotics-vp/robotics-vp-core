"""Dense value-target sidecars derived from governed runtime artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


@dataclass(frozen=True)
class ValueTarget:
    """Single dense supervision target for routing/adaptation decisions."""

    target_id: str
    name: str
    target_kind: str
    target_value: float
    confidence: float
    source_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "name": self.name,
            "target_kind": self.target_kind,
            "target_value": float(self.target_value),
            "confidence": float(self.confidence),
            "source_refs": _mapping(self.source_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValueTarget":
        return cls(
            target_id=str(payload.get("target_id", "")),
            name=str(payload.get("name", "")),
            target_kind=str(payload.get("target_kind", "")),
            target_value=float(payload.get("target_value", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            source_refs=_mapping(payload.get("source_refs")),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class ValueTargetPack:
    """Dense supervision bundle attached to one episode/runtime packet."""

    pack_id: str
    run_id: str
    episode_id: str
    runtime_packet_id: Optional[str]
    targets: list[ValueTarget] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "value_target_pack_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "runtime_packet_id": self.runtime_packet_id,
            "targets": [target.to_dict() for target in self.targets],
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValueTargetPack":
        return cls(
            pack_id=str(payload.get("pack_id", "")),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            runtime_packet_id=payload.get("runtime_packet_id"),
            targets=[
                ValueTarget.from_dict(item)
                for item in payload.get("targets", []) or []
            ],
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "value_target_pack_v1")),
        )


def build_value_target_pack(
    *,
    run_id: str,
    episode_id: str,
    runtime_packet_id: Optional[str],
    base_value: float,
    recommended_value: float,
    disagreement: float,
    coverage: float,
    counterfactual_eval_id: Optional[str] = None,
    pricing_tick_ref: Optional[str] = None,
    governance_trace_ref: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ValueTargetPack:
    targets = [
        ValueTarget(
            target_id=f"target_{sha256_json({'episode_id': episode_id, 'name': 'route_value'})[:12]}",
            name="route_value",
            target_kind="route",
            target_value=float(recommended_value),
            confidence=float(max(0.05, min(1.0, coverage))),
            source_refs={
                "counterfactual_eval_id": counterfactual_eval_id,
                "pricing_tick_ref": pricing_tick_ref,
            },
        ),
        ValueTarget(
            target_id=f"target_{sha256_json({'episode_id': episode_id, 'name': 'collect_data_value'})[:12]}",
            name="collect_data_value",
            target_kind="collect_data",
            target_value=float(max(0.0, disagreement) * max(1.0, base_value * 0.05)),
            confidence=float(max(0.05, min(1.0, disagreement + 0.1))),
            source_refs={"governance_trace_ref": governance_trace_ref},
        ),
        ValueTarget(
            target_id=f"target_{sha256_json({'episode_id': episode_id, 'name': 'adapt_value'})[:12]}",
            name="adapt_value",
            target_kind="adapt",
            target_value=float(max(0.0, recommended_value - base_value)),
            confidence=float(max(0.05, min(1.0, coverage * (1.0 - disagreement)))),
            source_refs={
                "counterfactual_eval_id": counterfactual_eval_id,
                "governance_trace_ref": governance_trace_ref,
            },
        ),
    ]
    payload = {
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "runtime_packet_id": runtime_packet_id,
        "targets": [target.to_dict() for target in targets],
        "metadata": _mapping(metadata),
        "version": "value_target_pack_v1",
    }
    pack_id = f"value_targets_{sha256_json(payload)[:16]}"
    return ValueTargetPack(
        pack_id=pack_id,
        run_id=str(run_id),
        episode_id=str(episode_id),
        runtime_packet_id=runtime_packet_id,
        targets=targets,
        metadata=_mapping(metadata),
    )


__all__ = ["ValueTarget", "ValueTargetPack", "build_value_target_pack"]

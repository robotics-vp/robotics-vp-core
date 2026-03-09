"""Canonical action adapter contracts for runtime packets and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from src.runtime.packets import SchemaRef
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
class ActionAdapterV2:
    """Embodiment-aware action contract with timing and translator provenance."""

    schema_id: str
    channel_order: list[str] = field(
        default_factory=lambda: ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    )
    control_hz: float = 10.0
    latency_ms: float = 0.0
    translator_ref: Optional[str] = None
    embodiment_id: Optional[str] = None
    bounds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "action_adapter_v2"

    def normalize(self, action: Mapping[str, Any] | Sequence[Any]) -> Dict[str, Any]:
        if isinstance(action, Mapping):
            action_map = _float_mapping(action)
            vector = [float(action_map.get(channel, 0.0)) for channel in self.channel_order]
        else:
            values = list(action)
            vector = []
            action_map: Dict[str, float] = {}
            for idx, channel in enumerate(self.channel_order):
                value = float(values[idx]) if idx < len(values) else 0.0
                action_map[channel] = value
                vector.append(value)
        return {
            "schema_id": self.schema_id,
            "channel_order": list(self.channel_order),
            "vector": vector,
            "named": action_map,
            "timing": {"apply_hz": float(self.control_hz), "latency_ms": float(self.latency_ms)},
            "translator_ref": self.translator_ref,
            "embodiment_id": self.embodiment_id,
        }

    def to_schema_ref(self) -> SchemaRef:
        return SchemaRef(
            schema_id=self.schema_id,
            version=self.version,
            shape={"action_vector": len(self.channel_order), "channels": list(self.channel_order)},
            timing={"apply_hz": float(self.control_hz), "latency_ms": float(self.latency_ms)},
            provenance={
                **_mapping(self.provenance),
                "translator_ref": self.translator_ref,
                "embodiment_id": self.embodiment_id,
            },
            metadata={
                **_mapping(self.metadata),
                "bounds": _mapping(self.bounds),
            },
        )


__all__ = ["ActionAdapterV2"]

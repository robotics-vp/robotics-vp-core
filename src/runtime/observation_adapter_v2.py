"""Canonical observation adapter contracts for runtime packets and replay."""

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


def _sequence(values: Optional[Sequence[Any]]) -> list[float]:
    seq: list[float] = []
    for value in values or []:
        try:
            seq.append(float(value))
        except Exception:
            seq.append(0.0)
    return seq


@dataclass(frozen=True)
class ObservationAdapterV2:
    """Observation contract that keeps refs and timing explicit."""

    schema_id: str
    proprio_fields: list[str] = field(default_factory=list)
    sensor_refs: list[str] = field(default_factory=list)
    sample_hz: float = 10.0
    latency_ms: float = 0.0
    translator_ref: Optional[str] = None
    embodiment_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "observation_adapter_v2"

    def normalize(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        proprio = observation.get("proprio")
        if isinstance(proprio, Mapping):
            proprio_map = _float_mapping(proprio)
            proprio_vector = [float(proprio_map.get(field, 0.0)) for field in self.proprio_fields]
        else:
            proprio_vector = _sequence(proprio if isinstance(proprio, Sequence) else [])
            proprio_map = {
                field: float(proprio_vector[idx]) if idx < len(proprio_vector) else 0.0
                for idx, field in enumerate(self.proprio_fields)
            }

        quality_metrics = _float_mapping(observation.get("quality_metrics"))
        refs = _mapping(observation.get("artifact_refs"))
        for key in ("scene_tracks_ref", "semantic_fusion_ref", "belief_state_ref", "teacher_trace_ref"):
            value = observation.get(key)
            if value:
                refs[key] = str(value)

        return {
            "schema_id": self.schema_id,
            "proprio": proprio_map,
            "proprio_vector": proprio_vector,
            "artifact_refs": refs,
            "quality_metrics": quality_metrics,
            "timing": {"sample_hz": float(self.sample_hz), "latency_ms": float(self.latency_ms)},
            "translator_ref": self.translator_ref,
            "embodiment_id": self.embodiment_id,
        }

    def to_schema_ref(self) -> SchemaRef:
        return SchemaRef(
            schema_id=self.schema_id,
            version=self.version,
            shape={
                "proprio_vector": len(self.proprio_fields),
                "proprio_fields": list(self.proprio_fields),
                "sensor_refs": list(self.sensor_refs),
            },
            timing={"sample_hz": float(self.sample_hz), "latency_ms": float(self.latency_ms)},
            provenance={
                **_mapping(self.provenance),
                "translator_ref": self.translator_ref,
                "embodiment_id": self.embodiment_id,
            },
            metadata=_mapping(self.metadata),
        )


__all__ = ["ObservationAdapterV2"]

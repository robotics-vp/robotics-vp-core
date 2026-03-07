"""Embodiment registry scaffolding for capability and schema normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

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


def _bounds(payload: Optional[Mapping[str, Mapping[str, Any]]]) -> Dict[str, Dict[str, float]]:
    bounds: Dict[str, Dict[str, float]] = {}
    for axis, spec in dict(payload or {}).items():
        axis_bounds: Dict[str, float] = {}
        for key, value in dict(spec or {}).items():
            try:
                axis_bounds[str(key)] = float(value)
            except Exception:
                continue
        bounds[str(axis)] = axis_bounds
    return bounds


@dataclass(frozen=True)
class CapabilityProfile:
    """Embodiment-agnostic capability summary for a robot family or executor."""

    profile_id: str
    robot_family: str
    sensor_modalities: List[str] = field(default_factory=list)
    action_spaces: List[str] = field(default_factory=list)
    workspace_bounds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    skill_capabilities: Dict[str, float] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    safety_envelopes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "robot_family": self.robot_family,
            "sensor_modalities": list(self.sensor_modalities),
            "action_spaces": list(self.action_spaces),
            "workspace_bounds": _bounds(self.workspace_bounds),
            "skill_capabilities": _float_mapping(self.skill_capabilities),
            "timing": _float_mapping(self.timing),
            "safety_envelopes": _mapping(self.safety_envelopes),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityProfile":
        return cls(
            profile_id=str(payload.get("profile_id", "")),
            robot_family=str(payload.get("robot_family", "")),
            sensor_modalities=[str(value) for value in payload.get("sensor_modalities", []) or []],
            action_spaces=[str(value) for value in payload.get("action_spaces", []) or []],
            workspace_bounds=_bounds(payload.get("workspace_bounds")),
            skill_capabilities=_float_mapping(payload.get("skill_capabilities")),
            timing=_float_mapping(payload.get("timing")),
            safety_envelopes=_mapping(payload.get("safety_envelopes")),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class EmbodimentRegistryEntry:
    """Single embodiment mapping from robot identity to normalized capabilities."""

    embodiment_id: str
    robot_id: str
    robot_family: str
    capability_profile: CapabilityProfile
    observation_schema_id: str
    action_schema_id: str
    translator_refs: Dict[str, str] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "embodiment_id": self.embodiment_id,
            "robot_id": self.robot_id,
            "robot_family": self.robot_family,
            "capability_profile": self.capability_profile.to_dict(),
            "observation_schema_id": self.observation_schema_id,
            "action_schema_id": self.action_schema_id,
            "translator_refs": {str(key): str(value) for key, value in self.translator_refs.items()},
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmbodimentRegistryEntry":
        return cls(
            embodiment_id=str(payload.get("embodiment_id", "")),
            robot_id=str(payload.get("robot_id", "")),
            robot_family=str(payload.get("robot_family", "")),
            capability_profile=CapabilityProfile.from_dict(payload.get("capability_profile", {}) or {}),
            observation_schema_id=str(payload.get("observation_schema_id", "")),
            action_schema_id=str(payload.get("action_schema_id", "")),
            translator_refs={
                str(key): str(value)
                for key, value in dict(payload.get("translator_refs", {}) or {}).items()
            },
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
        )


class EmbodimentRegistry:
    """In-memory registry used to normalize embodiment-specific translators."""

    def __init__(self, entries: Optional[Iterable[EmbodimentRegistryEntry]] = None) -> None:
        self._entries: Dict[str, EmbodimentRegistryEntry] = {}
        for entry in entries or []:
            self.register(entry)

    def register(self, entry: EmbodimentRegistryEntry) -> None:
        self._entries[entry.embodiment_id] = entry

    def get(self, embodiment_id: str) -> Optional[EmbodimentRegistryEntry]:
        return self._entries.get(str(embodiment_id))

    def list_entries(self) -> List[EmbodimentRegistryEntry]:
        return [self._entries[key] for key in sorted(self._entries.keys())]

    def resolve_capability_profile(self, embodiment_id: str) -> Optional[CapabilityProfile]:
        entry = self.get(embodiment_id)
        return entry.capability_profile if entry else None

    def resolve_observation_schema(self, embodiment_id: str) -> Optional[str]:
        entry = self.get(embodiment_id)
        return entry.observation_schema_id if entry else None

    def resolve_action_schema(self, embodiment_id: str) -> Optional[str]:
        entry = self.get(embodiment_id)
        return entry.action_schema_id if entry else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "embodiment_registry_v1",
            "entries": [entry.to_dict() for entry in self.list_entries()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmbodimentRegistry":
        entries = [
            EmbodimentRegistryEntry.from_dict(entry)
            for entry in payload.get("entries", []) or []
        ]
        return cls(entries=entries)


__all__ = [
    "CapabilityProfile",
    "EmbodimentRegistry",
    "EmbodimentRegistryEntry",
]

"""Teacher-trace sidecars for external VLA or foundation-model outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
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


def _strings(values: Optional[Sequence[Any]]) -> list[str]:
    return [str(value) for value in (values or [])]


@dataclass(frozen=True)
class TeacherStep:
    """Single advisory teacher output at one step or clip slice."""

    step_idx: int
    timestamp: str = ""
    instruction: str = ""
    action: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    semantic_tags: list[str] = field(default_factory=list)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": int(self.step_idx),
            "timestamp": self.timestamp,
            "instruction": self.instruction,
            "action": _float_mapping(self.action),
            "confidence": float(self.confidence),
            "semantic_tags": list(self.semantic_tags),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TeacherStep":
        return cls(
            step_idx=int(payload.get("step_idx", 0)),
            timestamp=str(payload.get("timestamp", "")),
            instruction=str(payload.get("instruction", "")),
            action=_float_mapping(payload.get("action")),
            confidence=float(payload.get("confidence", 0.0)),
            semantic_tags=_strings(payload.get("semantic_tags")),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
        )


@dataclass(frozen=True)
class TeacherTrace:
    """Advisory teacher trace kept separate from native truth."""

    trace_id: str
    episode_id: str
    teacher_id: str
    modality: str
    advisory_only: bool
    instruction: str
    steps: list[TeacherStep] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "teacher_trace_v1"

    @classmethod
    def from_components(
        cls,
        *,
        episode_id: str,
        teacher_id: str,
        modality: str,
        advisory_only: bool = True,
        instruction: str = "",
        steps: Optional[Sequence[TeacherStep]] = None,
        summary: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        trace_id: Optional[str] = None,
        version: str = "teacher_trace_v1",
    ) -> "TeacherTrace":
        step_list = [step if isinstance(step, TeacherStep) else TeacherStep.from_dict(step) for step in (steps or [])]
        resolved_episode_id = str(episode_id)
        resolved_teacher_id = str(teacher_id)
        resolved_modality = str(modality)
        resolved_advisory_only = bool(advisory_only)
        resolved_instruction = str(instruction)
        resolved_summary = _float_mapping(summary)
        resolved_provenance = _mapping(provenance)
        resolved_metadata = _mapping(metadata)
        resolved_version = str(version)
        payload: Dict[str, Any] = {
            "episode_id": resolved_episode_id,
            "teacher_id": resolved_teacher_id,
            "modality": resolved_modality,
            "advisory_only": resolved_advisory_only,
            "instruction": resolved_instruction,
            "steps": [step.to_dict() for step in step_list],
            "summary": resolved_summary,
            "provenance": resolved_provenance,
            "metadata": resolved_metadata,
            "version": resolved_version,
        }
        resolved_id = trace_id or f"teacher_{sha256_json(payload)[:16]}"
        return cls(
            trace_id=resolved_id,
            episode_id=resolved_episode_id,
            teacher_id=resolved_teacher_id,
            modality=resolved_modality,
            advisory_only=resolved_advisory_only,
            instruction=resolved_instruction,
            steps=step_list,
            summary=resolved_summary,
            provenance=resolved_provenance,
            metadata=resolved_metadata,
            version=resolved_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "episode_id": self.episode_id,
            "teacher_id": self.teacher_id,
            "modality": self.modality,
            "advisory_only": bool(self.advisory_only),
            "instruction": self.instruction,
            "steps": [step.to_dict() for step in self.steps],
            "summary": _float_mapping(self.summary),
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TeacherTrace":
        return cls(
            trace_id=str(payload.get("trace_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            teacher_id=str(payload.get("teacher_id", "")),
            modality=str(payload.get("modality", "")),
            advisory_only=bool(payload.get("advisory_only", True)),
            instruction=str(payload.get("instruction", "")),
            steps=[
                TeacherStep.from_dict(step)
                for step in payload.get("steps", []) or []
            ],
            summary=_float_mapping(payload.get("summary")),
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "teacher_trace_v1")),
        )

    @classmethod
    def from_vla_action(
        cls,
        *,
        episode_id: str,
        instruction: str,
        semantic_tags: Optional[Sequence[Any]] = None,
        action: Optional[Mapping[str, Any]] = None,
        teacher_id: str = "openvla",
        timestamp: str = "",
        availability_reason: Optional[str] = None,
    ) -> "TeacherTrace":
        action_payload = _float_mapping(action)
        confidence = float(action_payload.get("confidence", 0.0))
        if confidence <= 0.0 and action_payload.get("vla_available", 0.0) > 0.0:
            confidence = 0.35
        step = TeacherStep(
            step_idx=0,
            timestamp=str(timestamp),
            instruction=str(instruction),
            action=action_payload,
            confidence=float(confidence),
            semantic_tags=_strings(semantic_tags),
            metadata={
                "availability_reason": str(availability_reason or ""),
                "vla_available": bool(action_payload.get("vla_available", 0.0) > 0.0),
            },
        )
        return cls.from_components(
            episode_id=episode_id,
            teacher_id=teacher_id,
            modality="action_semantics",
            advisory_only=True,
            instruction=instruction,
            steps=[step],
            summary={
                "teacher_confidence_mean": float(confidence),
                "step_count": 1.0,
            },
            provenance={
                "source": teacher_id,
                "availability_reason": str(availability_reason or ""),
            },
            metadata={"semantic_tags": _strings(semantic_tags)},
        )


def save_teacher_trace_json(path: Path, trace: TeacherTrace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace.to_dict(), indent=2))


def load_teacher_trace_json(path: Path) -> TeacherTrace:
    return TeacherTrace.from_dict(json.loads(path.read_text()))


__all__ = [
    "TeacherStep",
    "TeacherTrace",
    "load_teacher_trace_json",
    "save_teacher_trace_json",
]

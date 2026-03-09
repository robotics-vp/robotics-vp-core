"""Teacher-runtime contracts for external action/semantic teachers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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
class TeacherAdapterContract:
    """Static contract for an external teacher runtime."""

    teacher_id: str
    model_name: str
    modality: str
    advisory_only: bool = True
    available: bool = False
    action_schema_id: str = "teacher_action_envelope_v1"
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "teacher_adapter_contract_v1"

    @property
    def contract_id(self) -> str:
        return f"teacher_contract_{sha256_json(self.to_dict())[:16]}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher_id": self.teacher_id,
            "model_name": self.model_name,
            "modality": self.modality,
            "advisory_only": bool(self.advisory_only),
            "available": bool(self.available),
            "action_schema_id": self.action_schema_id,
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class TeacherActionEnvelope:
    """Runtime teacher output with explicit availability and fallback semantics."""

    teacher_id: str
    model_name: str
    instruction: str
    available: bool
    action: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    failure_mode: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "teacher_action_envelope_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher_id": self.teacher_id,
            "model_name": self.model_name,
            "instruction": self.instruction,
            "available": bool(self.available),
            "action": _float_mapping(self.action),
            "confidence": float(self.confidence),
            "failure_mode": self.failure_mode,
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }

    def to_vla_payload(self) -> Dict[str, Any]:
        payload = _float_mapping(self.action)
        payload.update(
            {
                "vla_available": bool(self.available),
                "confidence": float(self.confidence),
                "source": str(self.model_name),
                "fallback_mode": str(
                    self.failure_mode or ("teacher_available" if self.available else "teacher_unavailable")
                ),
            }
        )
        return payload

    @classmethod
    def unavailable(
        cls,
        *,
        teacher_id: str,
        model_name: str,
        instruction: str,
        failure_mode: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "TeacherActionEnvelope":
        return cls(
            teacher_id=teacher_id,
            model_name=model_name,
            instruction=instruction,
            available=False,
            action={},
            confidence=0.0,
            failure_mode=str(failure_mode),
            metadata=_mapping(metadata),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TeacherActionEnvelope":
        return cls(
            teacher_id=str(payload.get("teacher_id", "")),
            model_name=str(payload.get("model_name", "")),
            instruction=str(payload.get("instruction", "")),
            available=bool(payload.get("available", False)),
            action=_float_mapping(payload.get("action")),
            confidence=float(payload.get("confidence", 0.0)),
            failure_mode=str(payload.get("failure_mode", "")),
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
            version=str(payload.get("version", "teacher_action_envelope_v1")),
        )


class OpenVLATeacherRuntime:
    """Thin contract wrapper around OpenVLAController."""

    def __init__(self, controller: Any) -> None:
        self.controller = controller

    def describe_contract(self) -> TeacherAdapterContract:
        cfg = getattr(self.controller, "cfg", None)
        model_name = str(getattr(cfg, "model_name", "openvla"))
        available = bool(getattr(self.controller, "available", False))
        return TeacherAdapterContract(
            teacher_id="openvla",
            model_name=model_name,
            modality="action_semantics",
            advisory_only=True,
            available=available,
            metadata={
                "device": str(getattr(cfg, "device", "unknown")),
                "dtype": str(getattr(cfg, "dtype", "unknown")),
            },
        )

    def predict_action(self, image: Any, instruction: str) -> TeacherActionEnvelope:
        contract = self.describe_contract()
        try:
            payload = self.controller.predict_action(image, instruction)
        except Exception as exc:
            unavailable = TeacherActionEnvelope.unavailable(
                teacher_id=contract.teacher_id,
                model_name=contract.model_name,
                instruction=instruction,
                failure_mode=str(exc),
                metadata={"contract_id": contract.contract_id},
            )
            return TeacherActionEnvelope(
                teacher_id=unavailable.teacher_id,
                model_name=unavailable.model_name,
                instruction=unavailable.instruction,
                available=unavailable.available,
                action=unavailable.action,
                confidence=unavailable.confidence,
                failure_mode=unavailable.failure_mode,
                provenance={
                    "contract_id": contract.contract_id,
                    "action_schema_id": contract.action_schema_id,
                },
                metadata=unavailable.metadata,
            )
        available = bool(payload.get("vla_available", False))
        return TeacherActionEnvelope(
            teacher_id=contract.teacher_id,
            model_name=contract.model_name,
            instruction=instruction,
            available=available,
            action=_float_mapping(payload),
            confidence=float(payload.get("confidence", 0.0)),
            failure_mode=str(payload.get("fallback_mode", "teacher_available" if available else "teacher_unavailable")),
            provenance={
                "contract_id": contract.contract_id,
                "action_schema_id": contract.action_schema_id,
            },
            metadata={"available": available},
        )


def save_teacher_adapter_contract_json(path: Path, contract: TeacherAdapterContract) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract.to_dict(), indent=2))


def load_teacher_adapter_contract_json(path: Path) -> TeacherAdapterContract:
    payload = json.loads(path.read_text())
    return TeacherAdapterContract(
        teacher_id=str(payload.get("teacher_id", "")),
        model_name=str(payload.get("model_name", "")),
        modality=str(payload.get("modality", "")),
        advisory_only=bool(payload.get("advisory_only", True)),
        available=bool(payload.get("available", False)),
        action_schema_id=str(payload.get("action_schema_id", "teacher_action_envelope_v1")),
        metadata=_mapping(payload.get("metadata")),
        version=str(payload.get("version", "teacher_adapter_contract_v1")),
    )


def save_teacher_action_envelope_json(path: Path, envelope: TeacherActionEnvelope) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope.to_dict(), indent=2))


def load_teacher_action_envelope_json(path: Path) -> TeacherActionEnvelope:
    return TeacherActionEnvelope.from_dict(json.loads(path.read_text()))


__all__ = [
    "OpenVLATeacherRuntime",
    "TeacherActionEnvelope",
    "TeacherAdapterContract",
    "load_teacher_action_envelope_json",
    "load_teacher_adapter_contract_json",
    "save_teacher_action_envelope_json",
    "save_teacher_adapter_contract_json",
]

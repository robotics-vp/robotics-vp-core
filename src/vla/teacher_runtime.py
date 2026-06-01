"""Teacher-runtime contracts for external action/semantic teachers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.evidence.preconditions import build_execution_preconditions
from src.evidence.teacher_trace import (
    build_teacher_provider_truth,
    infer_teacher_semantics,
)
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
    provider_truth: Dict[str, Any] = field(default_factory=dict)
    version: str = "teacher_adapter_contract_v1"

    @property
    def contract_id(self) -> str:
        return f"teacher_contract_{sha256_json(self.to_dict())[:16]}"

    def _resolved_provider_truth(self) -> Dict[str, Any]:
        if self.provider_truth:
            return _mapping(self.provider_truth)
        metadata = _mapping(self.metadata)
        backend_status = metadata.get("backend_status")
        backend_selected = ""
        failure_reason = ""
        if isinstance(backend_status, Mapping):
            backend_selected = str(backend_status.get("backend_selected", "") or "")
            failure_reason = str(backend_status.get("failure_reason", "") or "")
        backend_selected = backend_selected or str(
            metadata.get("backend_selected", "") or ""
        )
        failure_reason = failure_reason or str(
            metadata.get("availability_reason", "") or ""
        )
        return build_teacher_provider_truth(
            provider_id=self.teacher_id,
            provider_name=self.model_name,
            available=bool(self.available),
            backend_selected=backend_selected,
            fallback_mode=failure_reason,
            confidence=1.0 if self.available else 0.0,
            metadata={
                "vision_backbone_selected": str(
                    metadata.get("vision_backbone_selected", "") or ""
                ),
                "backend_policy": str(metadata.get("backend_policy", "") or ""),
                "failure_reason": failure_reason,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher_id": self.teacher_id,
            "model_name": self.model_name,
            "modality": self.modality,
            "advisory_only": bool(self.advisory_only),
            "available": bool(self.available),
            "action_schema_id": self.action_schema_id,
            "metadata": _mapping(self.metadata),
            "provider_truth": self._resolved_provider_truth(),
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
    semantic_tags: list[str] = field(default_factory=list)
    object_refs: list[str] = field(default_factory=list)
    affordance_hints: list[str] = field(default_factory=list)
    risk_hints: list[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provider_truth: Dict[str, Any] = field(default_factory=dict)
    version: str = "teacher_action_envelope_v1"

    def _resolved_provider_truth(self) -> Dict[str, Any]:
        if self.provider_truth:
            return _mapping(self.provider_truth)
        metadata = _mapping(self.metadata)
        return build_teacher_provider_truth(
            provider_id=self.teacher_id,
            provider_name=self.model_name,
            available=bool(self.available),
            backend_selected=str(
                metadata.get("backend_selected", "")
                or ("real" if self.available else "unavailable")
            ),
            fallback_mode=str(self.failure_mode or metadata.get("failure_reason", "")),
            confidence=float(self.confidence),
            metadata={
                "vision_backbone_selected": str(
                    metadata.get("vision_backbone_selected", "") or ""
                ),
                "backend_policy": str(metadata.get("backend_policy", "") or ""),
                "failure_reason": str(
                    metadata.get("failure_reason", "") or self.failure_mode
                ),
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "teacher_id": self.teacher_id,
            "model_name": self.model_name,
            "instruction": self.instruction,
            "available": bool(self.available),
            "action": _float_mapping(self.action),
            "confidence": float(self.confidence),
            "failure_mode": self.failure_mode,
            "semantic_tags": list(self.semantic_tags),
            "object_refs": list(self.object_refs),
            "affordance_hints": list(self.affordance_hints),
            "risk_hints": list(self.risk_hints),
            "provenance": _mapping(self.provenance),
            "metadata": _mapping(self.metadata),
            "provider_truth": self._resolved_provider_truth(),
            "version": self.version,
        }

    def to_vla_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(_float_mapping(self.action))
        payload.update(
            {
                "vla_available": bool(self.available),
                "confidence": float(self.confidence),
                "source": str(self.model_name),
                "fallback_mode": str(
                    self.failure_mode
                    or (
                        "teacher_available" if self.available else "teacher_unavailable"
                    )
                ),
                "semantic_tags": list(self.semantic_tags),
                "object_refs": list(self.object_refs),
                "affordance_hints": list(self.affordance_hints),
                "risk_hints": list(self.risk_hints),
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
        resolved_metadata = _mapping(metadata)
        semantic_bundle = infer_teacher_semantics(
            instruction=instruction, metadata=resolved_metadata
        )
        return cls(
            teacher_id=teacher_id,
            model_name=model_name,
            instruction=instruction,
            available=False,
            action={},
            confidence=0.0,
            failure_mode=str(failure_mode),
            semantic_tags=semantic_bundle["semantic_tags"],
            object_refs=semantic_bundle["object_refs"],
            affordance_hints=semantic_bundle["affordance_hints"],
            risk_hints=semantic_bundle["risk_hints"],
            metadata=resolved_metadata,
            provider_truth=build_teacher_provider_truth(
                provider_id=teacher_id,
                provider_name=model_name,
                available=False,
                backend_selected=str(
                    resolved_metadata.get("backend_selected", "") or "unavailable"
                ),
                fallback_mode=failure_mode,
                confidence=0.0,
                metadata=resolved_metadata,
            ),
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
            semantic_tags=[str(tag) for tag in payload.get("semantic_tags", []) or []],
            object_refs=[str(tag) for tag in payload.get("object_refs", []) or []],
            affordance_hints=[
                str(tag) for tag in payload.get("affordance_hints", []) or []
            ],
            risk_hints=[str(tag) for tag in payload.get("risk_hints", []) or []],
            provenance=_mapping(payload.get("provenance")),
            metadata=_mapping(payload.get("metadata")),
            provider_truth=_mapping(payload.get("provider_truth")),
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
        backend_status = (
            self.controller.backend_status()
            if hasattr(self.controller, "backend_status")
            else {
                "backend_selected": "real" if available else "unavailable",
                "backend_policy": str(getattr(cfg, "backend_policy", "auto")),
                "failure_reason": "",
                "vision_backbone_selected": "",
                "vision_backbone_policy": str(
                    getattr(cfg, "vision_backbone_policy", "auto")
                ),
            }
        )
        preconditions = build_execution_preconditions(
            subject_id="openvla",
            subject_kind="teacher_runtime",
            artifact_refs={"model_name": model_name},
            signal_values={
                "teacher_available": available,
                "teacher_real_backend": str(backend_status.get("backend_selected", ""))
                == "real",
                "advisory_only": True,
            },
            required_boolean_signals={"teacher_available": True},
            metadata={
                "device": str(getattr(cfg, "device", "unknown")),
                "dtype": str(getattr(cfg, "dtype", "unknown")),
                "backend_status": backend_status,
            },
        )
        return TeacherAdapterContract(
            teacher_id="openvla",
            model_name=model_name,
            modality="action_semantics",
            advisory_only=True,
            available=available,
            metadata={
                "device": str(getattr(cfg, "device", "unknown")),
                "dtype": str(getattr(cfg, "dtype", "unknown")),
                "backend_status": backend_status,
                "execution_preconditions": preconditions.to_dict(),
            },
            provider_truth=build_teacher_provider_truth(
                provider_id="openvla",
                provider_name=model_name,
                available=available,
                backend_selected=str(backend_status.get("backend_selected", "")),
                fallback_mode=str(backend_status.get("failure_reason", "")),
                confidence=1.0 if available else 0.0,
                metadata={
                    "vision_backbone_selected": str(
                        backend_status.get("vision_backbone_selected", "")
                    ),
                    "backend_policy": str(backend_status.get("backend_policy", "")),
                    "failure_reason": str(backend_status.get("failure_reason", "")),
                },
            ),
        )

    def predict_action(self, image: Any, instruction: str) -> TeacherActionEnvelope:
        contract = self.describe_contract()
        try:
            payload = self.controller.predict_action(image, instruction)
        except Exception as exc:
            execution_preconditions = build_execution_preconditions(
                subject_id=contract.teacher_id,
                subject_kind="teacher_runtime_prediction",
                artifact_refs={"contract_id": contract.contract_id},
                signal_values={"teacher_available": False, "failure_mode": str(exc)},
                required_boolean_signals={"teacher_available": True},
                metadata={"instruction": instruction},
            )
            unavailable = TeacherActionEnvelope.unavailable(
                teacher_id=contract.teacher_id,
                model_name=contract.model_name,
                instruction=instruction,
                failure_mode=str(exc),
                metadata={
                    "contract_id": contract.contract_id,
                    "backend_selected": str(
                        contract.provider_truth.get("backend_selected", "")
                        or "unavailable"
                    ),
                    "execution_preconditions": execution_preconditions.to_dict(),
                },
            )
            return TeacherActionEnvelope(
                teacher_id=unavailable.teacher_id,
                model_name=unavailable.model_name,
                instruction=unavailable.instruction,
                available=unavailable.available,
                action=unavailable.action,
                confidence=unavailable.confidence,
                failure_mode=unavailable.failure_mode,
                semantic_tags=unavailable.semantic_tags,
                object_refs=unavailable.object_refs,
                affordance_hints=unavailable.affordance_hints,
                risk_hints=unavailable.risk_hints,
                provenance={
                    "contract_id": contract.contract_id,
                    "action_schema_id": contract.action_schema_id,
                },
                metadata=unavailable.metadata,
                provider_truth=build_teacher_provider_truth(
                    provider_id=contract.teacher_id,
                    provider_name=contract.model_name,
                    available=False,
                    backend_selected=str(
                        contract.provider_truth.get("backend_selected", "")
                        or "unavailable"
                    ),
                    fallback_mode=str(exc),
                    confidence=0.0,
                    metadata={
                        **dict(contract.provider_truth.get("metadata", {}) or {}),
                        "contract_id": contract.contract_id,
                    },
                ),
            )
        available = bool(payload.get("vla_available", False))
        execution_preconditions = build_execution_preconditions(
            subject_id=contract.teacher_id,
            subject_kind="teacher_runtime_prediction",
            artifact_refs={"contract_id": contract.contract_id},
            signal_values={
                "teacher_available": available,
                "teacher_real_backend": str(payload.get("backend_selected", ""))
                == "real",
                "confidence": float(payload.get("confidence", 0.0)),
            },
            required_boolean_signals={"teacher_available": True},
            metadata={"instruction": instruction},
        )
        semantic_bundle = infer_teacher_semantics(
            instruction=instruction,
            semantic_tags=payload.get("semantic_tags")
            if isinstance(payload, Mapping)
            else None,
            action=payload if isinstance(payload, Mapping) else None,
            metadata=payload if isinstance(payload, Mapping) else None,
        )
        return TeacherActionEnvelope(
            teacher_id=contract.teacher_id,
            model_name=contract.model_name,
            instruction=instruction,
            available=available,
            action=_float_mapping(payload),
            confidence=float(payload.get("confidence", 0.0)),
            failure_mode=str(
                payload.get(
                    "fallback_mode",
                    "teacher_available" if available else "teacher_unavailable",
                )
            ),
            semantic_tags=semantic_bundle["semantic_tags"],
            object_refs=semantic_bundle["object_refs"],
            affordance_hints=semantic_bundle["affordance_hints"],
            risk_hints=semantic_bundle["risk_hints"],
            provenance={
                "contract_id": contract.contract_id,
                "action_schema_id": contract.action_schema_id,
            },
            metadata={
                "available": available,
                "backend_selected": str(
                    payload.get(
                        "backend_selected", "real" if available else "unavailable"
                    )
                ),
                "backend_policy": str(payload.get("backend_policy", "")),
                "vision_backbone_selected": str(
                    payload.get("vision_backbone_selected", "")
                ),
                "vision_backbone_policy": str(
                    payload.get("vision_backbone_policy", "")
                ),
                "failure_reason": str(payload.get("failure_reason", "")),
                "semantic_summary": semantic_bundle,
                "execution_preconditions": execution_preconditions.to_dict(),
            },
            provider_truth=build_teacher_provider_truth(
                provider_id=contract.teacher_id,
                provider_name=contract.model_name,
                available=available,
                backend_selected=str(
                    payload.get(
                        "backend_selected", "real" if available else "unavailable"
                    )
                ),
                fallback_mode=str(
                    payload.get("failure_reason", payload.get("fallback_mode", ""))
                ),
                confidence=float(payload.get("confidence", 0.0)),
                metadata={
                    "vision_backbone_selected": str(
                        payload.get("vision_backbone_selected", "")
                    ),
                    "backend_policy": str(payload.get("backend_policy", "")),
                    "failure_reason": str(payload.get("failure_reason", "")),
                    "contract_id": contract.contract_id,
                },
            ),
        )


def save_teacher_adapter_contract_json(
    path: Path, contract: TeacherAdapterContract
) -> None:
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
        action_schema_id=str(
            payload.get("action_schema_id", "teacher_action_envelope_v1")
        ),
        metadata=_mapping(payload.get("metadata")),
        provider_truth=_mapping(payload.get("provider_truth")),
        version=str(payload.get("version", "teacher_adapter_contract_v1")),
    )


def save_teacher_action_envelope_json(
    path: Path, envelope: TeacherActionEnvelope
) -> None:
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

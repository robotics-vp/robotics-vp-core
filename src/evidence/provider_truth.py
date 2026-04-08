"""Canonical metadata for external provider availability, fallback, and grounding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from src.utils.json_safe import to_json_safe


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _normalize_class(value: Any, *, default: str) -> str:
    text = str(value or "").strip().lower()
    return text or default


def _derive_availability_class(*, available: bool, backend_selected: str, fallback_mode: str) -> str:
    backend = str(backend_selected or "").strip().lower()
    fallback = str(fallback_mode or "").strip().lower()
    if not available and fallback == "disabled":
        return "disabled"
    if backend in {"disabled", "unavailable"}:
        return backend
    if backend == "real":
        return "real_backend"
    if backend:
        return f"{backend}_backend"
    if fallback:
        return fallback
    return "available" if available else "unavailable"


@dataclass(frozen=True)
class ExternalProviderTruth:
    """Canonical metadata describing provider quality without promoting provider outputs to truth."""

    provider_id: str
    provider_kind: str
    provider_name: str = ""
    advisory_only: bool = True
    available: bool = False
    backend_selected: str = ""
    fallback_mode: str = ""
    availability_class: str = "unavailable"
    calibration_class: str = "unknown"
    grounding_class: str = "not_applicable"
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "external_provider_truth_v1"
    receipt_kind: str = "external_provider_truth_v1"
    authority_class: str = "canonical_metadata"
    decision_scope: str = "external_provider_status"
    reward_math_mutation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_kind": self.receipt_kind,
            "authority_class": self.authority_class,
            "decision_scope": self.decision_scope,
            "reward_math_mutation": bool(self.reward_math_mutation),
            "provider_id": self.provider_id,
            "provider_kind": self.provider_kind,
            "provider_name": self.provider_name,
            "advisory_only": bool(self.advisory_only),
            "available": bool(self.available),
            "backend_selected": self.backend_selected,
            "fallback_mode": self.fallback_mode,
            "availability_class": self.availability_class,
            "calibration_class": self.calibration_class,
            "grounding_class": self.grounding_class,
            "confidence": float(self.confidence),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalProviderTruth":
        return cls(
            provider_id=str(payload.get("provider_id", "")),
            provider_kind=str(payload.get("provider_kind", "")),
            provider_name=str(payload.get("provider_name", "")),
            advisory_only=bool(payload.get("advisory_only", True)),
            available=bool(payload.get("available", False)),
            backend_selected=str(payload.get("backend_selected", "")),
            fallback_mode=str(payload.get("fallback_mode", "")),
            availability_class=str(payload.get("availability_class", "unavailable")),
            calibration_class=str(payload.get("calibration_class", "unknown")),
            grounding_class=str(payload.get("grounding_class", "not_applicable")),
            confidence=float(payload.get("confidence", 0.0)),
            metadata=_mapping(payload.get("metadata")),
            schema_version=str(payload.get("schema_version", "external_provider_truth_v1")),
            receipt_kind=str(payload.get("receipt_kind", "external_provider_truth_v1")),
            authority_class=str(payload.get("authority_class", "canonical_metadata")),
            decision_scope=str(payload.get("decision_scope", "external_provider_status")),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
        )


def build_external_provider_truth(
    *,
    provider_id: str,
    provider_kind: str,
    provider_name: str = "",
    advisory_only: bool = True,
    available: bool = False,
    backend_selected: Any = "",
    fallback_mode: Any = "",
    availability_class: Any = "",
    calibration_class: Any = "",
    grounding_class: Any = "",
    confidence: Any = 0.0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    backend = str(backend_selected or "").strip()
    fallback = str(fallback_mode or "").strip()
    try:
        resolved_confidence = float(confidence)
    except Exception:
        resolved_confidence = 0.0
    truth = ExternalProviderTruth(
        provider_id=str(provider_id),
        provider_kind=str(provider_kind),
        provider_name=str(provider_name),
        advisory_only=bool(advisory_only),
        available=bool(available),
        backend_selected=backend,
        fallback_mode=fallback,
        availability_class=_normalize_class(
            availability_class,
            default=_derive_availability_class(
                available=bool(available),
                backend_selected=backend,
                fallback_mode=fallback,
            ),
        ),
        calibration_class=_normalize_class(calibration_class, default="unknown"),
        grounding_class=_normalize_class(grounding_class, default="not_applicable"),
        confidence=resolved_confidence,
        metadata=_mapping(metadata),
    )
    return truth.to_dict()


def coerce_external_provider_truth(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return ExternalProviderTruth.from_dict(payload).to_dict()


__all__ = [
    "ExternalProviderTruth",
    "build_external_provider_truth",
    "coerce_external_provider_truth",
]

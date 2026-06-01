"""Typed Economic WM regime-broadcast substrate.

This module implements the low-bandwidth broadcast-conditioning interface from
the bio/neuro doctrine. It is explicitly advisory and does not grant Economic
WM control over lower-WM truth, reward math, or live policy execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.scaffold import AllocationEnvelope, EconomicState

DENIED_REGIME_BROADCAST_AUTHORITIES = (
    "high_bandwidth_control",
    "lower_wm_truth_redefinition",
    "live_policy_control",
    "reward_math_mutation",
    "promotion_eligible",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_dict(payload: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{sha256_json(_mapping(payload))[:16]}"


@dataclass(frozen=True)
class RegimeBroadcast:
    """Small typed signal that conditions lower-WM operating posture."""

    broadcast_id: str
    source_state_id: str
    regime_id: str
    regime_class: str
    posture_settings: dict[str, str] = field(default_factory=dict)
    numeric_modulation: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    persistence_annotation: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    denied_authorities: list[str] = field(
        default_factory=lambda: list(DENIED_REGIME_BROADCAST_AUTHORITIES)
    )
    authority_class: str = "advisory_conditioning_only"
    high_bandwidth_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    version: str = "regime_broadcast_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "broadcast_id": self.broadcast_id,
            "source_state_id": self.source_state_id,
            "regime_id": self.regime_id,
            "regime_class": self.regime_class,
            "posture_settings": {str(k): str(v) for k, v in self.posture_settings.items()},
            "numeric_modulation": _float_dict(self.numeric_modulation),
            "confidence": max(0.0, min(1.0, float(self.confidence))),
            "persistence_annotation": _mapping(self.persistence_annotation),
            "provenance": _mapping(self.provenance),
            "denied_authorities": list(self.denied_authorities),
            "authority_class": self.authority_class,
            "high_bandwidth_control": bool(self.high_bandwidth_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "version": self.version,
        }


@dataclass(frozen=True)
class RegimeAcknowledgmentReceipt:
    """Downstream WM receipt for accepting, ignoring, or degrading a broadcast."""

    receipt_id: str
    broadcast_id: str
    wm_id: str
    adaptation_status: str
    accepted_posture_settings: dict[str, str] = field(default_factory=dict)
    ignored_settings: list[str] = field(default_factory=list)
    non_compliance_reasons: list[str] = field(default_factory=list)
    local_authority_preserved: bool = True
    lower_wm_truth_redefined: bool = False
    authority_class: str = "acknowledgment_only"
    promotion_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "regime_acknowledgment_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "broadcast_id": self.broadcast_id,
            "wm_id": self.wm_id,
            "adaptation_status": self.adaptation_status,
            "accepted_posture_settings": {
                str(k): str(v) for k, v in self.accepted_posture_settings.items()
            },
            "ignored_settings": [str(item) for item in self.ignored_settings],
            "non_compliance_reasons": [
                str(item) for item in self.non_compliance_reasons
            ],
            "local_authority_preserved": bool(self.local_authority_preserved),
            "lower_wm_truth_redefined": bool(self.lower_wm_truth_redefined),
            "authority_class": self.authority_class,
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }


def build_regime_broadcast(
    economic_state: EconomicState,
    allocation_envelope: AllocationEnvelope | None = None,
) -> RegimeBroadcast:
    resource = _float_dict(economic_state.resource_reservoirs)
    dissipation = _float_dict(economic_state.dissipation_fields)
    gpu_available = resource.get("training_gpu_budget_available", 0.0) > 0.0
    provider_available = resource.get("provider_runtime_capacity_available", 0.0) > 0.0
    promotion_friction = dissipation.get("promotion_friction", 1.0)
    readiness = (
        allocation_envelope.readiness_class
        if allocation_envelope is not None
        else economic_state.regime
    )
    posture_settings = {
        "trust_posture": "conservative",
        "compute_posture": "compute_scarce" if not gpu_available else "compute_available",
        "exploration_posture": "information_seeking"
        if provider_available
        else "manifest_preparation",
        "training_posture": "training_closed" if not gpu_available else "training_candidate",
        "energy_posture": "conserve",
        "degraded_mode": "allowed",
    }
    if promotion_friction >= 0.5:
        posture_settings["trust_posture"] = "promotion_blocked_conservative"
    numeric_modulation = {
        "max_shadow_work_order_priority": 0.5 if not provider_available else 0.75,
        "active_sensing_value_prior": 0.25 if not provider_available else 0.5,
        "training_open_fraction": 0.0 if not gpu_available else 0.25,
    }
    payload = {
        "state_id": economic_state.state_id,
        "regime": economic_state.regime,
        "readiness": readiness,
        "posture": posture_settings,
    }
    return RegimeBroadcast(
        broadcast_id=_stable_id("regime_broadcast", payload),
        source_state_id=economic_state.state_id,
        regime_id=f"economic::{economic_state.regime}",
        regime_class=readiness,
        posture_settings=posture_settings,
        numeric_modulation=numeric_modulation,
        confidence=economic_state.confidence,
        persistence_annotation={
            "rate_class": "slow_governance",
            "default_ttl_s": 60.0,
            "adiabatic_separation": True,
        },
        provenance={
            "economic_state_id": economic_state.state_id,
            "allocation_envelope_id": allocation_envelope.envelope_id
            if allocation_envelope is not None
            else "",
        },
    )


def build_regime_acknowledgment(
    broadcast: RegimeBroadcast | Mapping[str, Any],
    *,
    wm_id: str,
    accepted_settings: Optional[Mapping[str, Any]] = None,
    non_compliance_reasons: Optional[list[str]] = None,
) -> RegimeAcknowledgmentReceipt:
    payload = broadcast.to_dict() if isinstance(broadcast, RegimeBroadcast) else _mapping(broadcast)
    posture = {str(k): str(v) for k, v in dict(payload.get("posture_settings", {})).items()}
    accepted = {
        str(k): str(v)
        for k, v in dict(accepted_settings or posture).items()
        if str(k) in posture
    }
    ignored = sorted(set(posture) - set(accepted))
    reasons = [str(item) for item in list(non_compliance_reasons or [])]
    if ignored and not reasons:
        reasons.append("wm_declined_some_posture_settings")
    status = "accepted_with_local_bounds" if not ignored else "partially_accepted"
    return RegimeAcknowledgmentReceipt(
        receipt_id=_stable_id(
            "regime_acknowledgment_receipt",
            {
                "broadcast_id": payload.get("broadcast_id", ""),
                "wm_id": wm_id,
                "accepted": accepted,
                "ignored": ignored,
            },
        ),
        broadcast_id=str(payload.get("broadcast_id", "")),
        wm_id=wm_id,
        adaptation_status=status,
        accepted_posture_settings=accepted,
        ignored_settings=ignored,
        non_compliance_reasons=reasons,
        metadata={"broadcast_conditioning_only": True},
    )


__all__ = [
    "DENIED_REGIME_BROADCAST_AUTHORITIES",
    "RegimeAcknowledgmentReceipt",
    "RegimeBroadcast",
    "build_regime_acknowledgment",
    "build_regime_broadcast",
]

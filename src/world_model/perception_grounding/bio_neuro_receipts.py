"""Perception-side bio/neuro-inspired receipts.

Perception consumes Embodiment-owned expectations and emits typed comparison
receipts. These receipts are diagnostic/advisory only and preserve Perception
as the owner of observed scene truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .common import clip01, mapping, stable_id, strings


def _float_dict(payload: Optional[Mapping[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _sum_abs(payload: Mapping[str, Any]) -> float:
    total = 0.0
    for value in payload.values():
        try:
            total += abs(float(value))
        except Exception:
            continue
    return total


@dataclass(frozen=True)
class SelfDisturbanceReceipt:
    """Perception comparison of expected self-motion vs observed change."""

    receipt_id: str
    expectation_id: str
    perception_state_id: str
    embodiment_state_id: str
    mismatch_magnitude: float = 0.0
    attribution: str = "ambiguous"
    temporal_alignment_quality: float = 0.0
    self_caused_confidence: float = 0.0
    external_change_confidence: float = 0.0
    expected_observed_delta: dict[str, float] = field(default_factory=dict)
    observed_delta: dict[str, float] = field(default_factory=dict)
    missing_evidence: list[str] = field(default_factory=list)
    authority_level: str = "none"
    promotion_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "self_disturbance_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "expectation_id": self.expectation_id,
            "perception_state_id": self.perception_state_id,
            "embodiment_state_id": self.embodiment_state_id,
            "mismatch_magnitude": clip01(self.mismatch_magnitude),
            "attribution": self.attribution,
            "temporal_alignment_quality": clip01(self.temporal_alignment_quality),
            "self_caused_confidence": clip01(self.self_caused_confidence),
            "external_change_confidence": clip01(self.external_change_confidence),
            "expected_observed_delta": _float_dict(self.expected_observed_delta),
            "observed_delta": _float_dict(self.observed_delta),
            "missing_evidence": strings(self.missing_evidence),
            "authority_level": self.authority_level,
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class ActiveSensingReceipt:
    """Outcome receipt for a bounded active-sensing action."""

    receipt_id: str
    proposal_id: str
    perception_state_id: str
    action_type: str
    outcome_status: str = "not_executed"
    expected_information_gain: float = 0.0
    actual_information_gain: float = 0.0
    cost_incurred: dict[str, float] = field(default_factory=dict)
    uncertainty_before: dict[str, float] = field(default_factory=dict)
    uncertainty_after: dict[str, float] = field(default_factory=dict)
    perception_state_delta: dict[str, float] = field(default_factory=dict)
    missing_evidence: list[str] = field(default_factory=list)
    authority_level: str = "none"
    promotion_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "active_sensing_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "proposal_id": self.proposal_id,
            "perception_state_id": self.perception_state_id,
            "action_type": self.action_type,
            "outcome_status": self.outcome_status,
            "expected_information_gain": clip01(self.expected_information_gain),
            "actual_information_gain": clip01(self.actual_information_gain),
            "cost_incurred": _float_dict(self.cost_incurred),
            "uncertainty_before": _float_dict(self.uncertainty_before),
            "uncertainty_after": _float_dict(self.uncertainty_after),
            "perception_state_delta": _float_dict(self.perception_state_delta),
            "missing_evidence": strings(self.missing_evidence),
            "authority_level": self.authority_level,
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


def build_self_disturbance_receipt(
    expectation: Mapping[str, Any],
    *,
    perception_state_id: str,
    observed_delta: Optional[Mapping[str, Any]] = None,
    temporal_alignment_quality: float = 0.0,
    external_change_score: float = 0.0,
) -> SelfDisturbanceReceipt:
    expected_delta = _float_dict(expectation.get("predicted_body_delta", {}))
    observed = _float_dict(observed_delta or {})
    missing: list[str] = []
    if not expected_delta:
        missing.append("self_motion_expectation_delta")
    if not observed:
        missing.append("observed_perception_delta")
    expected_mag = _sum_abs(expected_delta)
    observed_mag = _sum_abs(observed)
    denominator = max(1.0, expected_mag, observed_mag)
    mismatch = abs(expected_mag - observed_mag) / denominator
    alignment = clip01(temporal_alignment_quality)
    external_confidence = clip01(max(external_change_score, mismatch) * (1.0 - alignment * 0.25))
    self_confidence = clip01((1.0 - mismatch) * alignment)
    if self_confidence >= 0.6:
        attribution = "self_caused"
    elif external_confidence >= 0.6:
        attribution = "external_or_unmodeled"
    else:
        attribution = "ambiguous"

    expectation_id = str(expectation.get("expectation_id", ""))
    embodiment_state_id = str(expectation.get("embodiment_state_id", ""))
    return SelfDisturbanceReceipt(
        receipt_id=stable_id(
            "self_disturbance_receipt",
            expectation_id,
            perception_state_id,
            str(observed),
        ),
        expectation_id=expectation_id,
        perception_state_id=perception_state_id,
        embodiment_state_id=embodiment_state_id,
        mismatch_magnitude=mismatch,
        attribution=attribution,
        temporal_alignment_quality=alignment,
        self_caused_confidence=self_confidence,
        external_change_confidence=external_confidence,
        expected_observed_delta=expected_delta,
        observed_delta=observed,
        missing_evidence=missing,
        metadata={
            "perception_truth_owner": True,
            "embodiment_expectation_is_typed_input": True,
        },
    )


def build_active_sensing_receipt(
    proposal: Mapping[str, Any],
    *,
    perception_state_id: str,
    uncertainty_before: Mapping[str, Any],
    uncertainty_after: Optional[Mapping[str, Any]] = None,
    cost_incurred: Optional[Mapping[str, Any]] = None,
    executed: bool = False,
) -> ActiveSensingReceipt:
    before = _float_dict(uncertainty_before)
    after = _float_dict(uncertainty_after or {})
    missing: list[str] = []
    if not after:
        missing.append("post_action_uncertainty")
    before_mag = _sum_abs(before)
    after_mag = _sum_abs(after)
    gain = max(0.0, before_mag - after_mag) / max(1.0, before_mag)
    if not executed:
        outcome_status = "not_executed"
    elif gain > 0.05:
        outcome_status = "information_gain_observed"
    else:
        outcome_status = "no_measured_information_gain"
    return ActiveSensingReceipt(
        receipt_id=stable_id(
            "active_sensing_receipt",
            str(proposal.get("proposal_id", "")),
            perception_state_id,
            str(after),
        ),
        proposal_id=str(proposal.get("proposal_id", "")),
        perception_state_id=perception_state_id,
        action_type=str(proposal.get("action_type", "unknown")),
        outcome_status=outcome_status,
        expected_information_gain=float(proposal.get("expected_information_gain", 0.0)),
        actual_information_gain=gain,
        cost_incurred=_float_dict(cost_incurred or proposal.get("cost_vector", {})),
        uncertainty_before=before,
        uncertainty_after=after,
        perception_state_delta={
            "uncertainty_mass_delta": before_mag - after_mag,
            "information_gain_fraction": gain,
        },
        missing_evidence=missing,
        metadata={
            "generic_exploration_bonus": False,
            "economic_value_of_information_not_applied": True,
        },
    )


__all__ = [
    "ActiveSensingReceipt",
    "SelfDisturbanceReceipt",
    "build_active_sensing_receipt",
    "build_self_disturbance_receipt",
]

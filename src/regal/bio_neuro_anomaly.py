"""Bio/neuro-inspired local anomaly receipts for regal governance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe


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


def _clip01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


@dataclass(frozen=True)
class AnomalySuspicionReceipt:
    """Domain-local anomaly receipt with bounded confidence and abstention."""

    receipt_id: str
    domain: str
    anomaly_type: str
    severity: float = 0.0
    confidence: float = 0.0
    evidence_status: str = "insufficient_evidence"
    abstained: bool = True
    evidence_scores: dict[str, float] = field(default_factory=dict)
    evidence_refs: dict[str, Any] = field(default_factory=dict)
    recommended_action_class: str = "observe"
    authority_level: str = "advisory"
    live_control_allowed: bool = False
    promotion_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "anomaly_suspicion_receipt_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "domain": self.domain,
            "anomaly_type": self.anomaly_type,
            "severity": _clip01(self.severity),
            "confidence": _clip01(self.confidence),
            "evidence_status": self.evidence_status,
            "abstained": bool(self.abstained),
            "evidence_scores": _float_dict(self.evidence_scores),
            "evidence_refs": _mapping(self.evidence_refs),
            "recommended_action_class": self.recommended_action_class,
            "authority_level": self.authority_level,
            "live_control_allowed": bool(self.live_control_allowed),
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }


@dataclass(frozen=True)
class GovernanceEscalationEvent:
    """Composition event over domain-local anomaly receipts."""

    event_id: str
    source_domains: list[str] = field(default_factory=list)
    triggering_receipt_ids: list[str] = field(default_factory=list)
    escalation_level: str = "none"
    recommended_action_class: str = "observe"
    escalation_status: str = "advisory_only"
    requires_human_review: bool = False
    authority_level: str = "advisory"
    live_control_allowed: bool = False
    promotion_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "governance_escalation_event_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_domains": [str(item) for item in self.source_domains],
            "triggering_receipt_ids": [
                str(item) for item in self.triggering_receipt_ids
            ],
            "escalation_level": self.escalation_level,
            "recommended_action_class": self.recommended_action_class,
            "escalation_status": self.escalation_status,
            "requires_human_review": bool(self.requires_human_review),
            "authority_level": self.authority_level,
            "live_control_allowed": bool(self.live_control_allowed),
            "promotion_eligible": bool(self.promotion_eligible),
            "metadata": _mapping(self.metadata),
            "version": self.version,
        }


def build_anomaly_suspicion_receipt(
    *,
    domain: str,
    anomaly_type: str,
    evidence_scores: Optional[Mapping[str, Any]] = None,
    evidence_refs: Optional[Mapping[str, Any]] = None,
    threshold: float = 0.65,
    abstain_below: float = 0.2,
) -> AnomalySuspicionReceipt:
    scores = _float_dict(evidence_scores or {})
    severity = max((_clip01(value) for value in scores.values()), default=0.0)
    confidence = sum(_clip01(value) for value in scores.values()) / max(len(scores), 1)
    if not scores or confidence < abstain_below:
        status = "insufficient_evidence"
        abstained = True
        action = "observe"
    elif severity >= threshold:
        status = "suspicion_supported"
        abstained = False
        action = "escalate_review"
    else:
        status = "no_suspicion_supported"
        abstained = False
        action = "continue_monitoring"
    return AnomalySuspicionReceipt(
        receipt_id=_stable_id(
            "anomaly_suspicion_receipt",
            {
                "domain": domain,
                "anomaly_type": anomaly_type,
                "scores": scores,
                "status": status,
            },
        ),
        domain=domain,
        anomaly_type=anomaly_type,
        severity=severity,
        confidence=confidence,
        evidence_status=status,
        abstained=abstained,
        evidence_scores=scores,
        evidence_refs=_mapping(evidence_refs),
        recommended_action_class=action,
        metadata={
            "bounded_confidence": True,
            "threshold": threshold,
            "abstain_below": abstain_below,
        },
    )


def build_governance_escalation_event(
    receipts: Sequence[AnomalySuspicionReceipt | Mapping[str, Any]],
) -> GovernanceEscalationEvent:
    payloads = [
        receipt.to_dict() if isinstance(receipt, AnomalySuspicionReceipt) else _mapping(receipt)
        for receipt in receipts
    ]
    supported = [
        payload
        for payload in payloads
        if str(payload.get("evidence_status", "")) == "suspicion_supported"
        and not bool(payload.get("abstained", True))
    ]
    max_severity = max(
        (_clip01(payload.get("severity", 0.0)) for payload in supported),
        default=0.0,
    )
    if max_severity >= 0.85 or len(supported) >= 2:
        level = "operator_review"
        action = "pause_or_reroute_for_review"
        human = True
    elif supported:
        level = "domain_review"
        action = "increase_monitoring_and_capture_context"
        human = False
    else:
        level = "none"
        action = "observe"
        human = False
    return GovernanceEscalationEvent(
        event_id=_stable_id(
            "governance_escalation_event",
            {
                "receipt_ids": [payload.get("receipt_id", "") for payload in payloads],
                "level": level,
            },
        ),
        source_domains=sorted({str(payload.get("domain", "")) for payload in supported}),
        triggering_receipt_ids=[
            str(payload.get("receipt_id", "")) for payload in supported
        ],
        escalation_level=level,
        recommended_action_class=action,
        requires_human_review=human,
        metadata={
            "domain_local_first": True,
            "meta_regal_composition_trained": False,
        },
    )


__all__ = [
    "AnomalySuspicionReceipt",
    "GovernanceEscalationEvent",
    "build_anomaly_suspicion_receipt",
    "build_governance_escalation_event",
]

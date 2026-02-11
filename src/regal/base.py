"""Lightweight meta-regal base abstractions for artifact gating."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class RegalDecision(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REROUTE = "REROUTE"
    REPAIR = "REPAIR"
    REGENERATE = "REGENERATE"


@dataclass
class RegalReport:
    node_id: str
    decision: RegalDecision
    reason_codes: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    artifact_refs: Dict[str, str] = field(default_factory=dict)
    recommended_action: Optional[str] = None
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "decision": self.decision.value,
            "reason_codes": list(self.reason_codes),
            "details": dict(self.details),
            "artifact_refs": dict(self.artifact_refs),
            "recommended_action": self.recommended_action,
            "confidence": float(self.confidence),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RegalReport":
        return cls(
            node_id=str(payload.get("node_id", "unknown_regal")),
            decision=RegalDecision(str(payload.get("decision", RegalDecision.ALLOW.value))),
            reason_codes=list(payload.get("reason_codes", []) or []),
            details=dict(payload.get("details", {}) or {}),
            artifact_refs=dict(payload.get("artifact_refs", {}) or {}),
            recommended_action=payload.get("recommended_action"),
            confidence=float(payload.get("confidence", 1.0)),
            timestamp=str(payload.get("timestamp", datetime.now(timezone.utc).isoformat())),
        )


class RegalNode:
    """Base interface for additive meta-regal artifact gates."""

    node_id: str = "regal_node"

    def evaluate(self, context: Mapping[str, Any]) -> RegalReport:
        raise NotImplementedError

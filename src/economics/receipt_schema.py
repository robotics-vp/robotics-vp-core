"""Deployment-ready receipt and invoice-like economic event schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class PricingAcceptanceLabel:
    """Synthetic or real pricing acceptance/rejection signal."""

    schema_version: str
    receipt_id: str
    run_id: str
    episode_id: str
    quoted_rate: float
    accepted_rate: float
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "quoted_rate": float(self.quoted_rate),
            "accepted_rate": float(self.accepted_rate),
            "accepted": bool(self.accepted),
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PricingAcceptanceLabel":
        return cls(
            schema_version=str(payload.get("schema_version", "pricing_acceptance_label_v1")),
            receipt_id=str(payload.get("receipt_id", "")),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            quoted_rate=float(payload.get("quoted_rate", 0.0)),
            accepted_rate=float(payload.get("accepted_rate", 0.0)),
            accepted=bool(payload.get("accepted", False)),
            reasons=[str(value) for value in payload.get("reasons", []) or []],
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class DeploymentReceiptRecord:
    """Sparse, auditable deployment-style receipt for current shadow runs and future live deployments."""

    schema_version: str
    run_id: str
    episode_id: str
    deployment_id: str
    source_domain: str
    objective_profile_id: str
    predicted_value: float
    realized_value: float
    quoted_rate: float
    billed_rate: float
    pricing_acceptance: PricingAcceptanceLabel
    realized_reward: Optional[float] = None
    task_success: Optional[bool] = None
    objective_satisfied: Optional[bool] = None
    incident_events: list[str] = field(default_factory=list)
    human_review_label: Optional[str] = None
    override_label: Optional[str] = None
    adaptation_outcome_ref: Optional[str] = None
    datapack_label_ref: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def record_id(self) -> str:
        return sha256_json(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "episode_id": self.episode_id,
                "deployment_id": self.deployment_id,
                "objective_profile_id": self.objective_profile_id,
            }
        )[:20]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "deployment_id": self.deployment_id,
            "source_domain": self.source_domain,
            "objective_profile_id": self.objective_profile_id,
            "predicted_value": float(self.predicted_value),
            "realized_value": float(self.realized_value),
            "quoted_rate": float(self.quoted_rate),
            "billed_rate": float(self.billed_rate),
            "pricing_acceptance": self.pricing_acceptance.to_dict(),
            "realized_reward": self.realized_reward,
            "task_success": self.task_success,
            "objective_satisfied": self.objective_satisfied,
            "incident_events": list(self.incident_events),
            "human_review_label": self.human_review_label,
            "override_label": self.override_label,
            "adaptation_outcome_ref": self.adaptation_outcome_ref,
            "datapack_label_ref": self.datapack_label_ref,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeploymentReceiptRecord":
        return cls(
            schema_version=str(payload.get("schema_version", "deployment_receipt_record_v1")),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            deployment_id=str(payload.get("deployment_id", "")),
            source_domain=str(payload.get("source_domain", "")),
            objective_profile_id=str(payload.get("objective_profile_id", "")),
            predicted_value=float(payload.get("predicted_value", 0.0)),
            realized_value=float(payload.get("realized_value", 0.0)),
            quoted_rate=float(payload.get("quoted_rate", 0.0)),
            billed_rate=float(payload.get("billed_rate", 0.0)),
            pricing_acceptance=PricingAcceptanceLabel.from_dict(payload.get("pricing_acceptance", {}) or {}),
            realized_reward=(
                float(payload.get("realized_reward"))
                if payload.get("realized_reward") is not None
                else None
            ),
            task_success=payload.get("task_success"),
            objective_satisfied=payload.get("objective_satisfied"),
            incident_events=[str(value) for value in payload.get("incident_events", []) or []],
            human_review_label=payload.get("human_review_label"),
            override_label=payload.get("override_label"),
            adaptation_outcome_ref=payload.get("adaptation_outcome_ref"),
            datapack_label_ref=payload.get("datapack_label_ref"),
            provenance=dict(payload.get("provenance", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

"""Future-ready deployment and adaptation outcome labels for shadow and real receipts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class DeploymentOutcomeLabel:
    """Observed deployment outcome tied to a run or receipt."""

    schema_version: str
    run_id: str
    episode_id: str
    source_domain: str
    deployment_id: str
    objective_profile_id: str
    predicted_value: float
    realized_value: float
    pricing_accepted: bool
    task_success: Optional[bool] = None
    objective_satisfied: Optional[bool] = None
    realized_reward: Optional[float] = None
    failure_events: List[str] = field(default_factory=list)
    risk_events: List[str] = field(default_factory=list)
    incident_events: List[str] = field(default_factory=list)
    human_review_label: Optional[str] = None
    override_label: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def label_id(self) -> str:
        return sha256_json(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "episode_id": self.episode_id,
                "deployment_id": self.deployment_id,
                "objective_profile_id": self.objective_profile_id,
            }
        )[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label_id": self.label_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "source_domain": self.source_domain,
            "deployment_id": self.deployment_id,
            "objective_profile_id": self.objective_profile_id,
            "predicted_value": float(self.predicted_value),
            "realized_value": float(self.realized_value),
            "pricing_accepted": bool(self.pricing_accepted),
            "task_success": self.task_success,
            "objective_satisfied": self.objective_satisfied,
            "realized_reward": self.realized_reward,
            "failure_events": list(self.failure_events),
            "risk_events": list(self.risk_events),
            "incident_events": list(self.incident_events),
            "human_review_label": self.human_review_label,
            "override_label": self.override_label,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DeploymentOutcomeLabel":
        return cls(
            schema_version=str(payload.get("schema_version", "deployment_outcome_label_v1")),
            run_id=str(payload.get("run_id", "")),
            episode_id=str(payload.get("episode_id", "")),
            source_domain=str(payload.get("source_domain", "")),
            deployment_id=str(payload.get("deployment_id", "")),
            objective_profile_id=str(payload.get("objective_profile_id", "")),
            predicted_value=float(payload.get("predicted_value", 0.0)),
            realized_value=float(payload.get("realized_value", 0.0)),
            pricing_accepted=bool(payload.get("pricing_accepted", False)),
            task_success=payload.get("task_success"),
            objective_satisfied=payload.get("objective_satisfied"),
            realized_reward=(
                float(payload.get("realized_reward"))
                if payload.get("realized_reward") is not None
                else None
            ),
            failure_events=[str(value) for value in payload.get("failure_events", []) or []],
            risk_events=[str(value) for value in payload.get("risk_events", []) or []],
            incident_events=[str(value) for value in payload.get("incident_events", []) or []],
            human_review_label=payload.get("human_review_label"),
            override_label=payload.get("override_label"),
            provenance=dict(payload.get("provenance", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class AdaptationOutcomeLabel:
    """Observed outcome of an adaptation or inferential training attempt."""

    schema_version: str
    run_id: str
    adaptation_id: str
    source_domain: str
    recommended_mode: str
    realized_mode: str
    expected_gain: float
    realized_gain: float
    compute_cost: float
    risk_cost: float
    review_required: bool
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def label_id(self) -> str:
        return sha256_json({"run_id": self.run_id, "adaptation_id": self.adaptation_id, "schema_version": self.schema_version})[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label_id": self.label_id,
            "run_id": self.run_id,
            "adaptation_id": self.adaptation_id,
            "source_domain": self.source_domain,
            "recommended_mode": self.recommended_mode,
            "realized_mode": self.realized_mode,
            "expected_gain": float(self.expected_gain),
            "realized_gain": float(self.realized_gain),
            "compute_cost": float(self.compute_cost),
            "risk_cost": float(self.risk_cost),
            "review_required": bool(self.review_required),
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdaptationOutcomeLabel":
        return cls(
            schema_version=str(payload.get("schema_version", "adaptation_outcome_label_v1")),
            run_id=str(payload.get("run_id", "")),
            adaptation_id=str(payload.get("adaptation_id", "")),
            source_domain=str(payload.get("source_domain", "")),
            recommended_mode=str(payload.get("recommended_mode", "no_op")),
            realized_mode=str(payload.get("realized_mode", "no_op")),
            expected_gain=float(payload.get("expected_gain", 0.0)),
            realized_gain=float(payload.get("realized_gain", 0.0)),
            compute_cost=float(payload.get("compute_cost", 0.0)),
            risk_cost=float(payload.get("risk_cost", 0.0)),
            review_required=bool(payload.get("review_required", False)),
            provenance=dict(payload.get("provenance", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class DatapackContributionLabel:
    """Outcome label for post-deployment datapack contribution quality."""

    schema_version: str
    datapack_id: str
    run_id: str
    source_domain: str
    marginal_frontier_gain_predicted: float
    marginal_frontier_gain_realized: float
    data_share_credit_predicted: float
    data_share_credit_realized: float
    downweight_recommended: bool
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def label_id(self) -> str:
        return sha256_json({"schema_version": self.schema_version, "datapack_id": self.datapack_id, "run_id": self.run_id})[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label_id": self.label_id,
            "datapack_id": self.datapack_id,
            "run_id": self.run_id,
            "source_domain": self.source_domain,
            "marginal_frontier_gain_predicted": float(self.marginal_frontier_gain_predicted),
            "marginal_frontier_gain_realized": float(self.marginal_frontier_gain_realized),
            "data_share_credit_predicted": float(self.data_share_credit_predicted),
            "data_share_credit_realized": float(self.data_share_credit_realized),
            "downweight_recommended": bool(self.downweight_recommended),
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatapackContributionLabel":
        return cls(
            schema_version=str(payload.get("schema_version", "datapack_contribution_label_v1")),
            datapack_id=str(payload.get("datapack_id", "")),
            run_id=str(payload.get("run_id", "")),
            source_domain=str(payload.get("source_domain", "")),
            marginal_frontier_gain_predicted=float(payload.get("marginal_frontier_gain_predicted", 0.0)),
            marginal_frontier_gain_realized=float(payload.get("marginal_frontier_gain_realized", 0.0)),
            data_share_credit_predicted=float(payload.get("data_share_credit_predicted", 0.0)),
            data_share_credit_realized=float(payload.get("data_share_credit_realized", 0.0)),
            downweight_recommended=bool(payload.get("downweight_recommended", False)),
            provenance=dict(payload.get("provenance", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

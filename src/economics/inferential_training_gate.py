"""Economically gated inferential training decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from src.regality.promotion_policy import RegalMaturityStage, RegalPromotionPolicy
from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class InferentialTrainingCandidate:
    """Candidate adaptation or inferential training opportunity."""

    run_id: str
    episode_id: str
    objective_profile_id: str
    source_domain: str
    expected_value_gain: float
    compute_cost: float
    risk_cost: float
    uncertainty: float
    ood_score: float
    data_quality: float
    provenance_quality: float
    pricing_summary: Dict[str, Any]
    regal_statuses: Dict[str, str]
    regal_scores: Dict[str, float]
    replay_policy_uncertainty: float
    learned_data_value: float
    expected_adaptation_benefit: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "objective_profile_id": self.objective_profile_id,
            "source_domain": self.source_domain,
            "expected_value_gain": float(self.expected_value_gain),
            "compute_cost": float(self.compute_cost),
            "risk_cost": float(self.risk_cost),
            "uncertainty": float(self.uncertainty),
            "ood_score": float(self.ood_score),
            "data_quality": float(self.data_quality),
            "provenance_quality": float(self.provenance_quality),
            "pricing_summary": dict(self.pricing_summary),
            "regal_statuses": dict(self.regal_statuses),
            "regal_scores": dict(self.regal_scores),
            "replay_policy_uncertainty": float(self.replay_policy_uncertainty),
            "learned_data_value": float(self.learned_data_value),
            "expected_adaptation_benefit": float(self.expected_adaptation_benefit),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class InferentialTrainingDecision:
    """Capital-allocation output for inferential training or data collection."""

    decision: str
    expected_gain: float
    expected_cost: float
    expected_risk: float
    net_benefit: float
    allowed_budget: float
    recommended_training_mode: str
    reasons: list[str]
    artifact_summary: Dict[str, Any]

    @property
    def decision_id(self) -> str:
        return sha256_json(self._base_dict())[:18]

    def _base_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "expected_gain": float(self.expected_gain),
            "expected_cost": float(self.expected_cost),
            "expected_risk": float(self.expected_risk),
            "net_benefit": float(self.net_benefit),
            "allowed_budget": float(self.allowed_budget),
            "recommended_training_mode": self.recommended_training_mode,
            "reasons": list(self.reasons),
            "artifact_summary": dict(self.artifact_summary),
        }

    def to_dict(self) -> Dict[str, Any]:
        payload = self._base_dict()
        payload["decision_id"] = self.decision_id
        return payload


class InferentialTrainingGate:
    """Deterministic adaptation budget gate governed by econ and regal evidence."""

    def __init__(
        self,
        *,
        promotion_policy: RegalPromotionPolicy,
        min_net_benefit: float = 0.05,
        max_uncertainty_for_adapt: float = 0.45,
        max_ood_for_adapt: float = 0.5,
    ) -> None:
        self.promotion_policy = promotion_policy
        self.min_net_benefit = float(min_net_benefit)
        self.max_uncertainty_for_adapt = float(max_uncertainty_for_adapt)
        self.max_ood_for_adapt = float(max_ood_for_adapt)

    def evaluate(self, candidate: InferentialTrainingCandidate) -> InferentialTrainingDecision:
        reasons: list[str] = []
        expected_gain = float(candidate.expected_value_gain + candidate.expected_adaptation_benefit + 0.25 * candidate.learned_data_value)
        expected_cost = float(candidate.compute_cost)
        expected_risk = float(candidate.risk_cost + 0.5 * candidate.uncertainty + 0.5 * candidate.ood_score)
        net_benefit = expected_gain - expected_cost - expected_risk
        allowed_budget = max(0.0, net_benefit)

        integrity_failed = candidate.regal_statuses.get("objective_integrity_regal") == "fail"
        pricing_failed = candidate.regal_statuses.get("pricing_truth_regal") == "fail"
        reward_failed = candidate.regal_statuses.get("reward_safety_regal") == "fail"
        hard_gate_enabled = self.promotion_policy.node_stage("pricing_truth_regal") == RegalMaturityStage.NARROW_HARD_GATE

        if integrity_failed:
            reasons.append("objective_integrity_failure")
            return InferentialTrainingDecision(
                decision="require_review",
                expected_gain=expected_gain,
                expected_cost=expected_cost,
                expected_risk=expected_risk,
                net_benefit=net_benefit,
                allowed_budget=0.0,
                recommended_training_mode="no_training",
                reasons=reasons,
                artifact_summary={
                    "promotion_policy": self.promotion_policy.policy_name,
                    "hard_gate_enabled": hard_gate_enabled,
                    "run_id": candidate.run_id,
                    "episode_id": candidate.episode_id,
                },
            )

        if hard_gate_enabled and pricing_failed:
            reasons.append("pricing_truth_hard_gate")
            return InferentialTrainingDecision(
                decision="no_op",
                expected_gain=expected_gain,
                expected_cost=expected_cost,
                expected_risk=expected_risk,
                net_benefit=net_benefit,
                allowed_budget=0.0,
                recommended_training_mode="shadow_compare_only",
                reasons=reasons,
                artifact_summary={
                    "promotion_policy": self.promotion_policy.policy_name,
                    "hard_gate_enabled": hard_gate_enabled,
                    "run_id": candidate.run_id,
                    "episode_id": candidate.episode_id,
                },
            )

        if candidate.provenance_quality < 0.5 or candidate.data_quality < 0.5:
            reasons.append("collect_more_data_due_to_quality")
            return InferentialTrainingDecision(
                decision="collect_more_data",
                expected_gain=expected_gain,
                expected_cost=expected_cost,
                expected_risk=expected_risk,
                net_benefit=net_benefit,
                allowed_budget=allowed_budget,
                recommended_training_mode="behavior_cloning_refresh",
                reasons=reasons,
                artifact_summary={
                    "promotion_policy": self.promotion_policy.policy_name,
                    "run_id": candidate.run_id,
                    "episode_id": candidate.episode_id,
                },
            )

        if reward_failed:
            reasons.append("reward_safety_requires_review")
            return InferentialTrainingDecision(
                decision="require_review",
                expected_gain=expected_gain,
                expected_cost=expected_cost,
                expected_risk=expected_risk,
                net_benefit=net_benefit,
                allowed_budget=allowed_budget,
                recommended_training_mode="shadow_compare_only",
                reasons=reasons,
                artifact_summary={
                    "promotion_policy": self.promotion_policy.policy_name,
                    "run_id": candidate.run_id,
                    "episode_id": candidate.episode_id,
                },
            )

        if net_benefit < self.min_net_benefit:
            reasons.append("net_benefit_below_threshold")
            return InferentialTrainingDecision(
                decision="no_op",
                expected_gain=expected_gain,
                expected_cost=expected_cost,
                expected_risk=expected_risk,
                net_benefit=net_benefit,
                allowed_budget=allowed_budget,
                recommended_training_mode="no_training",
                reasons=reasons,
                artifact_summary={
                    "promotion_policy": self.promotion_policy.policy_name,
                    "run_id": candidate.run_id,
                    "episode_id": candidate.episode_id,
                },
            )

        if candidate.uncertainty > self.max_uncertainty_for_adapt or candidate.ood_score > self.max_ood_for_adapt:
            reasons.append("collect_more_data_due_to_uncertainty")
            return InferentialTrainingDecision(
                decision="collect_more_data",
                expected_gain=expected_gain,
                expected_cost=expected_cost,
                expected_risk=expected_risk,
                net_benefit=net_benefit,
                allowed_budget=allowed_budget,
                recommended_training_mode="behavior_cloning_refresh",
                reasons=reasons,
                artifact_summary={"promotion_policy": self.promotion_policy.policy_name},
            )

        reasons.append("adaptation_budget_admitted")
        return InferentialTrainingDecision(
            decision="adapt_now",
            expected_gain=expected_gain,
            expected_cost=expected_cost,
            expected_risk=expected_risk,
            net_benefit=net_benefit,
            allowed_budget=allowed_budget,
            recommended_training_mode="offline_td3_bc_shadow",
            reasons=reasons,
            artifact_summary={
                "promotion_policy": self.promotion_policy.policy_name,
                "pricing_confidence": float(candidate.pricing_summary.get("confidence", 0.0)),
                "run_id": candidate.run_id,
                "episode_id": candidate.episode_id,
            },
        )

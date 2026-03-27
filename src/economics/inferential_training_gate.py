"""Economically gated inferential training decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.economics.inferential_contract import (
    build_inferential_learnability_contract,
    coerce_inferential_learnability_contract,
)
from src.economics.inferential_reward import compile_inferential_reward
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
    frontier_gain: float = 0.0
    epiplexity_delta: float = 0.0
    epiplexity_confidence: float = 0.0
    transfer_score: float = 0.0
    governance_penalty: float = 0.0
    signal_yield_score: Optional[float] = None
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
            "frontier_gain": float(self.frontier_gain),
            "epiplexity_delta": float(self.epiplexity_delta),
            "epiplexity_confidence": float(self.epiplexity_confidence),
            "transfer_score": float(self.transfer_score),
            "governance_penalty": float(self.governance_penalty),
            "signal_yield_score": (
                float(self.signal_yield_score) if self.signal_yield_score is not None else None
            ),
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
        inferential_reward = compile_inferential_reward(
            expected_value_gain=candidate.expected_value_gain,
            expected_adaptation_benefit=candidate.expected_adaptation_benefit,
            learned_data_value=candidate.learned_data_value,
            compute_cost=candidate.compute_cost,
            risk_cost=candidate.risk_cost,
            uncertainty=candidate.uncertainty,
            ood_score=candidate.ood_score,
            data_quality=candidate.data_quality,
            provenance_quality=candidate.provenance_quality,
            frontier_gain=candidate.frontier_gain,
            epiplexity_delta=candidate.epiplexity_delta,
            epiplexity_confidence=candidate.epiplexity_confidence,
            transfer_score=candidate.transfer_score,
            governance_penalty=candidate.governance_penalty,
            signal_yield_override=candidate.signal_yield_score,
        )
        expected_gain = inferential_reward.expected_gain
        expected_cost = inferential_reward.expected_cost
        expected_risk = inferential_reward.expected_risk
        net_benefit = inferential_reward.net_benefit
        allowed_budget = max(0.0, net_benefit)

        integrity_failed = candidate.regal_statuses.get("objective_integrity_regal") == "fail"
        pricing_failed = candidate.regal_statuses.get("pricing_truth_regal") == "fail"
        reward_failed = candidate.regal_statuses.get("reward_safety_regal") == "fail"
        hard_gate_enabled = self.promotion_policy.node_stage("pricing_truth_regal") == RegalMaturityStage.NARROW_HARD_GATE
        artifact_summary = {
            "promotion_policy": self.promotion_policy.policy_name,
            "hard_gate_enabled": hard_gate_enabled,
            "pricing_confidence": float(candidate.pricing_summary.get("confidence", 0.0)),
            "run_id": candidate.run_id,
            "episode_id": candidate.episode_id,
            "inferential_reward": inferential_reward.to_dict(),
        }
        learnability_contract = coerce_inferential_learnability_contract(
            candidate.metadata.get("inferential_learnability_contract")
        )
        if learnability_contract is None:
            inferential_metadata = dict(candidate.metadata.get("inferential_metadata", {}) or {})
            learnability_contract = build_inferential_learnability_contract(
                subject_id=candidate.episode_id,
                subject_kind="replay_episode",
                datapack_id=str(
                    candidate.metadata.get("datapack_id")
                    or inferential_metadata.get("datapack_id")
                    or candidate.episode_id
                ),
                frontier_gain=candidate.frontier_gain,
                epiplexity_delta=candidate.epiplexity_delta,
                epiplexity_confidence=candidate.epiplexity_confidence,
                transfer_score=candidate.transfer_score,
                data_quality=candidate.data_quality,
                provenance_quality=candidate.provenance_quality,
                trust_score=float(candidate.pricing_summary.get("confidence", 0.0) or 0.5),
                overlay_joined=bool(
                    candidate.metadata.get("epiplexity_overlay_joined")
                    or inferential_metadata.get("overlay_joined")
                ),
                benchmark_eligible=bool(inferential_metadata.get("benchmark_eligible", False)),
                semantic_grounding_non_heuristic=bool(
                    inferential_metadata.get("semantic_grounding_non_heuristic", False)
                ),
                promotion_trace_complete=bool(inferential_metadata.get("promotion_trace_complete", False)),
                budget_settlement_live=bool(inferential_metadata.get("budget_settlement_live", False)),
                summary_present=inferential_metadata.get("summary_present"),
                signal_yield=inferential_reward.signal_yield,
                metadata={
                    "source_domain": candidate.source_domain,
                    "objective_profile_id": candidate.objective_profile_id,
                    **inferential_metadata,
                },
            )
        artifact_summary["inferential_learnability_contract"] = learnability_contract.to_dict()

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
                artifact_summary=dict(artifact_summary),
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
                artifact_summary=dict(artifact_summary),
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
                artifact_summary=dict(artifact_summary),
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
                artifact_summary=dict(artifact_summary),
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
                artifact_summary=dict(artifact_summary),
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
                artifact_summary=dict(artifact_summary),
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
            artifact_summary=dict(artifact_summary),
        )

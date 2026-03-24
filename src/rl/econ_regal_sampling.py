"""Advisory replay sampling modifiers derived from econ, regality, and contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

from src.regality.promotion_policy import RegalPromotionPolicy


@dataclass(frozen=True)
class ReplaySamplingRecommendation:
    """Per-slice replay sampling recommendation."""

    priority_score: float
    priority_label: str
    queue_tags: list[str]
    replay_action: str
    weight_multiplier: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority_score": float(self.priority_score),
            "priority_label": self.priority_label,
            "queue_tags": list(self.queue_tags),
            "replay_action": self.replay_action,
            "weight_multiplier": float(self.weight_multiplier),
            "reasons": list(self.reasons),
        }


def recommend_sampling(
    *,
    objective_profile_coverage_gap: float,
    constraint_violation_count: int,
    uncertainty: float,
    datapack_value: float,
    signal_yield_score: float = 0.0,
    regal_support_score: float,
    deploy_recommendation: str,
    pricing_recommendation: str,
    datapack_recommendation: str,
    promotion_policy: RegalPromotionPolicy,
    replay_policy_error: float,
    provenance_quality: float,
) -> ReplaySamplingRecommendation:
    score = 0.35
    reasons: list[str] = []
    tags = []
    if objective_profile_coverage_gap > 0.25:
        score += 0.15
        reasons.append("objective_profile_coverage_gap")
    if constraint_violation_count > 0:
        score += min(0.2, 0.05 * constraint_violation_count)
        reasons.append("constraint_violations_present")
        tags.append("reward_safety_review")
    if uncertainty > 0.4:
        score += 0.15
        reasons.append("uncertainty_high")
        tags.append("high_value_uncertain")
    if datapack_value > 1.0:
        score += 0.15
        reasons.append("frontier_candidate")
        tags.append("frontier_candidate")
    if signal_yield_score > 0.0:
        score += min(0.15, float(signal_yield_score))
        reasons.append("signal_yield_positive")
        tags.append("signal_yield_candidate")
    if regal_support_score > 0.6:
        score += 0.1
        reasons.append("regal_support_risk_high")
        tags.append("pricing_truth_review")
        tags.append("pricing_review")
    if pricing_recommendation != "publish":
        score += 0.1
        tags.append("pricing_truth_review")
        tags.append("pricing_review")
    if datapack_recommendation in {"downweight", "review"}:
        tags.append("downweight_candidate")
    if provenance_quality < 0.55:
        score += 0.1
        tags.append("low_provenance_review")
        tags.append("low_provenance")
    if replay_policy_error > 0.25:
        score += 0.1
        reasons.append("policy_error_high")
    if deploy_recommendation != "allow_shadow":
        score += 0.05
    if promotion_policy.stage_allows("reward_safety_regal", "budget_gate"):
        score += 0.03

    score = max(0.0, min(1.0, score))
    label = "low"
    if score >= 0.7:
        label = "high"
    elif score >= 0.45:
        label = "medium"

    action = "holdout"
    weight_multiplier = 1.0
    if label == "high" and datapack_recommendation != "downweight":
        action = "upweight"
        weight_multiplier = 1.2
    elif datapack_recommendation == "downweight" or provenance_quality < 0.45:
        action = "downweight"
        weight_multiplier = 0.7
    elif uncertainty > 0.45:
        action = "collect_more_like_this"
        weight_multiplier = 1.05
        tags.append("collect_more_like_this")
    else:
        tags.append("holdout_candidate")

    return ReplaySamplingRecommendation(
        priority_score=score,
        priority_label=label,
        queue_tags=sorted(set(tags)),
        replay_action=action,
        weight_multiplier=weight_multiplier,
        reasons=reasons or ["baseline_sampling"],
    )

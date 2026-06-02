"""Meta-regal aggregation for the shadow economic control plane."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.regality.shadow_nodes import (
    DataValueRegal,
    PricingTruthRegal,
    RewardSafetyRegal,
    ShadowRegalContext,
    ShadowRegalNode,
    ShadowRegalStatus,
    default_shadow_nodes,
)
from src.utils.config_digest import sha256_json


@dataclass(frozen=True)
class MetaRegalDecision:
    """Aggregate shadow control-plane decision."""

    overall_status: ShadowRegalStatus
    deploy_recommendation: str
    adaptation_recommendation: str
    pricing_recommendation: str
    datapack_recommendation: str
    reasons: List[str]
    node_decisions: List[Dict[str, object]]
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def decision_hash(self) -> str:
        return sha256_json(self._base_dict())

    def _base_dict(self) -> Dict[str, object]:
        return {
            "overall_status": self.overall_status.value,
            "deploy_recommendation": self.deploy_recommendation,
            "adaptation_recommendation": self.adaptation_recommendation,
            "pricing_recommendation": self.pricing_recommendation,
            "datapack_recommendation": self.datapack_recommendation,
            "reasons": list(self.reasons),
            "node_decisions": list(self.node_decisions),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> Dict[str, object]:
        payload = self._base_dict()
        payload["decision_hash"] = self.decision_hash
        return payload


class MetaRegalController:
    """Default shadow governance plane."""

    def __init__(self, nodes: Optional[Sequence[ShadowRegalNode]] = None) -> None:
        self.nodes = list(nodes or default_shadow_nodes())

    def evaluate(self, context: ShadowRegalContext) -> MetaRegalDecision:
        decisions = [node.evaluate(context) for node in self.nodes]
        overall_status = ShadowRegalStatus.PASS
        if any(decision.status == ShadowRegalStatus.FAIL for decision in decisions):
            overall_status = ShadowRegalStatus.FAIL
        elif any(decision.status == ShadowRegalStatus.WARN for decision in decisions):
            overall_status = ShadowRegalStatus.WARN

        decision_map = {decision.node_id: decision for decision in decisions}
        reasons = [reason for decision in decisions for reason in decision.reasons]

        deploy_recommendation = "allow_shadow"
        integrity_or_pricing_fail = any(
            decision.status == ShadowRegalStatus.FAIL
            and decision.node_id in {"objective_integrity_regal", "pricing_truth_regal"}
            for decision in decisions
        )
        if integrity_or_pricing_fail:
            deploy_recommendation = "deny_shadow"
        elif overall_status != ShadowRegalStatus.PASS:
            deploy_recommendation = "require_review"

        adaptation_recommendation = "no_op"
        data_decision = decision_map.get(DataValueRegal.node_id)
        reward_decision = decision_map.get(RewardSafetyRegal.node_id)
        marginal_gain = float(context.datapack_credit_update.get("marginal_frontier_gain", 0.0))
        if data_decision and reward_decision:
            if data_decision.status == ShadowRegalStatus.PASS and reward_decision.status == ShadowRegalStatus.PASS and marginal_gain > 0.05:
                adaptation_recommendation = "adapt"
            elif data_decision.status == ShadowRegalStatus.FAIL or reward_decision.status == ShadowRegalStatus.FAIL:
                adaptation_recommendation = "collect_data"

        pricing_recommendation = "publish"
        pricing_decision = decision_map.get(PricingTruthRegal.node_id)
        if pricing_decision and pricing_decision.status == ShadowRegalStatus.FAIL:
            pricing_recommendation = "suppress"
        elif overall_status != ShadowRegalStatus.PASS:
            pricing_recommendation = "publish_discounted"

        datapack_recommendation = "keep"
        if data_decision:
            if data_decision.status == ShadowRegalStatus.FAIL:
                datapack_recommendation = "downweight"
            elif data_decision.status == ShadowRegalStatus.WARN:
                datapack_recommendation = "review"
            elif float(context.datapack_credit_update.get("data_share_credit", 0.0)) > 0.0:
                datapack_recommendation = "reward_credit"

        return MetaRegalDecision(
            overall_status=overall_status,
            deploy_recommendation=deploy_recommendation,
            adaptation_recommendation=adaptation_recommendation,
            pricing_recommendation=pricing_recommendation,
            datapack_recommendation=datapack_recommendation,
            reasons=sorted(reasons),
            node_decisions=[decision.to_dict() for decision in decisions],
            metadata={
                "run_id": context.run_id,
                "episode_id": context.episode_id,
                "node_count": len(decisions),
            },
        )

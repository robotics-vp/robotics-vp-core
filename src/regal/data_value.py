"""Regal node that prices datapack value via frontier expansion and reliability."""
from __future__ import annotations

from typing import Mapping

from src.objectives.frontier import ParetoFrontierTracker
from src.objectives.tensor import ObjectiveTensor
from src.regal.base import RegalDecision, RegalNode, RegalReport


class RegalDataValueNode(RegalNode):
    """Promote datapacks that add marginal Pareto frontier gain per compute."""

    node_id = "regal_data_value"

    def __init__(self, frontier_tracker: ParetoFrontierTracker | None = None) -> None:
        self.frontier_tracker = frontier_tracker or ParetoFrontierTracker(
            maximize={
                "throughput": True,
                "error": False,
                "safety": True,
                "energy": False,
            }
        )

    def evaluate(self, context: Mapping[str, object]) -> RegalReport:
        objective_tensor_payload = context.get("objective_tensor")
        if isinstance(objective_tensor_payload, ObjectiveTensor):
            objective_tensor = objective_tensor_payload
        elif isinstance(objective_tensor_payload, dict):
            objective_tensor = ObjectiveTensor.from_dict(objective_tensor_payload)
        else:
            return RegalReport(
                node_id=self.node_id,
                decision=RegalDecision.BLOCK,
                reason_codes=["missing_objective_tensor"],
                recommended_action="attach_objective_tensor_v1",
                confidence=0.95,
            )

        task_id = str(context.get("task_id", "unknown_task"))
        env_id = str(context.get("env_id", "unknown_env"))
        profile_id = str(context.get("profile_id", "default"))
        compute_cost = float(context.get("compute_cost", 1.0) or 1.0)

        plausibility = float(context.get("plausibility_score", 1.0) or 1.0)
        reward_safety = float(context.get("reward_safety_score", 1.0) or 1.0)
        gen2sim_validity = float(context.get("gen2sim_validity_score", 1.0) or 1.0)
        reliability = max(0.0, min(1.0, plausibility * reward_safety * gen2sim_validity))

        marginal_gain = self.frontier_tracker.marginal_gain(
            objective_tensor,
            task_id=task_id,
            env_id=env_id,
            profile_id=profile_id,
            compute_cost=compute_cost,
        )
        effective_gain = marginal_gain * reliability

        if effective_gain <= 0.0:
            return RegalReport(
                node_id=self.node_id,
                decision=RegalDecision.REPAIR,
                reason_codes=["dominated_or_low_reliability"],
                details={
                    "marginal_gain": marginal_gain,
                    "reliability": reliability,
                    "effective_gain": effective_gain,
                },
                recommended_action="downgrade_datapack_or_collect_counterfactual",
                confidence=0.8,
            )

        self.frontier_tracker.add(
            objective_tensor,
            task_id=task_id,
            env_id=env_id,
            profile_id=profile_id,
            metadata={"source": context.get("source", "unknown")},
            compute_cost=compute_cost,
        )

        return RegalReport(
            node_id=self.node_id,
            decision=RegalDecision.ALLOW,
            reason_codes=["frontier_gain_positive"],
            details={
                "marginal_gain": marginal_gain,
                "reliability": reliability,
                "effective_gain": effective_gain,
            },
            recommended_action="promote_datapack",
            confidence=0.85,
        )

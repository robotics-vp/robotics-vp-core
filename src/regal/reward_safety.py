"""Reward safety regal gate."""
from __future__ import annotations

from typing import Mapping

from src.regal.base import RegalDecision, RegalNode, RegalReport


class RegalRewardSafetyNode(RegalNode):
    """Probe reward metrics for exploit signatures."""

    node_id = "regal_reward_safety"

    def __init__(self, exploit_reward_ratio_threshold: float = 3.0) -> None:
        self.exploit_reward_ratio_threshold = float(exploit_reward_ratio_threshold)

    def evaluate(self, context: Mapping[str, object]) -> RegalReport:
        reward_scalar = _as_float(context.get("reward_scalar"), 0.0)
        productive_signal = _as_float(context.get("productive_signal"), 0.0)
        constraint_violations = _as_list(context.get("constraint_violations"))
        anomaly_score = _as_float(context.get("anomaly_score"), 0.0)

        ratio = reward_scalar / max(1e-6, abs(productive_signal)) if productive_signal != 0 else float("inf")
        exploit_like = ratio >= self.exploit_reward_ratio_threshold and reward_scalar > 0

        if exploit_like and (constraint_violations or anomaly_score > 0.5):
            return RegalReport(
                node_id=self.node_id,
                decision=RegalDecision.REROUTE,
                reason_codes=["reward_exploit_suspected"],
                details={
                    "reward_scalar": reward_scalar,
                    "productive_signal": productive_signal,
                    "ratio": ratio,
                    "constraint_violations": constraint_violations,
                    "anomaly_score": anomaly_score,
                },
                recommended_action="refine_taskspec_and_recompile_reward",
                confidence=0.85,
            )

        return RegalReport(
            node_id=self.node_id,
            decision=RegalDecision.ALLOW,
            reason_codes=["reward_safety_ok"],
            details={"ratio": ratio, "anomaly_score": anomaly_score},
            confidence=0.8,
        )


def _as_float(value: object, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value)
        return default
    except (TypeError, ValueError):
        return default


def _as_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return []

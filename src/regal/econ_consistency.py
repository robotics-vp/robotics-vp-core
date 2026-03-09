"""Economic consistency regal gate."""
from __future__ import annotations

from typing import Mapping

from src.regal.base import RegalDecision, RegalNode, RegalReport


class RegalEconConsistencyNode(RegalNode):
    """Check econ updates remain consistent with objective outcomes and constraints."""

    node_id = "regal_econ_consistency"

    def evaluate(self, context: Mapping[str, object]) -> RegalReport:
        objective_score = _as_float(context.get("objective_score"), 0.0)
        econ_delta_value = _as_float(context.get("econ_delta_value"), 0.0)
        constraint_violations = _as_list(context.get("constraint_violations"))
        uncertainty = _as_float(context.get("uncertainty"), 0.0)

        if econ_delta_value > 0 and constraint_violations:
            return RegalReport(
                node_id=self.node_id,
                decision=RegalDecision.BLOCK,
                reason_codes=["positive_econ_claim_with_constraint_violation"],
                details={
                    "objective_score": objective_score,
                    "econ_delta_value": econ_delta_value,
                    "constraint_violations": constraint_violations,
                    "uncertainty": uncertainty,
                },
                recommended_action="void_positive_pricing_and_recompute",
                confidence=0.95,
            )

        if econ_delta_value > 0 and uncertainty > 0.6:
            return RegalReport(
                node_id=self.node_id,
                decision=RegalDecision.REROUTE,
                reason_codes=["positive_econ_claim_high_uncertainty"],
                details={
                    "objective_score": objective_score,
                    "econ_delta_value": econ_delta_value,
                    "uncertainty": uncertainty,
                },
                recommended_action="discount_value_by_uncertainty",
                confidence=0.8,
            )

        return RegalReport(
            node_id=self.node_id,
            decision=RegalDecision.ALLOW,
            reason_codes=["econ_consistency_ok"],
            details={
                "objective_score": objective_score,
                "econ_delta_value": econ_delta_value,
                "uncertainty": uncertainty,
            },
            confidence=0.85,
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

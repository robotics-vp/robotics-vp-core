"""Objective integrity regal gate."""
from __future__ import annotations

from typing import Any, Mapping

from src.regal.base import RegalDecision, RegalNode, RegalReport


class RegalObjectiveIntegrityNode(RegalNode):
    """Block runs that collapse ObjectiveTensor to scalar before compiler stage."""

    node_id = "regal_objective_integrity"

    def evaluate(self, context: Mapping[str, Any]) -> RegalReport:
        objective_tensor = context.get("objective_tensor")
        scalarized_upstream = bool(context.get("scalarized_upstream", False))
        compiler_stage_seen = bool(context.get("compiler_stage_seen", False))
        lineage = list(context.get("lineage", []) or [])

        if objective_tensor is None and scalarized_upstream and not compiler_stage_seen:
            return RegalReport(
                node_id=self.node_id,
                decision=RegalDecision.BLOCK,
                reason_codes=["objective_tensor_missing", "early_scalarization_detected"],
                details={"lineage": lineage},
                recommended_action="reconstruct_objective_tensor_then_compile",
                confidence=0.95,
            )

        if scalarized_upstream and compiler_stage_seen:
            return RegalReport(
                node_id=self.node_id,
                decision=RegalDecision.REROUTE,
                reason_codes=["scalarization_seen", "requires_trace_audit"],
                details={"lineage": lineage},
                recommended_action="verify_scalarization_trace",
                confidence=0.7,
            )

        return RegalReport(
            node_id=self.node_id,
            decision=RegalDecision.ALLOW,
            reason_codes=["objective_integrity_ok"],
            details={"lineage": lineage},
            confidence=0.9,
        )

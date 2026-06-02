"""Regal node that prices datapack value via frontier expansion and reliability."""
from __future__ import annotations

from typing import Any, Mapping

from src.evidence.gen2sim_validity import resolve_gen2sim_validity_assessment
from src.evidence.gen2sim_validity_runtime import resolve_gen2sim_validity_helper
from src.objectives.frontier import ParetoFrontierTracker
from src.objectives.tensor import ObjectiveTensor
from src.regal.base import RegalDecision, RegalNode, RegalReport


class RegalDataValueNode(RegalNode):
    """Promote datapacks that add marginal Pareto frontier gain per compute."""

    node_id = "regal_data_value"

    def __init__(
        self,
        frontier_tracker: ParetoFrontierTracker | None = None,
        *,
        gen2sim_validity_helper: Any = None,
        gen2sim_validity_mode: str = "auto",
    ) -> None:
        self.frontier_tracker = frontier_tracker or ParetoFrontierTracker(
            maximize={
                "throughput": True,
                "error": False,
                "safety": True,
                "energy": False,
            }
        )
        self.gen2sim_validity_helper = gen2sim_validity_helper
        self.gen2sim_validity_mode = str(gen2sim_validity_mode or "auto")
        self._resolved_gen2sim_helper: Any = None
        self._resolved_gen2sim_helper_status: dict[str, Any] = {
            "mode": self.gen2sim_validity_mode,
            "status": "unresolved",
            "promotion_stage": "heuristic_fallback",
            "benchmark_gate_ready": False,
        }
        self._gen2sim_helper_attempted = False

    def _resolve_gen2sim_helper(self, context: Mapping[str, object]) -> tuple[Any, dict[str, Any]]:
        override_helper = (
            context.get("gen2sim_validity_helper")
            or context.get("gen2sim_validity_package_path")
            or context.get("gen2sim_validity_package")
        )
        override_mode = str(context.get("gen2sim_validity_mode") or self.gen2sim_validity_mode)
        if override_helper is not None or override_mode != self.gen2sim_validity_mode:
            return resolve_gen2sim_validity_helper(
                override_helper if override_helper is not None else self.gen2sim_validity_helper,
                mode=override_mode,  # type: ignore[arg-type]
            )
        if not self._gen2sim_helper_attempted:
            self._resolved_gen2sim_helper, self._resolved_gen2sim_helper_status = (
                resolve_gen2sim_validity_helper(
                    self.gen2sim_validity_helper,
                    mode=self.gen2sim_validity_mode,  # type: ignore[arg-type]
                )
            )
            self._gen2sim_helper_attempted = True
        return self._resolved_gen2sim_helper, dict(self._resolved_gen2sim_helper_status)

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
        compute_cost = _as_float(context.get("compute_cost"), 1.0)

        plausibility = _as_float(context.get("plausibility_score"), 1.0)
        reward_safety = _as_float(context.get("reward_safety_score"), 1.0)
        helper_runtime, helper_status = self._resolve_gen2sim_helper(context)
        gen2sim_assessment = resolve_gen2sim_validity_assessment(
            context,
            subject_id=str(context.get("datapack_id") or task_id),
            subject_kind="datapack",
            helper=helper_runtime,
            helper_status=helper_status,
        )
        base_reliability = max(0.0, min(1.0, plausibility * reward_safety))
        reliability = (
            base_reliability
            if "gen2sim_not_applicable" in gen2sim_assessment.reason_codes
            else float(gen2sim_assessment.admission_score)
        )

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
                    "base_reliability": base_reliability,
                    "effective_gain": effective_gain,
                    "gen2sim_validity_assessment": gen2sim_assessment.to_dict(),
                    "gen2sim_helper_status": helper_status,
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
                "base_reliability": base_reliability,
                "effective_gain": effective_gain,
                "gen2sim_validity_assessment": gen2sim_assessment.to_dict(),
                "gen2sim_helper_status": helper_status,
            },
            recommended_action="promote_datapack",
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

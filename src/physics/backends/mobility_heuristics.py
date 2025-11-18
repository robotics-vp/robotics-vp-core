"""
Heuristic mobility micro-policy.
"""
from typing import Dict
import math

from src.physics.backends.mobility import MobilityPolicy, MobilityContext, MobilityAdjustment


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


class HeuristicMobilityPolicy(MobilityPolicy):
    def compute_adjustment(self, ctx: MobilityContext) -> MobilityAdjustment:
        stability = _clamp01(ctx.stability_margin)
        target_mm = max(float(ctx.target_precision_mm), 1.0)
        drift = abs(ctx.pose.get("drift_mm", 0.0)) if ctx.pose else 0.0
        contacts = ctx.contacts or {}
        slip_rate = float(contacts.get("slip_rate", 0.0))

        precision_gate_passed = drift <= target_mm
        recovery_required = stability < 0.3 or slip_rate > 0.3 or drift > target_mm * 2
        risk_level = "LOW"
        if stability < 0.3 or slip_rate > 0.4:
            risk_level = "HIGH"
        elif stability < 0.6 or slip_rate > 0.2:
            risk_level = "MEDIUM"

        delta_scale = max(0.0, (drift - target_mm) / max(target_mm, 1.0))
        delta_pose: Dict[str, float] = {}
        if drift > 0:
            delta_pose["drift_mm"] = -min(drift, target_mm * 0.5)
        if slip_rate > 0.1:
            delta_pose["grip_force_delta"] = -0.05 * (1.0 + delta_scale)

        stabilization_hint = "stable"
        if recovery_required:
            stabilization_hint = "micro_recovery"
        elif not precision_gate_passed:
            stabilization_hint = "tighten_precision"
        elif risk_level == "MEDIUM":
            stabilization_hint = "widen_stance"

        metadata = {
            "stability_margin": stability,
            "slip_rate": slip_rate,
            "drift_mm": drift,
        }
        return MobilityAdjustment(
            delta_pose=delta_pose,
            stabilization_hint=stabilization_hint,
            precision_gate_passed=precision_gate_passed,
            recovery_required=recovery_required,
            risk_level=risk_level,
            metadata=metadata,
        )

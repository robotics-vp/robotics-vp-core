"""Calibration receipt helpers for the sim/synth/physics world model."""

from __future__ import annotations

from typing import Any, Dict

from .common import clip01, mapping, safe_float, stable_id
from .physics_contracts import PhysicsExecutionContract
from .receipts import PhysicsCalibrationReceipt
from .state import SimSynthPhysicsWorldState


def build_physics_calibration_receipt(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
) -> PhysicsCalibrationReceipt:
    benchmark_signals = mapping(world_state.physics_context.metadata.get("benchmark_signals", {}))
    benchmark_gate_ready = bool(
        benchmark_signals.get("ready", False)
        or benchmark_signals.get("benchmark_eligible", False)
    )
    helper_status = mapping(world_state.physics_context.metadata.get("backend_helper_status", {}))
    base_quality = 0.2
    if execution_contract.route_status == "ready":
        base_quality = 0.62
    elif execution_contract.route_status == "fallback":
        base_quality = 0.38
    if benchmark_gate_ready:
        base_quality += 0.18
    if str(world_state.physics_context.fidelity_tier) == "high_fidelity":
        base_quality += 0.08
    if str(helper_status.get("promotion_stage", "")) == "promoted":
        base_quality += 0.05
    quality_score = clip01(base_quality)
    payload: Dict[str, Any] = {
        "state_id": world_state.state_id,
        "requested_backend": execution_contract.requested_backend,
        "resolved_backend": execution_contract.resolved_backend,
        "route_status": execution_contract.route_status,
        "quality_score": quality_score,
    }
    return PhysicsCalibrationReceipt(
        receipt_id=stable_id("physics_calibration_receipt", payload),
        backend=execution_contract.resolved_backend,
        fidelity_tier=world_state.physics_context.fidelity_tier,
        calibration_profile=world_state.physics_context.calibration_profile,
        quality_score=quality_score,
        metadata={
            "requested_backend": execution_contract.requested_backend,
            "resolved_backend": execution_contract.resolved_backend,
            "route_status": execution_contract.route_status,
            "fallback_reason": execution_contract.fallback_reason,
            "benchmark_gate_ready": benchmark_gate_ready,
            "backend_selection_policy": world_state.physics_context.selection_policy,
            "backend_helper_status": helper_status,
            "benchmark_signals": benchmark_signals,
            "explicit_gap_kind": str(
                mapping(execution_contract.metadata.get("requested_adapter", {})).get("metadata", {}).get(
                    "gap_kind",
                    "",
                )
            ),
            "timestep_ms": safe_float(world_state.physics_context.timestep_ms, 0.0),
        },
    )


__all__ = ["build_physics_calibration_receipt"]

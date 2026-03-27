"""Calibration receipt helpers for the sim/synth/physics world model."""

from __future__ import annotations

from typing import Any, Dict

from .common import clip01, mapping, safe_float, stable_id
from .physics_contracts import PhysicsExecutionContract
from .receipts import PhysicsAdaptationReceipt, PhysicsCalibrationReceipt
from .state import SimSynthPhysicsWorldState


def build_physics_adaptation_receipt(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    *,
    runtime_evidence: Dict[str, Any] | None = None,
) -> PhysicsAdaptationReceipt:
    adaptation_policy = world_state.physics_adaptation_policy
    benchmark_signals = mapping(world_state.physics_context.metadata.get("benchmark_signals", {}))
    benchmark_gate_ready = bool(
        benchmark_signals.get("ready", False)
        or benchmark_signals.get("benchmark_eligible", False)
    )
    runtime_evidence = mapping(runtime_evidence)
    target_hardware_class = (
        ""
        if adaptation_policy is None
        else str(adaptation_policy.target_hardware_class)
    )
    readiness = 0.26
    if execution_contract.route_status == "ready":
        readiness = 0.64
    elif execution_contract.route_status == "fallback":
        readiness = 0.38
    if benchmark_gate_ready:
        readiness += 0.14
    if str(execution_contract.fidelity_tier) == "high_fidelity":
        readiness += 0.08
    if str(runtime_evidence.get("runtime_execution_status", "")) == "runtime_execution_completed":
        readiness += 0.16
    if runtime_evidence.get("shadow_execution_status"):
        readiness += 0.08
    readiness += 0.04 * min(
        3.0,
        safe_float(runtime_evidence.get("materialized_render_provider_count", 0.0), 0.0),
    )
    readiness -= 0.03 * min(
        4.0,
        safe_float(runtime_evidence.get("shadow_missing_asset_count", 0.0), 0.0),
    )
    readiness -= 0.02 * min(
        4.0,
        safe_float(runtime_evidence.get("render_precondition_gap_count", 0.0), 0.0),
    )
    if target_hardware_class == "unitree_g1_r1_class" and execution_contract.resolved_backend not in {
        "isaac",
        "holosoma",
    }:
        readiness -= 0.12
    if adaptation_policy is None:
        payload = {
            "state_id": world_state.state_id,
            "backend": execution_contract.resolved_backend,
            "route_status": execution_contract.route_status,
        }
        return PhysicsAdaptationReceipt(
            receipt_id=stable_id("physics_adaptation_receipt", payload),
            policy_id="",
            backend=execution_contract.resolved_backend,
            target_hardware_class="unknown",
            domain_randomization_profile=world_state.physics_context.domain_randomization_regime,
            system_identification_profile="unknown",
            readiness_score=clip01(readiness),
            metadata={
                "route_status": execution_contract.route_status,
                "fallback_reason": execution_contract.fallback_reason,
                "benchmark_gate_ready": benchmark_gate_ready,
                "runtime_evidence": runtime_evidence,
            },
        )
    payload = {
        "state_id": world_state.state_id,
        "backend": execution_contract.resolved_backend,
        "policy_id": adaptation_policy.policy_id,
        "route_status": execution_contract.route_status,
    }
    return PhysicsAdaptationReceipt(
        receipt_id=stable_id("physics_adaptation_receipt", payload),
        policy_id=adaptation_policy.policy_id,
        backend=execution_contract.resolved_backend,
        target_hardware_class=adaptation_policy.target_hardware_class,
        domain_randomization_profile=adaptation_policy.domain_randomization_profile,
        system_identification_profile=adaptation_policy.system_identification_profile,
        readiness_score=clip01(readiness),
        metadata={
            "requested_backend": execution_contract.requested_backend,
            "resolved_backend": execution_contract.resolved_backend,
            "simulator_family": execution_contract.simulator_family,
            "route_status": execution_contract.route_status,
            "fallback_reason": execution_contract.fallback_reason,
            "benchmark_gate_ready": benchmark_gate_ready,
            "selection_policy": adaptation_policy.selection_policy,
            "randomization_axes": list(adaptation_policy.randomization_axes),
            "calibration_targets": list(adaptation_policy.calibration_targets),
            "robot_asset_profile": adaptation_policy.robot_asset_profile,
            "latency_budget_ms": safe_float(adaptation_policy.metadata.get("latency_budget_ms", 0.0), 0.0),
            "contact_risk_score": clip01(adaptation_policy.metadata.get("contact_risk_score", 0.0)),
            "runtime_evidence": runtime_evidence,
        },
    )


def build_physics_calibration_receipt(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    *,
    adaptation_receipt: PhysicsAdaptationReceipt | None = None,
    runtime_evidence: Dict[str, Any] | None = None,
) -> PhysicsCalibrationReceipt:
    benchmark_signals = mapping(world_state.physics_context.metadata.get("benchmark_signals", {}))
    benchmark_gate_ready = bool(
        benchmark_signals.get("ready", False)
        or benchmark_signals.get("benchmark_eligible", False)
    )
    helper_status = mapping(world_state.physics_context.metadata.get("backend_helper_status", {}))
    runtime_evidence = mapping(runtime_evidence)
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
    if str(runtime_evidence.get("runtime_execution_status", "")) == "runtime_execution_completed":
        base_quality += 0.12
    if adaptation_receipt is not None:
        base_quality += 0.12 * clip01(adaptation_receipt.readiness_score)
        if (
            str(adaptation_receipt.target_hardware_class) == "unitree_g1_r1_class"
            and execution_contract.resolved_backend not in {"isaac", "holosoma"}
        ):
            base_quality -= 0.08
    if runtime_evidence.get("shadow_execution_status"):
        base_quality += 0.06
    base_quality += 0.03 * min(
        3.0,
        safe_float(runtime_evidence.get("materialized_render_provider_count", 0.0), 0.0),
    )
    base_quality += 0.01 * min(
        6.0,
        safe_float(runtime_evidence.get("render_artifact_count", 0.0), 0.0),
    )
    base_quality -= 0.02 * min(
        4.0,
        safe_float(runtime_evidence.get("shadow_missing_asset_count", 0.0), 0.0),
    )
    base_quality -= 0.02 * min(
        4.0,
        safe_float(runtime_evidence.get("render_precondition_gap_count", 0.0), 0.0),
    )
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
        domain_randomization_regime=world_state.physics_context.domain_randomization_regime,
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
            "adaptation_policy_id": execution_contract.adaptation_policy_id,
            "simulator_family": execution_contract.simulator_family,
            "target_hardware_class": execution_contract.target_hardware_class,
            "adaptation_readiness_score": (
                0.0 if adaptation_receipt is None else clip01(adaptation_receipt.readiness_score)
            ),
            "runtime_evidence": runtime_evidence,
        },
    )


__all__ = ["build_physics_adaptation_receipt", "build_physics_calibration_receipt"]

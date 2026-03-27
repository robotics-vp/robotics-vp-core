"""Typed physics adaptation policy helpers for the sim/synth/physics WM."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .backend_adapters import BackendAdapterDescriptor
from .common import clip01, mapping, safe_float, stable_id, strings
from .state import PhysicsAdaptationPolicyState, PhysicsContextState


def _active_embodiments(embodiment_context: Optional[Mapping[str, Any]]) -> list[str]:
    payload = mapping(embodiment_context)
    return strings(
        payload.get("active_embodiments")
        or payload.get("target_embodiments")
        or payload.get("robot_families")
    )


def _target_hardware_class(
    *,
    adapter: BackendAdapterDescriptor,
    embodiment_context: Optional[Mapping[str, Any]],
) -> str:
    embodiment_values = " ".join(_active_embodiments(embodiment_context)).lower()
    if any(token in embodiment_values for token in ("unitree", "g1", "r1", "humanoid")):
        return "unitree_g1_r1_class"
    if str(adapter.target_hardware_class):
        return str(adapter.target_hardware_class)
    return "tabletop_fixed_base"


def _domain_randomization_profile(
    *,
    physics_context: PhysicsContextState,
    target_hardware_class: str,
    contact_risk_score: float,
) -> str:
    if target_hardware_class == "unitree_g1_r1_class":
        if physics_context.domain_randomization_regime == "benchmark_focus":
            return "humanoid_contact_latency_and_sensor_calibration"
        if physics_context.domain_randomization_regime == "calibration_focus":
            return "humanoid_system_id_shadow"
        return "humanoid_shadow_randomization"
    if contact_risk_score >= 0.6:
        return "contact_guarded_tabletop_randomization"
    if physics_context.domain_randomization_regime == "calibration_focus":
        return "tabletop_system_id"
    if physics_context.domain_randomization_regime == "coverage_exploration":
        return "coverage_randomization"
    return "steady_state"


def _system_identification_profile(
    *,
    physics_context: PhysicsContextState,
    target_hardware_class: str,
    latency_budget_ms: float,
    benchmark_ready: bool,
) -> str:
    if target_hardware_class == "unitree_g1_r1_class":
        if benchmark_ready or physics_context.fidelity_tier == "high_fidelity":
            return "whole_body_latency_contact_and_actuator_id"
        return "humanoid_shadow_system_id"
    if latency_budget_ms > 0.0:
        return "latency_and_contact_system_id"
    return "tabletop_contact_system_id"


def _randomization_axes(
    *,
    physics_context: PhysicsContextState,
    target_hardware_class: str,
    contact_risk_score: float,
) -> list[str]:
    axes = ["lighting", "camera_pose", "material_friction", "surface_noise"]
    if physics_context.fidelity_tier in {"branch_balanced", "high_fidelity"}:
        axes.extend(["latency", "action_repeat", "sensor_noise"])
    if contact_risk_score >= 0.4:
        axes.extend(["contact_offset", "object_mass", "joint_damping"])
    if target_hardware_class == "unitree_g1_r1_class":
        axes.extend(
            [
                "imu_bias",
                "joint_backlash",
                "ground_friction",
                "center_of_mass_shift",
                "foot_contact_threshold",
                "battery_voltage_sag",
            ]
        )
    return sorted(set(axes))


def _calibration_targets(
    *,
    target_hardware_class: str,
    benchmark_ready: bool,
) -> list[str]:
    targets = ["camera_extrinsics", "reward_overlay_alignment", "timing_signature"]
    if benchmark_ready:
        targets.append("benchmark_ready_scene_alignment")
    if target_hardware_class == "unitree_g1_r1_class":
        targets.extend(
            [
                "whole_body_joint_map",
                "imu_alignment",
                "foot_contact_model",
                "actuator_delay_profile",
            ]
        )
    return sorted(set(targets))


def compile_physics_adaptation_policy(
    physics_context: PhysicsContextState,
    *,
    adapter: BackendAdapterDescriptor,
    benchmark_signals: Mapping[str, Any],
    embodiment_context: Optional[Mapping[str, Any]] = None,
) -> PhysicsAdaptationPolicyState:
    benchmark_ready = bool(
        benchmark_signals.get("ready", False) or benchmark_signals.get("benchmark_eligible", False)
    )
    embodiment_payload = mapping(embodiment_context)
    control_constraints = mapping(embodiment_payload.get("control_constraints"))
    latency_budget_ms = safe_float(
        control_constraints.get("latency_budget_ms", embodiment_payload.get("latency_budget_ms", 0.0)),
        0.0,
    )
    contact_risk_score = clip01(
        embodiment_payload.get("contact_risk_score", control_constraints.get("contact_risk_score", 0.0))
    )
    target_hardware_class = _target_hardware_class(
        adapter=adapter,
        embodiment_context=embodiment_payload,
    )
    domain_randomization_profile = _domain_randomization_profile(
        physics_context=physics_context,
        target_hardware_class=target_hardware_class,
        contact_risk_score=contact_risk_score,
    )
    system_identification_profile = _system_identification_profile(
        physics_context=physics_context,
        target_hardware_class=target_hardware_class,
        latency_budget_ms=latency_budget_ms,
        benchmark_ready=benchmark_ready,
    )
    randomization_axes = _randomization_axes(
        physics_context=physics_context,
        target_hardware_class=target_hardware_class,
        contact_risk_score=contact_risk_score,
    )
    calibration_targets = _calibration_targets(
        target_hardware_class=target_hardware_class,
        benchmark_ready=benchmark_ready,
    )
    payload = {
        "backend": physics_context.backend,
        "fidelity_tier": physics_context.fidelity_tier,
        "domain_randomization_profile": domain_randomization_profile,
        "system_identification_profile": system_identification_profile,
        "target_hardware_class": target_hardware_class,
    }
    return PhysicsAdaptationPolicyState(
        policy_id=stable_id("physics_adaptation_policy", payload),
        backend=physics_context.backend,
        simulator_family=adapter.simulator_family,
        target_hardware_class=target_hardware_class,
        robot_asset_profile=(
            "unitree_humanoid_shadow_assets"
            if target_hardware_class == "unitree_g1_r1_class"
            else "tabletop_workcell_assets"
        ),
        domain_randomization_profile=domain_randomization_profile,
        system_identification_profile=system_identification_profile,
        randomization_axes=randomization_axes,
        calibration_targets=calibration_targets,
        selection_policy="wm_physics_adaptation_policy_v1",
        metadata={
            "benchmark_ready": benchmark_ready,
            "control_constraints": control_constraints,
            "active_embodiments": _active_embodiments(embodiment_payload),
            "latency_budget_ms": latency_budget_ms,
            "contact_risk_score": contact_risk_score,
            "domain_randomization_regime": physics_context.domain_randomization_regime,
            "calibration_profile": physics_context.calibration_profile,
            "adapter_status": adapter.adapter_status,
            "adapter_supports_execution": bool(adapter.supports_execution),
            "target_runtime_stack": list(adapter.metadata.get("target_runtime_stack", []) or []),
            "requires_unitree_assets": bool(target_hardware_class == "unitree_g1_r1_class"),
        },
    )


__all__ = ["compile_physics_adaptation_policy"]

"""Phase 1.x task, scene, and provider-lane surface compilers."""

from __future__ import annotations

from statistics import mean
from typing import Any, Mapping, Optional, Sequence

from .common import mapping, safe_float, stable_id, strings
from .state import (
    BackendExecutionBindingState,
    DifferentiablePhysicsProviderState,
    PhysicsContextState,
    RobotAssetContractState,
    SceneHierarchyState,
    SimulatorBackendContractState,
    SurrogatePhysicsProviderState,
    SyntheticBranchPlan,
    TaskDefinitionContractState,
    TaskMeasurementSurface,
)


def _benchmark_ready(benchmark_signals: Mapping[str, Any]) -> bool:
    payload = mapping(benchmark_signals)
    return bool(payload.get("ready", False) or payload.get("benchmark_eligible", False))


def _average(values: Sequence[float]) -> float:
    return 0.0 if not values else float(mean(values))


def compile_task_measurement_surface(
    jobs: Sequence[Any],
    *,
    physics_context: PhysicsContextState,
    benchmark_signals: Mapping[str, Any],
) -> TaskMeasurementSurface:
    """Compile a Habitat-style task measurement surface from the current agenda."""

    task_families = strings([getattr(job, "task_family", "") for job in jobs])
    task_family = task_families[0] if task_families else "unknown"
    measurement_values = {
        "coverage_gap_score": _average(
            [safe_float(getattr(job, "coverage_gap_score", 0.0), 0.0) for job in jobs]
        ),
        "economic_priority": _average(
            [safe_float(getattr(job, "economic_priority", 0.0), 0.0) for job in jobs]
        ),
        "trust_priority": _average(
            [safe_float(getattr(job, "trust_priority", 0.0), 0.0) for job in jobs]
        ),
        "promotion_readiness": _average(
            [safe_float(getattr(job, "readiness", 0.0), 0.0) for job in jobs]
        ),
    }
    benchmark_gate_ready = _benchmark_ready(benchmark_signals)
    measurement_status = {
        "coverage_gap_score": "available" if jobs else "deferred",
        "economic_priority": "available" if jobs else "deferred",
        "trust_priority": "available" if jobs else "deferred",
        "promotion_readiness": (
            "benchmark_ready" if benchmark_gate_ready else "shadow_only"
        ),
    }
    benchmark_payload = mapping(benchmark_signals)
    payload = {
        "task_family": task_family,
        "job_ids": [str(getattr(job, "job_id", "") or "") for job in jobs],
        "backend": physics_context.backend,
        "benchmark_gate_ready": benchmark_gate_ready,
    }
    return TaskMeasurementSurface(
        surface_id=stable_id("task_measurement_surface", payload),
        task_family=task_family,
        measurement_names=list(measurement_values),
        measurement_values=measurement_values,
        measurement_status=measurement_status,
        measurement_dependencies={
            "promotion_readiness": ["coverage_gap_score", "trust_priority"],
        },
        episode_refs=[str(getattr(job, "job_id", "") or "") for job in jobs],
        vector_env_count=int(benchmark_payload.get("vector_env_count", 0) or 0),
        measurement_window_steps=int(
            benchmark_payload.get(
                "measurement_window_steps",
                benchmark_payload.get("measurement_window_frames", len(jobs)),
            )
            or 0
        ),
        benchmark_gate_ready=benchmark_gate_ready,
        metadata={
            "physics_context_id": physics_context.context_id,
            "physics_backend": physics_context.backend,
            "ranking_policies": sorted(
                {str(getattr(job, "ranking_policy", "") or "") for job in jobs if getattr(job, "ranking_policy", "")}
            ),
            "objective_presets": sorted(
                {str(getattr(job, "objective_preset", "") or "") for job in jobs if getattr(job, "objective_preset", "")}
            ),
            "benchmark_signals": benchmark_payload,
        },
    )


def _object_ids_from_perception_state(perception_grounding_state: Any) -> list[str]:
    scene_graph = getattr(perception_grounding_state, "scene_graph", None)
    tracks = getattr(scene_graph, "object_tracks", []) or []
    return strings([getattr(track, "track_id", "") for track in tracks])


def compile_scene_hierarchy_state(
    jobs: Sequence[Any],
    *,
    robot_asset_contract: RobotAssetContractState,
    semantic_context: Optional[Mapping[str, Any]],
    perception_grounding_state: Any = None,
) -> SceneHierarchyState:
    """Compile a scene hierarchy surface without importing provider ownership."""

    semantic_payload = mapping(semantic_context)
    scene_id = str(
        semantic_payload.get("scene_id")
        or semantic_payload.get("world_id")
        or (getattr(jobs[0], "task_family", "") if jobs else "unbound_scene")
        or "unbound_scene"
    )
    scene_kind = str(semantic_payload.get("scene_kind") or "task_scene")
    hierarchy_levels = strings(
        semantic_payload.get("scene_hierarchy_levels")
        or semantic_payload.get("hierarchy_levels")
        or ["scene", "region", "object"]
    )
    region_ids = strings(
        semantic_payload.get("region_ids")
        or semantic_payload.get("scene_region_ids")
        or []
    )
    object_ids = strings(
        semantic_payload.get("object_ids")
        or semantic_payload.get("scene_object_ids")
        or _object_ids_from_perception_state(perception_grounding_state)
    )
    node_counts_by_level = {
        "scene": 1 if scene_id else 0,
        "region": len(region_ids),
        "object": len(object_ids),
    }
    node_counts_by_level.update(
        {
            str(key): int(value)
            for key, value in mapping(semantic_payload.get("node_counts_by_level")).items()
        }
    )
    missing_assets = list(robot_asset_contract.missing_assets)
    materialization_status = "asset_contract_ready"
    if missing_assets:
        materialization_status = "asset_contract_incomplete"
    elif not jobs:
        materialization_status = "no_scene_requested"
    payload = {
        "scene_id": scene_id,
        "scene_kind": scene_kind,
        "hierarchy_levels": hierarchy_levels,
        "asset_profile": robot_asset_contract.asset_profile,
    }
    return SceneHierarchyState(
        hierarchy_id=stable_id("scene_hierarchy", payload),
        scene_id=scene_id,
        scene_kind=scene_kind,
        hierarchy_levels=hierarchy_levels,
        node_counts_by_level=node_counts_by_level,
        region_ids=region_ids,
        object_ids=object_ids,
        asset_refs=list(robot_asset_contract.available_assets),
        sensor_profile=str(semantic_payload.get("sensor_profile", "") or ""),
        materialization_status=materialization_status,
        metadata={
            "robot_asset_contract_id": robot_asset_contract.contract_id,
            "asset_profile": robot_asset_contract.asset_profile,
            "missing_assets": missing_assets,
            "source_job_ids": [str(getattr(job, "job_id", "") or "") for job in jobs],
            "semantic_context": semantic_payload,
        },
    )


def compile_simulator_backend_contract_state(
    jobs: Sequence[Any],
    *,
    physics_context: PhysicsContextState,
    backend_execution_binding: BackendExecutionBindingState,
) -> SimulatorBackendContractState:
    """Compile the simulator half of the simulator / task protocol split."""

    task_families = sorted(
        {
            str(getattr(job, "task_family", "") or "")
            for job in jobs
            if str(getattr(job, "task_family", "") or "")
        }
    )
    payload = {
        "backend": physics_context.backend,
        "binding_id": backend_execution_binding.binding_id,
        "task_families": task_families,
    }
    return SimulatorBackendContractState(
        contract_id=stable_id("simulator_backend_contract", payload),
        backend=physics_context.backend,
        simulator_family=(
            str(backend_execution_binding.metadata.get("adapter_descriptor", {}).get("simulator_family", ""))
            or str(physics_context.metadata.get("backend_adapter", {}).get("simulator_family", ""))
        ),
        fidelity_tier=physics_context.fidelity_tier,
        executor_entrypoint=backend_execution_binding.executor_entrypoint,
        observation_adapter_entrypoint=backend_execution_binding.observation_adapter_entrypoint,
        runtime_status=backend_execution_binding.binding_status,
        supported_task_families=task_families,
        metadata={
            "physics_context_id": physics_context.context_id,
            "backend_execution_binding_id": backend_execution_binding.binding_id,
            "selection_policy": physics_context.selection_policy,
        },
    )


def compile_task_definition_contract_state(
    jobs: Sequence[Any],
    *,
    task_measurements: TaskMeasurementSurface,
) -> TaskDefinitionContractState:
    """Compile the task half of the simulator / task protocol split."""

    top_job = jobs[0] if jobs else None
    task_family = "unknown" if top_job is None else str(getattr(top_job, "task_family", "unknown"))
    objective_preset = "balanced" if top_job is None else str(
        getattr(top_job, "objective_preset", "balanced")
    )
    payload = {
        "task_family": task_family,
        "objective_preset": objective_preset,
        "episode_refs": list(task_measurements.episode_refs),
    }
    return TaskDefinitionContractState(
        contract_id=stable_id("task_definition_contract", payload),
        task_family=task_family,
        objective_preset=objective_preset,
        episode_refs=list(task_measurements.episode_refs),
        required_measurements=list(task_measurements.measurement_names),
        reset_protocol="episode_reset_then_measurement_reset",
        update_protocol="step_update_then_measurement_update",
        metadata={
            "task_measurement_surface_id": task_measurements.surface_id,
            "job_count": len(jobs),
        },
    )


def compile_differentiable_physics_provider_state(
    physics_context: PhysicsContextState,
    branch_plans: Sequence[SyntheticBranchPlan],
    *,
    benchmark_signals: Mapping[str, Any],
) -> DifferentiablePhysicsProviderState:
    """Compile the reserved differentiable-provider lane as typed state."""

    benchmark_payload = mapping(benchmark_signals)
    provider_available = bool(benchmark_payload.get("differentiable_provider_available", False))
    provider_family = str(
        benchmark_payload.get("differentiable_provider_family")
        or ("jaxsim_like" if provider_available else "reserved_differentiable_lane")
    )
    provider_status = "shadow_ready" if provider_available else "contract_reserved"
    gradient_mode = str(
        benchmark_payload.get("gradient_mode")
        or ("reverse_mode_available" if provider_available else "not_bound")
    )
    payload = {
        "backend": physics_context.backend,
        "provider_family": provider_family,
        "provider_status": provider_status,
    }
    return DifferentiablePhysicsProviderState(
        provider_id=stable_id("differentiable_physics_provider", payload),
        provider_family=provider_family,
        provider_status=provider_status,
        backend=physics_context.backend,
        gradient_mode=gradient_mode,
        supported_features=(
            ["trajectory_gradient", "system_id_gradient", "inverse_design_gradient"]
            if provider_available
            else ["trajectory_gradient", "system_id_gradient"]
        ),
        compatible_branch_ids=[plan.plan_id for plan in branch_plans],
        available=provider_available,
        metadata={
            "physics_context_id": physics_context.context_id,
            "benchmark_signals": benchmark_payload,
            "lane_authority": "advisory_provider_only",
        },
    )


def compile_surrogate_physics_provider_state(
    branch_plans: Sequence[SyntheticBranchPlan],
    *,
    benchmark_signals: Mapping[str, Any],
) -> SurrogatePhysicsProviderState:
    """Compile the reserved surrogate-provider lane as typed state."""

    benchmark_payload = mapping(benchmark_signals)
    provider_available = bool(benchmark_payload.get("surrogate_provider_available", False))
    provider_family = str(
        benchmark_payload.get("surrogate_provider_family")
        or ("windinet_like" if provider_available else "reserved_surrogate_lane")
    )
    provider_status = "shadow_ready" if provider_available else "contract_reserved"
    calibration_status = str(
        benchmark_payload.get("surrogate_calibration_status")
        or ("shadow_calibrated" if provider_available else "not_calibrated")
    )
    payload = {
        "provider_family": provider_family,
        "provider_status": provider_status,
        "branch_plan_ids": [plan.plan_id for plan in branch_plans],
    }
    return SurrogatePhysicsProviderState(
        provider_id=stable_id("surrogate_physics_provider", payload),
        provider_family=provider_family,
        provider_status=provider_status,
        forecast_mode=str(
            benchmark_payload.get("surrogate_forecast_mode")
            or ("branch_preview" if provider_available else "unbound")
        ),
        calibration_status=calibration_status,
        supported_targets=(
            ["branch_scoring", "inverse_design_preview", "calibration_hint"]
            if provider_available
            else ["branch_scoring", "inverse_design_preview"]
        ),
        compatible_branch_ids=[plan.plan_id for plan in branch_plans],
        available=provider_available,
        metadata={
            "benchmark_signals": benchmark_payload,
            "lane_authority": "advisory_provider_only",
        },
    )


__all__ = [
    "compile_differentiable_physics_provider_state",
    "compile_scene_hierarchy_state",
    "compile_simulator_backend_contract_state",
    "compile_surrogate_physics_provider_state",
    "compile_task_definition_contract_state",
    "compile_task_measurement_surface",
]

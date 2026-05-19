"""Phase 1.x subsystem index for the sim/synth/physics world model.

This module makes the roadmap's 10-subsystem decomposition machine-readable.
It is intentionally structural: it maps existing state, receipt, learned-seam,
and blocker surfaces without claiming provider/GPU readiness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .common import mapping, stable_id, strings


@dataclass(frozen=True)
class Phase1xSubsystemSpec:
    """Static ownership record for one Phase 1.x subsystem."""

    subsystem_id: str
    ordinal: int
    name: str
    responsibility: str
    owned_modules: tuple[str, ...] = field(default_factory=tuple)
    typed_state_surfaces: tuple[str, ...] = field(default_factory=tuple)
    receipt_surfaces: tuple[str, ...] = field(default_factory=tuple)
    learned_seams: tuple[str, ...] = field(default_factory=tuple)
    promotion_gates: tuple[str, ...] = field(default_factory=tuple)
    provider_families: tuple[str, ...] = field(default_factory=tuple)
    external_blockers: tuple[str, ...] = field(default_factory=tuple)
    artifact_ref_keys: tuple[str, ...] = field(default_factory=tuple)
    local_status: str = "structural_mapping_ready"

    def to_dict(
        self,
        *,
        artifact_refs: Mapping[str, Any] | None = None,
        available_receipt_families: Sequence[str] = (),
    ) -> dict[str, Any]:
        refs = mapping(artifact_refs)
        receipt_family_set = {str(item) for item in available_receipt_families}
        artifact_refs_present = {
            key: refs[key]
            for key in self.artifact_ref_keys
            if key in refs and refs[key] not in (None, "", [])
        }
        receipt_surfaces_present = [
            receipt
            for receipt in self.receipt_surfaces
            if receipt in receipt_family_set
        ]
        return {
            "subsystem_id": self.subsystem_id,
            "ordinal": int(self.ordinal),
            "name": self.name,
            "responsibility": self.responsibility,
            "owned_modules": strings(self.owned_modules),
            "typed_state_surfaces": strings(self.typed_state_surfaces),
            "receipt_surfaces": strings(self.receipt_surfaces),
            "receipt_surfaces_present": receipt_surfaces_present,
            "learned_seams": strings(self.learned_seams),
            "promotion_gates": strings(self.promotion_gates),
            "provider_families": strings(self.provider_families),
            "provider_truth_owner": "sim_synth_physics_wm",
            "provider_ownership_rule": "providers_may_span_subsystems_but_never_own_wm_truth",
            "external_blockers": strings(self.external_blockers),
            "artifact_ref_keys": strings(self.artifact_ref_keys),
            "artifact_refs_present": artifact_refs_present,
            "local_status": self.local_status,
        }


PHASE1X_SUBSYSTEM_SPECS: tuple[Phase1xSubsystemSpec, ...] = (
    Phase1xSubsystemSpec(
        subsystem_id="phase1x_subsystem_01_backend_runtime_provider_surface",
        ordinal=1,
        name="Backend / Runtime / Provider Surface",
        responsibility="backend selection, runtime binding, provider truth, adapter lifecycle, and fidelity routing",
        owned_modules=(
            "backend_adapters.py",
            "backend_bindings.py",
            "backend_router.py",
            "backend_selector.py",
            "backend_selector_runtime.py",
            "runtime_bridge.py",
            "runtime_bundles.py",
            "runtime_launch.py",
            "adapters/",
        ),
        typed_state_surfaces=(
            "PhysicsContextState",
            "BackendExecutionBindingState",
            "BackendRuntimeBridgeState",
            "SimulatorBackendContractState",
        ),
        receipt_surfaces=(
            "physics_execution_contract_v1",
            "backend_execution_binding_receipt_v1",
            "backend_runtime_bridge_receipt_v1",
            "backend_runtime_work_order_receipt_v1",
            "backend_runtime_execution_receipt_v1",
            "backend_runtime_adapter_receipt_v1",
            "backend_runtime_launch_receipt_v1",
            "backend_runtime_outcome_receipt_v1",
        ),
        learned_seams=("LearnedBackendSelector",),
        promotion_gates=(
            "sim_synth_backend_selector_dataset_density",
            "phase1x_training_gate_v1",
        ),
        provider_families=("pybullet", "isaac_unitree", "holosoma"),
        external_blockers=(
            "isaac_lab_or_isaac_sim_runtime",
            "unitree_assets_and_latency_watchdog_profiles",
            "holosoma_runtime_execution",
            "gpu_host_for_concrete_provider_runs",
        ),
        artifact_ref_keys=(
            "physics_context_id",
            "physics_execution_contract_id",
            "backend_execution_binding_id",
            "backend_runtime_bridge_id",
            "simulator_backend_contract_id",
        ),
    ),
    Phase1xSubsystemSpec(
        subsystem_id="phase1x_subsystem_02_task_measurement_episode_layer",
        ordinal=2,
        name="Task / Measurement / Episode Layer",
        responsibility="task definitions, measurement surfaces, episode lifecycle, and benchmark/evaluation harness posture",
        owned_modules=("agenda.py", "phase1x_surfaces.py", "runtime.py", "vectorized_runtime.py"),
        typed_state_surfaces=(
            "SimulationAgenda",
            "SimulationJobSpec",
            "TaskMeasurementSurface",
            "TaskDefinitionContractState",
        ),
        receipt_surfaces=(
            "task_measurement_receipt_v1",
            "simulation_outcome_receipt_v1",
        ),
        promotion_gates=("task_measurement_benchmark_gate",),
        external_blockers=("gpu_backed_vectorized_simulation", "provider_episode_rollouts"),
        artifact_ref_keys=(
            "coverage_window_ref",
            "task_measurement_surface_id",
            "task_definition_contract_id",
        ),
    ),
    Phase1xSubsystemSpec(
        subsystem_id="phase1x_subsystem_03_scene_asset_materialization_layer",
        ordinal=3,
        name="Scene / Asset / Materialization Layer",
        responsibility="scene hierarchy, robot asset contracts, sensor geometry, and materialization readiness",
        owned_modules=(
            "asset_contracts.py",
            "asset_manifest.py",
            "phase1x_surfaces.py",
            "render_materialization.py",
            "utils/camera_geometry.py",
        ),
        typed_state_surfaces=("RobotAssetContractState", "SceneHierarchyState"),
        receipt_surfaces=(
            "robot_asset_contract_receipt_v1",
            "sensor_alignment_receipt_v1",
            "render_provider_receipt_v1",
        ),
        provider_families=("isaac_unitree", "unreal", "ggds_ldm"),
        external_blockers=(
            "unitree_asset_tree",
            "whole_body_latency_profile",
            "safety_watchdog_profile",
            "gpu_backed_scene_materialization",
        ),
        artifact_ref_keys=("robot_asset_contract_id", "scene_hierarchy_id"),
    ),
    Phase1xSubsystemSpec(
        subsystem_id="phase1x_subsystem_04_branch_planner_evaluator",
        ordinal=4,
        name="Branch Planner / Branch Evaluator",
        responsibility="agenda-to-branch planning, branch validity, branch feedback, and learned branch-helper routing",
        owned_modules=("agenda.py", "branch_planner.py", "branch_planner_runtime.py", "synthetic_branches.py"),
        typed_state_surfaces=("SimulationAgenda", "SyntheticBranchPlan"),
        receipt_surfaces=(
            "branch_validity_receipt_v1",
            "replay_validity_receipt_v1",
            "simulation_outcome_receipt_v1",
        ),
        learned_seams=("LearnedBranchPlanner",),
        promotion_gates=(
            "sim_synth_branch_planner_dataset_density",
            "phase1x_training_gate_v1",
        ),
        external_blockers=("held_out_branch_outcomes", "provider_rollout_benchmark_density"),
        artifact_ref_keys=("branch_plan_ids",),
    ),
    Phase1xSubsystemSpec(
        subsystem_id="phase1x_subsystem_05_sim_real_gap_realism_evaluator",
        ordinal=5,
        name="Sim-Real Gap / Realism Evaluator",
        responsibility="sim-real transfer quality, realism confidence, replay consistency, and transfer-risk evidence",
        owned_modules=("inferential.py", "runtime.py", "training_corpus.py"),
        typed_state_surfaces=("SyntheticBranchPlan",),
        receipt_surfaces=(
            "sim_real_gap_receipt_v1",
            "replay_validity_receipt_v1",
            "surrogate_physics_receipt_v1",
        ),
        learned_seams=("reserved_sim_real_gap_encoder",),
        promotion_gates=("sim_real_gap_benchmark_gate",),
        external_blockers=("paired_sim_real_execution_data", "real_robot_or_high_fidelity_sim_outcomes"),
        artifact_ref_keys=("branch_plan_ids",),
    ),
    Phase1xSubsystemSpec(
        subsystem_id="phase1x_subsystem_06_fidelity_randomization_calibration_allocator",
        ordinal=6,
        name="Fidelity / Randomization / Calibration Allocator",
        responsibility="domain-randomization policy, system identification, calibration targets, and fidelity/throughput tradeoffs",
        owned_modules=("randomization.py", "calibration.py", "physics_contracts.py"),
        typed_state_surfaces=("PhysicsAdaptationPolicyState", "PhysicsExecutionContract"),
        receipt_surfaces=(
            "physics_adaptation_receipt_v1",
            "physics_calibration_receipt_v1",
        ),
        learned_seams=("reserved_fidelity_allocator",),
        promotion_gates=("calibration_quality_gate",),
        external_blockers=("provider_calibration_evidence", "unitree_latency_and_watchdog_profiles"),
        artifact_ref_keys=("physics_adaptation_policy_id", "physics_execution_contract_id"),
    ),
    Phase1xSubsystemSpec(
        subsystem_id="phase1x_subsystem_07_render_diffusion_materialization_lane",
        ordinal=7,
        name="Render / Diffusion / Materialization Lane",
        responsibility="render-provider contracts, diffusion conditioning, visual materialization, and materialization quality",
        owned_modules=("diffusion_contracts.py", "render_providers.py", "render_materialization.py"),
        typed_state_surfaces=("DiffusionConditioningState", "BranchRenderProviderState"),
        receipt_surfaces=("render_provider_receipt_v1",),
        learned_seams=("reserved_render_quality_critic",),
        provider_families=("ggds_ldm", "unreal", "isaac_rendering"),
        external_blockers=("gpu_backed_diffusion_or_renderer_weights", "provider_render_outputs"),
        artifact_ref_keys=("diffusion_conditioning_id",),
    ),
    Phase1xSubsystemSpec(
        subsystem_id="phase1x_subsystem_08_differentiable_physics_provider_lane",
        ordinal=8,
        name="Differentiable-Physics Provider Lane",
        responsibility="typed reserve for differentiable physics providers and gradient-compatible branch markers",
        owned_modules=("phase1x_surfaces.py", "runtime_evidence.py"),
        typed_state_surfaces=(
            "DifferentiablePhysicsProviderState",
            "SurrogatePhysicsProviderState",
        ),
        receipt_surfaces=(
            "surrogate_physics_receipt_v1",
            "surrogate_calibration_receipt_v1",
        ),
        learned_seams=("reserved_differentiable_physics_calibrator",),
        provider_families=("jaxsim", "brax", "differentiable_mujoco", "newton"),
        external_blockers=("differentiable_backend_install", "gradient_verified_provider_runs"),
        artifact_ref_keys=(
            "differentiable_physics_provider_id",
            "surrogate_physics_provider_id",
        ),
        local_status="contract_reserved_until_provider_evidence",
    ),
    Phase1xSubsystemSpec(
        subsystem_id="phase1x_subsystem_09_drift_calibration_backend_mismatch_evaluator",
        ordinal=9,
        name="Drift / Calibration / Backend Mismatch Evaluator",
        responsibility="backend mismatch, calibration staleness, drift trend, and backend quality evidence",
        owned_modules=("calibration.py", "runtime_evidence.py", "training_corpus.py"),
        typed_state_surfaces=("PhysicsAdaptationPolicyState",),
        receipt_surfaces=(
            "backend_mismatch_receipt_v1",
            "physics_calibration_receipt_v1",
            "surrogate_calibration_receipt_v1",
        ),
        learned_seams=("reserved_drift_evaluator",),
        promotion_gates=("backend_quality_trend_gate",),
        external_blockers=("multi_backend_runtime_history", "calibration_trajectory_data"),
        artifact_ref_keys=("physics_adaptation_policy_id",),
    ),
    Phase1xSubsystemSpec(
        subsystem_id="phase1x_subsystem_10_training_worthiness_synthetic_yield_evaluator",
        ordinal=10,
        name="Training-Worthiness / Synthetic-Yield Evaluator",
        responsibility="branch admission, training-worthiness, yield estimation, and trainer-facing row admissibility",
        owned_modules=("gen2sim_admission.py", "inferential.py", "training_corpus.py"),
        typed_state_surfaces=("Gen2SimAdmissionState",),
        receipt_surfaces=(
            "gen2sim_admission_receipt_v1",
            "task_measurement_receipt_v1",
            "simulation_outcome_receipt_v1",
        ),
        learned_seams=(
            "reserved_synthetic_yield_predictor",
            "LearnedBackendSelector.reject_probability",
            "LearnedBranchPlanner.reject_probability",
        ),
        promotion_gates=(
            "phase1x_training_gate_v1",
            "training_admissibility_gate",
        ),
        external_blockers=("held_out_training_yield_evidence", "provider_benchmark_density"),
        artifact_ref_keys=("admission_id", "compiled_receipt_inventory_id"),
    ),
)


def _receipt_families_from_inventory(receipt_inventory: Mapping[str, Any] | None) -> list[str]:
    inventory = mapping(receipt_inventory)
    families: list[str] = []
    for key in ("compiler_owned_receipts", "runtime_owned_receipts", "per_branch_receipts"):
        families.extend(strings(inventory.get(key)))
    return sorted(set(families))


def build_phase1x_subsystem_index(
    *,
    artifact_refs: Mapping[str, Any] | None = None,
    receipt_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the machine-readable Phase 1.x subsystem decomposition index."""

    refs = mapping(artifact_refs)
    receipt_families = _receipt_families_from_inventory(receipt_inventory)
    subsystems = [
        spec.to_dict(
            artifact_refs=refs,
            available_receipt_families=receipt_families,
        )
        for spec in PHASE1X_SUBSYSTEM_SPECS
    ]
    payload = {
        "subsystem_ids": [item["subsystem_id"] for item in subsystems],
        "artifact_ref_keys": sorted(refs),
        "receipt_families": receipt_families,
    }
    return {
        "schema_version": "phase1x_subsystem_index_v1",
        "index_id": stable_id("phase1x_subsystem_index", payload),
        "world_model": "sim_synth_physics",
        "subsystem_count": len(subsystems),
        "subsystems": subsystems,
        "provider_ownership_rule": "providers_may_span_subsystems_but_never_own_wm_truth",
        "structural_status": "mapped_static_with_runtime_refs" if refs else "mapped_static",
        "honest_remaining_blocker_class": (
            "external_gpu_runtime_asset_benchmark_or_provider_evidence"
        ),
        "coverage_summary": {
            "subsystems_with_typed_state_surfaces": sum(
                1 for item in subsystems if item["typed_state_surfaces"]
            ),
            "subsystems_with_receipt_surfaces": sum(
                1 for item in subsystems if item["receipt_surfaces"]
            ),
            "subsystems_with_present_artifact_refs": sum(
                1 for item in subsystems if item["artifact_refs_present"]
            ),
            "subsystems_with_compiled_receipt_families": sum(
                1 for item in subsystems if item["receipt_surfaces_present"]
            ),
            "subsystems_with_learned_or_reserved_seams": sum(
                1 for item in subsystems if item["learned_seams"]
            ),
        },
    }


__all__ = [
    "PHASE1X_SUBSYSTEM_SPECS",
    "Phase1xSubsystemSpec",
    "build_phase1x_subsystem_index",
]

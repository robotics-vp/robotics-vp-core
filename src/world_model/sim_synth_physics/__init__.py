"""Canonical sim/synth/physics world-model contracts and compiler."""

from __future__ import annotations

from typing import Any

from .agenda import SimulationAgenda, SimulationJobSpec
from .asset_contracts import compile_robot_asset_contract
from .backend_adapters import BackendAdapterDescriptor, describe_backend_adapter
from .backend_bindings import compile_backend_execution_binding
from .backend_selector import LearnedBackendSelector, train_backend_selector
from .backend_selector_runtime import (
    BackendSelectorRuntimePackage,
    load_backend_selector_runtime_package,
    resolve_backend_selector_helper,
)
from .backend_router import build_physics_execution_contract
from .branch_planner import LearnedBranchPlanner, train_branch_planner
from .branch_planner_runtime import (
    BranchPlannerRuntimePackage,
    load_branch_planner_runtime_package,
    resolve_branch_planner_helper,
)
from .calibration import (
    build_physics_adaptation_receipt,
    build_physics_calibration_receipt,
)
from .diffusion_contracts import GapDrivenDiffusionPlan, compile_gap_driven_diffusion_plans
from .gen2sim_admission import (
    assess_local_branch_corpus_gen2sim,
    compile_gen2sim_admission_state,
)
from .physics_contracts import PhysicsExecutionContract
from .render_materialization import materialize_render_provider_receipts
from .receipts import (
    BackendExecutionBindingReceipt,
    BackendRuntimeBridgeReceipt,
    BackendRuntimeOutcomeReceipt,
    BackendShadowExecutionReceipt,
    BackendRuntimeWorkOrderReceipt,
    PhysicsAdaptationReceipt,
    PhysicsCalibrationReceipt,
    RenderProviderReceipt,
    RobotAssetContractReceipt,
    SimulationOutcomeReceipt,
)
from .runtime_bridge import (
    build_backend_runtime_bridge_receipt,
    compile_backend_runtime_bridge,
)
from .runtime_bundles import build_backend_runtime_bundle
from .runtime_launch import (
    execute_backend_runtime_launch,
    load_runtime_artifacts,
    prepare_backend_runtime_launch,
)
from .runtime_outcomes import (
    build_backend_runtime_outcome_receipt,
    build_backend_runtime_output_contract,
    harvest_backend_runtime_outcomes,
)
from .runtime_layouts import (
    describe_holosoma_policy_contract,
    describe_holosoma_runtime_layouts,
    describe_isaac_policy_contract,
    describe_isaac_runtime_layouts,
)
from .runtime_work_orders import build_backend_runtime_work_orders
from .runtime_evidence import summarize_runtime_evidence
from .shadow_execution import materialize_backend_shadow_execution
from .state import (
    BackendExecutionBindingState,
    BackendRuntimeBridgeState,
    BranchRenderProviderState,
    DiffusionConditioningState,
    Gen2SimAdmissionState,
    PhysicsAdaptationPolicyState,
    PhysicsContextState,
    RobotAssetContractState,
    SimSynthPhysicsWorldState,
    SyntheticBranchPlan,
)
from .synthetic_branches import (
    build_synthetic_branch_corpus_metadata,
    collect_local_synthetic_branch_records,
    compile_synthetic_branch_plans,
    compute_branch_gap_labels,
    extract_branch_features,
)
from .training_corpus import (
    build_backend_selector_rows_from_receipts,
    build_branch_planner_rows_from_receipts,
    harvest_sim_synth_receipt_bundles,
    load_sim_synth_receipt_bundles,
)

__all__ = [
    "BackendSelectorRuntimePackage",
    "BackendAdapterDescriptor",
    "BackendExecutionBindingReceipt",
    "BackendRuntimeBridgeReceipt",
    "BackendRuntimeOutcomeReceipt",
    "BackendRuntimeWorkOrderReceipt",
    "BackendShadowExecutionReceipt",
    "BackendExecutionBindingState",
    "BackendRuntimeBridgeState",
    "BranchRenderProviderState",
    "BranchPlannerRuntimePackage",
    "DiffusionConditioningState",
    "GapDrivenDiffusionPlan",
    "Gen2SimAdmissionState",
    "LearnedBackendSelector",
    "LearnedBranchPlanner",
    "PhysicsAdaptationPolicyState",
    "PhysicsAdaptationReceipt",
    "PhysicsCalibrationReceipt",
    "PhysicsExecutionContract",
    "PhysicsContextState",
    "RenderProviderReceipt",
    "RobotAssetContractReceipt",
    "RobotAssetContractState",
    "SimSynthPhysicsLoopResult",
    "SimSynthPhysicsRuntime",
    "SimSynthPhysicsRuntimeConfig",
    "SimSynthPhysicsWorldState",
    "SimulationAgenda",
    "SimulationJobSpec",
    "SimulationOutcomeReceipt",
    "SyntheticBranchPlan",
    "assess_local_branch_corpus_gen2sim",
    "build_physics_adaptation_receipt",
    "build_physics_calibration_receipt",
    "compile_robot_asset_contract",
    "compile_backend_runtime_bridge",
    "describe_holosoma_policy_contract",
    "describe_holosoma_runtime_layouts",
    "describe_isaac_policy_contract",
    "describe_isaac_runtime_layouts",
    "materialize_backend_shadow_execution",
    "materialize_render_provider_receipts",
    "summarize_runtime_evidence",
    "build_backend_runtime_bridge_receipt",
    "build_backend_runtime_bundle",
    "build_backend_runtime_outcome_receipt",
    "build_backend_runtime_output_contract",
    "build_backend_runtime_work_orders",
    "execute_backend_runtime_launch",
    "compile_backend_execution_binding",
    "build_physics_execution_contract",
    "build_synthetic_branch_corpus_metadata",
    "collect_local_synthetic_branch_records",
    "compile_sim_synth_physics_world_state",
    "compile_gen2sim_admission_state",
    "describe_backend_adapter",
    "compile_synthetic_branch_plans",
    "compute_branch_gap_labels",
    "extract_branch_features",
    "load_backend_selector_runtime_package",
    "load_branch_planner_runtime_package",
    "load_runtime_artifacts",
    "load_sim_synth_receipt_bundles",
    "prepare_backend_runtime_launch",
    "resolve_backend_selector_helper",
    "resolve_branch_planner_helper",
    "train_backend_selector",
    "train_branch_planner",
    "build_backend_selector_rows_from_receipts",
    "build_branch_planner_rows_from_receipts",
    "compile_gap_driven_diffusion_plans",
    "harvest_sim_synth_receipt_bundles",
    "harvest_backend_runtime_outcomes",
]


def __getattr__(name: str) -> Any:
    if name == "compile_sim_synth_physics_world_state":
        from .compiler import compile_sim_synth_physics_world_state

        return compile_sim_synth_physics_world_state
    if name in {
        "SimSynthPhysicsLoopResult",
        "SimSynthPhysicsRuntime",
        "SimSynthPhysicsRuntimeConfig",
    }:
        from .runtime import (
            SimSynthPhysicsLoopResult,
            SimSynthPhysicsRuntime,
            SimSynthPhysicsRuntimeConfig,
        )

        if name == "SimSynthPhysicsLoopResult":
            return SimSynthPhysicsLoopResult
        if name == "SimSynthPhysicsRuntime":
            return SimSynthPhysicsRuntime
        return SimSynthPhysicsRuntimeConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

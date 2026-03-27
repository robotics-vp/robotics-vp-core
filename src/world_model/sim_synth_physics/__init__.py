"""Canonical sim/synth/physics world-model contracts and compiler."""

from __future__ import annotations

from typing import Any

from .agenda import SimulationAgenda, SimulationJobSpec
from .backend_selector import LearnedBackendSelector, train_backend_selector
from .backend_selector_runtime import (
    BackendSelectorRuntimePackage,
    load_backend_selector_runtime_package,
    resolve_backend_selector_helper,
)
from .backend_router import (
    BackendAdapterDescriptor,
    build_physics_execution_contract,
    describe_backend_adapter,
)
from .branch_planner import LearnedBranchPlanner, train_branch_planner
from .branch_planner_runtime import (
    BranchPlannerRuntimePackage,
    load_branch_planner_runtime_package,
    resolve_branch_planner_helper,
)
from .calibration import build_physics_calibration_receipt
from .diffusion_contracts import GapDrivenDiffusionPlan, compile_gap_driven_diffusion_plans
from .gen2sim_admission import (
    assess_local_branch_corpus_gen2sim,
    compile_gen2sim_admission_state,
)
from .physics_contracts import PhysicsExecutionContract
from .receipts import PhysicsCalibrationReceipt, SimulationOutcomeReceipt
from .state import (
    DiffusionConditioningState,
    Gen2SimAdmissionState,
    PhysicsContextState,
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
    "BranchPlannerRuntimePackage",
    "DiffusionConditioningState",
    "GapDrivenDiffusionPlan",
    "Gen2SimAdmissionState",
    "LearnedBackendSelector",
    "LearnedBranchPlanner",
    "PhysicsCalibrationReceipt",
    "PhysicsExecutionContract",
    "PhysicsContextState",
    "SimSynthPhysicsLoopResult",
    "SimSynthPhysicsRuntime",
    "SimSynthPhysicsRuntimeConfig",
    "SimSynthPhysicsWorldState",
    "SimulationAgenda",
    "SimulationJobSpec",
    "SimulationOutcomeReceipt",
    "SyntheticBranchPlan",
    "assess_local_branch_corpus_gen2sim",
    "build_physics_calibration_receipt",
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
    "load_sim_synth_receipt_bundles",
    "resolve_backend_selector_helper",
    "resolve_branch_planner_helper",
    "train_backend_selector",
    "train_branch_planner",
    "build_backend_selector_rows_from_receipts",
    "build_branch_planner_rows_from_receipts",
    "compile_gap_driven_diffusion_plans",
    "harvest_sim_synth_receipt_bundles",
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

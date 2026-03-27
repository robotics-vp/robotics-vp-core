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
from .branch_planner import LearnedBranchPlanner, train_branch_planner
from .branch_planner_runtime import (
    BranchPlannerRuntimePackage,
    load_branch_planner_runtime_package,
    resolve_branch_planner_helper,
)
from .diffusion_contracts import GapDrivenDiffusionPlan, compile_gap_driven_diffusion_plans
from .receipts import PhysicsCalibrationReceipt, SimulationOutcomeReceipt
from .state import (
    DiffusionConditioningState,
    Gen2SimAdmissionState,
    PhysicsContextState,
    SimSynthPhysicsWorldState,
    SyntheticBranchPlan,
)
from .training_corpus import (
    build_backend_selector_rows_from_receipts,
    build_branch_planner_rows_from_receipts,
    harvest_sim_synth_receipt_bundles,
    load_sim_synth_receipt_bundles,
)

__all__ = [
    "BackendSelectorRuntimePackage",
    "BranchPlannerRuntimePackage",
    "DiffusionConditioningState",
    "GapDrivenDiffusionPlan",
    "Gen2SimAdmissionState",
    "LearnedBackendSelector",
    "LearnedBranchPlanner",
    "PhysicsCalibrationReceipt",
    "PhysicsContextState",
    "SimSynthPhysicsRuntime",
    "SimSynthPhysicsRuntimeConfig",
    "SimSynthPhysicsWorldState",
    "SimulationAgenda",
    "SimulationJobSpec",
    "SimulationOutcomeReceipt",
    "SyntheticBranchPlan",
    "compile_sim_synth_physics_world_state",
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
    if name in {"SimSynthPhysicsRuntime", "SimSynthPhysicsRuntimeConfig"}:
        from .runtime import SimSynthPhysicsRuntime, SimSynthPhysicsRuntimeConfig

        if name == "SimSynthPhysicsRuntime":
            return SimSynthPhysicsRuntime
        return SimSynthPhysicsRuntimeConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

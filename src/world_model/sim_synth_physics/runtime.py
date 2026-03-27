"""Runtime facade for the sim/synth/physics world model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from .backend_router import build_physics_execution_contract
from .calibration import (
    build_physics_adaptation_receipt,
    build_physics_calibration_receipt,
)
from .common import mapping
from .compiler import compile_sim_synth_physics_world_state
from .diffusion_contracts import GapDrivenDiffusionPlan, compile_gap_driven_diffusion_plans
from .physics_contracts import PhysicsExecutionContract
from .promotion import HelperMode
from .receipts import (
    BackendExecutionBindingReceipt,
    PhysicsAdaptationReceipt,
    PhysicsCalibrationReceipt,
    RenderProviderReceipt,
    SimulationOutcomeReceipt,
)
from .state import SimSynthPhysicsWorldState


@dataclass(frozen=True)
class SimSynthPhysicsRuntimeConfig:
    """Configuration for compiling and running the sim/synth/physics WM."""

    economic_weight: float = 1.0
    trust_weight: float = 1.0
    readiness_weight: float = 1.0
    agenda_limit: int = 10
    default_backend: str = "pybullet"
    default_objective: str = "balanced"
    gap_ranker_mode: Literal["disabled", "auto", "required"] = "auto"
    backend_selector_mode: HelperMode = "auto"
    branch_planner_mode: HelperMode = "auto"
    fallback_backend: str = "pybullet"


@dataclass(frozen=True)
class SimSynthPhysicsLoopResult:
    """Canonical result for one WM-owned planning/execution window."""

    world_state: SimSynthPhysicsWorldState
    physics_execution_contract: PhysicsExecutionContract
    physics_adaptation_receipt: PhysicsAdaptationReceipt
    backend_execution_binding_receipt: BackendExecutionBindingReceipt
    physics_calibration_receipt: PhysicsCalibrationReceipt
    render_provider_receipts: list[RenderProviderReceipt] = field(default_factory=list)
    outcome_receipts: list[SimulationOutcomeReceipt] = field(default_factory=list)
    training_feedback_manifest: Mapping[str, Any] = field(default_factory=dict)
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    version: str = "sim_synth_physics_loop_result_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_state": self.world_state.to_dict(),
            "physics_execution_contract": self.physics_execution_contract.to_dict(),
            "physics_adaptation_receipt": self.physics_adaptation_receipt.to_dict(),
            "backend_execution_binding_receipt": self.backend_execution_binding_receipt.to_dict(),
            "physics_calibration_receipt": self.physics_calibration_receipt.to_dict(),
            "render_provider_receipts": [
                receipt.to_dict() for receipt in self.render_provider_receipts
            ],
            "outcome_receipts": [receipt.to_dict() for receipt in self.outcome_receipts],
            "training_feedback_manifest": mapping(self.training_feedback_manifest),
            "artifact_paths": mapping(self.artifact_paths),
            "version": self.version,
        }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "world_state": root / "sim_synth_physics_world_state.json",
        "physics_execution_contract": root / "physics_execution_contract.json",
        "physics_adaptation_receipt": root / "physics_adaptation_receipt.json",
        "backend_execution_binding_receipt": root / "backend_execution_binding_receipt.json",
        "physics_calibration_receipt": root / "physics_calibration_receipt.json",
        "render_provider_receipts": root / "render_provider_receipts.json",
        "simulation_outcome_receipts": root / "simulation_outcome_receipts.json",
        "training_feedback_manifest": root / "sim_synth_training_feedback.json",
        "loop_summary": root / "sim_synth_physics_loop_summary.json",
        "diffusion_plans": root / "gap_driven_diffusion_plans.json",
    }


def _training_feedback_ref(
    *,
    manifest_path: Optional[Path],
    state_id: str,
    plan_id: str,
) -> str:
    if manifest_path is not None:
        return f"{manifest_path.resolve()}#branch_plan_id={plan_id}"
    return f"sim_synth_training_feedback:{state_id}#branch_plan_id={plan_id}"


def _outcome_status(
    *,
    plan_id: str,
    admissible_branch_ids: set[str],
    route_status: str,
) -> str:
    if plan_id not in admissible_branch_ids:
        return "blocked_by_admission"
    if route_status == "blocked":
        return "blocked_backend_unavailable"
    if route_status == "fallback":
        return "planned_with_backend_fallback"
    return "planned_for_execution"


def _build_outcome_receipts(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    adaptation_receipt: PhysicsAdaptationReceipt,
    backend_binding_receipt: BackendExecutionBindingReceipt,
    calibration_receipt: PhysicsCalibrationReceipt,
    *,
    training_feedback_path: Optional[Path] = None,
) -> list[SimulationOutcomeReceipt]:
    admissible_branch_ids = set(
        getattr(world_state.gen2sim_admission, "admissible_branch_ids", []) or []
    )
    job_by_id = {job.job_id: job for job in world_state.simulation_agenda.jobs}
    receipts: list[SimulationOutcomeReceipt] = []
    for index, plan in enumerate(world_state.synthetic_branch_plans):
        job = job_by_id.get(plan.source_job_id)
        receipt_payload = {
            "state_id": world_state.state_id,
            "job_id": plan.source_job_id,
            "branch_plan_id": plan.plan_id,
            "route_status": execution_contract.route_status,
            "status": _outcome_status(
                plan_id=plan.plan_id,
                admissible_branch_ids=admissible_branch_ids,
                route_status=execution_contract.route_status,
            ),
        }
        feedback_ref = _training_feedback_ref(
            manifest_path=training_feedback_path,
            state_id=world_state.state_id,
            plan_id=plan.plan_id,
        )
        render_provider = plan.render_provider
        receipts.append(
            SimulationOutcomeReceipt(
                receipt_id=f"simulation_outcome_receipt_{world_state.state_id}_{index + 1:03d}",
                job_id=plan.source_job_id,
                branch_plan_id=plan.plan_id,
                status=str(receipt_payload["status"]),
                governance_refs=[
                    world_state.state_id,
                    str(world_state.simulation_agenda.agenda_id),
                    str(world_state.gen2sim_admission.admission_id)
                    if world_state.gen2sim_admission is not None
                    else "",
                ],
                training_feedback_refs=[feedback_ref],
                metadata={
                    "world_state_id": world_state.state_id,
                    "physics_execution_contract_id": execution_contract.contract_id,
                    "physics_adaptation_receipt_id": adaptation_receipt.receipt_id,
                    "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
                    "physics_calibration_receipt_id": calibration_receipt.receipt_id,
                    "requested_backend": execution_contract.requested_backend,
                    "resolved_backend": execution_contract.resolved_backend,
                    "route_status": execution_contract.route_status,
                    "fallback_reason": execution_contract.fallback_reason,
                    "branch_family": plan.branch_family,
                    "generation_mode": plan.generation_mode,
                    "expected_yield_score": float(plan.expected_yield_score),
                    "branch_selection_policy": plan.selection_policy,
                    "render_provider_id": (
                        "" if render_provider is None else str(render_provider.provider_id)
                    ),
                    "render_provider_kind": (
                        "" if render_provider is None else str(render_provider.provider_kind)
                    ),
                    "render_provider_status": (
                        "" if render_provider is None else str(render_provider.provider_status)
                    ),
                    "job_rank": int(job.rank) if job is not None else index + 1,
                    "artifact_materialization": "planned_only",
                    "inferential_learnability_contract": mapping(
                        plan.inferential_learnability_contract
                    ),
                },
            )
        )
    return receipts


def _build_training_feedback_manifest(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    adaptation_receipt: PhysicsAdaptationReceipt,
    backend_binding_receipt: BackendExecutionBindingReceipt,
    calibration_receipt: PhysicsCalibrationReceipt,
    render_provider_receipts: list[RenderProviderReceipt],
    outcome_receipts: list[SimulationOutcomeReceipt],
) -> dict[str, Any]:
    render_receipts_by_plan = {
        str(receipt.branch_plan_id): receipt for receipt in render_provider_receipts
    }
    rows: list[dict[str, Any]] = []
    for receipt in outcome_receipts:
        render_receipt = render_receipts_by_plan.get(str(receipt.branch_plan_id))
        rows.append(
            {
                "branch_plan_id": str(receipt.branch_plan_id),
                "job_id": str(receipt.job_id),
                "status": str(receipt.status),
                "training_feedback_refs": list(receipt.training_feedback_refs),
                "render_provider_receipt_ref": (
                    None if render_receipt is None else render_receipt.receipt_id
                ),
                "metadata": mapping(receipt.metadata),
            }
        )
    return {
        "version": "sim_synth_training_feedback_v1",
        "world_state_id": world_state.state_id,
        "physics_execution_contract_id": execution_contract.contract_id,
        "physics_adaptation_receipt_id": adaptation_receipt.receipt_id,
        "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
        "physics_calibration_receipt_id": calibration_receipt.receipt_id,
        "route_status": execution_contract.route_status,
        "requested_backend": execution_contract.requested_backend,
        "resolved_backend": execution_contract.resolved_backend,
        "render_provider_receipt_count": len(render_provider_receipts),
        "benchmark_gate_ready": bool(
            getattr(world_state.gen2sim_admission, "benchmark_gate_ready", False)
        ),
        "blocked_branch_count": sum(
            1 for receipt in outcome_receipts if str(receipt.status).startswith("blocked_")
        ),
        "planned_branch_count": sum(
            1 for receipt in outcome_receipts if str(receipt.status).startswith("planned_")
        ),
        "rows": rows,
    }


def _build_backend_execution_binding_receipt(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    adaptation_receipt: PhysicsAdaptationReceipt,
) -> BackendExecutionBindingReceipt:
    binding = world_state.backend_execution_binding
    if binding is None:
        return BackendExecutionBindingReceipt(
            receipt_id=f"backend_execution_binding_receipt_{world_state.state_id}",
            binding_id="",
            backend=execution_contract.resolved_backend,
            binding_status="missing",
            executor_entrypoint="",
            asset_profile="",
            metadata={
                "world_state_id": world_state.state_id,
                "physics_execution_contract_id": execution_contract.contract_id,
                "physics_adaptation_receipt_id": adaptation_receipt.receipt_id,
            },
        )
    return BackendExecutionBindingReceipt(
        receipt_id=f"backend_execution_binding_receipt_{world_state.state_id}",
        binding_id=binding.binding_id,
        backend=binding.backend,
        binding_status=binding.binding_status,
        executor_entrypoint=binding.executor_entrypoint,
        asset_profile=binding.asset_profile,
        metadata={
            "world_state_id": world_state.state_id,
            "physics_execution_contract_id": execution_contract.contract_id,
            "physics_adaptation_receipt_id": adaptation_receipt.receipt_id,
            "executor_kind": binding.executor_kind,
            "observation_adapter_entrypoint": binding.observation_adapter_entrypoint,
            "target_runtime_stack": list(binding.target_runtime_stack),
            "required_assets": list(binding.required_assets),
            "available_assets": list(binding.available_assets),
            "missing_assets": list(binding.missing_assets),
            "binding_metadata": mapping(binding.metadata),
        },
    )


def _build_render_provider_receipts(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    adaptation_receipt: PhysicsAdaptationReceipt,
) -> list[RenderProviderReceipt]:
    receipts: list[RenderProviderReceipt] = []
    for index, plan in enumerate(world_state.synthetic_branch_plans, start=1):
        provider = plan.render_provider
        if provider is None:
            continue
        receipts.append(
            RenderProviderReceipt(
                receipt_id=f"render_provider_receipt_{world_state.state_id}_{index:03d}",
                branch_plan_id=plan.plan_id,
                provider_id=provider.provider_id,
                provider_kind=provider.provider_kind,
                provider_status=provider.provider_status,
                render_mode=provider.render_mode,
                counterfactual_mode=provider.counterfactual_mode,
                materialization_status=provider.materialization_status,
                materialization_entrypoint=provider.materialization_entrypoint,
                metadata={
                    "world_state_id": world_state.state_id,
                    "physics_execution_contract_id": execution_contract.contract_id,
                    "physics_adaptation_receipt_id": adaptation_receipt.receipt_id,
                    "generation_mode": plan.generation_mode,
                    "fallback_provider": provider.fallback_provider,
                    "fallback_reason": provider.fallback_reason,
                    "ggds_mode": provider.ggds_mode,
                    "provider_config": mapping(provider.provider_config),
                    "target_hardware_class": adaptation_receipt.target_hardware_class,
                    "provider_metadata": mapping(provider.metadata),
                },
            )
        )
    return receipts


class SimSynthPhysicsRuntime:
    """Compiler/runtime boundary for the sim/synth/physics WM."""

    def __init__(self, config: Optional[SimSynthPhysicsRuntimeConfig] = None) -> None:
        self.config = config or SimSynthPhysicsRuntimeConfig()

    def compile_world_state(
        self,
        coverage_graph: Any,
        *,
        semantic_context: Optional[Mapping[str, Any]] = None,
        economic_context: Optional[Mapping[str, Any]] = None,
        embodiment_context: Optional[Mapping[str, Any]] = None,
        benchmark_signals: Optional[Mapping[str, Any]] = None,
        gap_ranker: Any = None,
        backend_selector: Any = None,
        branch_planner: Any = None,
    ) -> SimSynthPhysicsWorldState:
        return compile_sim_synth_physics_world_state(
            coverage_graph,
            semantic_context=semantic_context,
            economic_context=economic_context,
            embodiment_context=embodiment_context,
            benchmark_signals=benchmark_signals,
            economic_weight=self.config.economic_weight,
            trust_weight=self.config.trust_weight,
            readiness_weight=self.config.readiness_weight,
            limit=self.config.agenda_limit,
            default_backend=self.config.default_backend,
            default_objective=self.config.default_objective,
            gap_ranker=gap_ranker,
            gap_ranker_mode=self.config.gap_ranker_mode,
            backend_selector=backend_selector,
            backend_selector_mode=self.config.backend_selector_mode,
            branch_planner=branch_planner,
            branch_planner_mode=self.config.branch_planner_mode,
        )

    def compile_legacy_agenda(
        self,
        coverage_graph: Any,
        *,
        semantic_context: Optional[Mapping[str, Any]] = None,
        economic_context: Optional[Mapping[str, Any]] = None,
        embodiment_context: Optional[Mapping[str, Any]] = None,
        benchmark_signals: Optional[Mapping[str, Any]] = None,
        gap_ranker: Any = None,
        backend_selector: Any = None,
        branch_planner: Any = None,
    ) -> list[dict[str, Any]]:
        return self.compile_world_state(
            coverage_graph,
            semantic_context=semantic_context,
            economic_context=economic_context,
            embodiment_context=embodiment_context,
            benchmark_signals=benchmark_signals,
            gap_ranker=gap_ranker,
            backend_selector=backend_selector,
            branch_planner=branch_planner,
        ).simulation_agenda.to_legacy_items()

    def compile_diffusion_plans(
        self,
        world_state: SimSynthPhysicsWorldState,
        *,
        coverage_graph: Any = None,
        limit: Optional[int] = None,
    ) -> list[GapDrivenDiffusionPlan]:
        return compile_gap_driven_diffusion_plans(
            world_state,
            coverage_graph=coverage_graph,
            limit=limit,
        )

    def compile_world_state_and_diffusion_plans(
        self,
        coverage_graph: Any,
        *,
        semantic_context: Optional[Mapping[str, Any]] = None,
        economic_context: Optional[Mapping[str, Any]] = None,
        embodiment_context: Optional[Mapping[str, Any]] = None,
        benchmark_signals: Optional[Mapping[str, Any]] = None,
        gap_ranker: Any = None,
        backend_selector: Any = None,
        branch_planner: Any = None,
        limit: Optional[int] = None,
    ) -> tuple[SimSynthPhysicsWorldState, list[GapDrivenDiffusionPlan]]:
        world_state = self.compile_world_state(
            coverage_graph,
            semantic_context=semantic_context,
            economic_context=economic_context,
            embodiment_context=embodiment_context,
            benchmark_signals=benchmark_signals,
            gap_ranker=gap_ranker,
            backend_selector=backend_selector,
            branch_planner=branch_planner,
        )
        diffusion_plans = self.compile_diffusion_plans(
            world_state,
            coverage_graph=coverage_graph,
            limit=limit,
        )
        return world_state, diffusion_plans

    def execute_world_state(
        self,
        world_state: SimSynthPhysicsWorldState,
        *,
        output_dir: str | Path | None = None,
    ) -> SimSynthPhysicsLoopResult:
        artifact_paths = _artifact_paths(output_dir) if output_dir is not None else {}
        execution_contract = build_physics_execution_contract(
            world_state,
            fallback_backend=self.config.fallback_backend,
        )
        adaptation_receipt = build_physics_adaptation_receipt(
            world_state,
            execution_contract,
        )
        backend_binding_receipt = _build_backend_execution_binding_receipt(
            world_state,
            execution_contract,
            adaptation_receipt,
        )
        calibration_receipt = build_physics_calibration_receipt(
            world_state,
            execution_contract,
            adaptation_receipt=adaptation_receipt,
        )
        training_feedback_path = artifact_paths.get("training_feedback_manifest")
        render_provider_receipts = _build_render_provider_receipts(
            world_state,
            execution_contract,
            adaptation_receipt,
        )
        outcome_receipts = _build_outcome_receipts(
            world_state,
            execution_contract,
            adaptation_receipt,
            backend_binding_receipt,
            calibration_receipt,
            training_feedback_path=training_feedback_path,
        )
        training_feedback_manifest = _build_training_feedback_manifest(
            world_state,
            execution_contract,
            adaptation_receipt,
            backend_binding_receipt,
            calibration_receipt,
            render_provider_receipts,
            outcome_receipts,
        )
        result = SimSynthPhysicsLoopResult(
            world_state=world_state,
            physics_execution_contract=execution_contract,
            physics_adaptation_receipt=adaptation_receipt,
            backend_execution_binding_receipt=backend_binding_receipt,
            physics_calibration_receipt=calibration_receipt,
            render_provider_receipts=render_provider_receipts,
            outcome_receipts=outcome_receipts,
            training_feedback_manifest=training_feedback_manifest,
            artifact_paths={
                key: str(path.resolve()) for key, path in artifact_paths.items()
            },
        )
        if artifact_paths:
            _write_json(artifact_paths["world_state"], world_state.to_dict())
            _write_json(
                artifact_paths["physics_execution_contract"],
                execution_contract.to_dict(),
            )
            _write_json(
                artifact_paths["physics_adaptation_receipt"],
                adaptation_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["backend_execution_binding_receipt"],
                backend_binding_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["physics_calibration_receipt"],
                calibration_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["render_provider_receipts"],
                {
                    "version": "render_provider_receipt_bundle_v1",
                    "receipts": [receipt.to_dict() for receipt in render_provider_receipts],
                },
            )
            _write_json(
                artifact_paths["simulation_outcome_receipts"],
                {
                    "version": "simulation_outcome_receipt_bundle_v1",
                    "receipts": [receipt.to_dict() for receipt in outcome_receipts],
                },
            )
            _write_json(
                artifact_paths["training_feedback_manifest"],
                training_feedback_manifest,
            )
            _write_json(
                artifact_paths["loop_summary"],
                {
                    "version": "sim_synth_physics_loop_summary_v1",
                    "world_state_id": world_state.state_id,
                    "physics_execution_contract_id": execution_contract.contract_id,
                    "physics_adaptation_receipt_id": adaptation_receipt.receipt_id,
                    "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
                    "physics_calibration_receipt_id": calibration_receipt.receipt_id,
                    "render_provider_receipt_count": len(render_provider_receipts),
                    "requested_backend": execution_contract.requested_backend,
                    "resolved_backend": execution_contract.resolved_backend,
                    "route_status": execution_contract.route_status,
                    "planned_branch_count": training_feedback_manifest.get(
                        "planned_branch_count",
                        0,
                    ),
                    "blocked_branch_count": training_feedback_manifest.get(
                        "blocked_branch_count",
                        0,
                    ),
                    "artifact_paths": result.artifact_paths,
                },
            )
        return result

    def run_planning_window(
        self,
        coverage_graph: Any,
        *,
        semantic_context: Optional[Mapping[str, Any]] = None,
        economic_context: Optional[Mapping[str, Any]] = None,
        embodiment_context: Optional[Mapping[str, Any]] = None,
        benchmark_signals: Optional[Mapping[str, Any]] = None,
        gap_ranker: Any = None,
        backend_selector: Any = None,
        branch_planner: Any = None,
        output_dir: str | Path | None = None,
    ) -> SimSynthPhysicsLoopResult:
        world_state = self.compile_world_state(
            coverage_graph,
            semantic_context=semantic_context,
            economic_context=economic_context,
            embodiment_context=embodiment_context,
            benchmark_signals=benchmark_signals,
            gap_ranker=gap_ranker,
            backend_selector=backend_selector,
            branch_planner=branch_planner,
        )
        result = self.execute_world_state(world_state, output_dir=output_dir)
        if output_dir is not None:
            artifact_paths = _artifact_paths(output_dir)
            diffusion_plans = self.compile_diffusion_plans(
                world_state,
                coverage_graph=coverage_graph,
            )
            _write_json(
                artifact_paths["diffusion_plans"],
                {
                    "version": "gap_driven_diffusion_plan_bundle_v1",
                    "plans": [plan.to_dict() for plan in diffusion_plans],
                },
            )
        return result


__all__ = [
    "SimSynthPhysicsLoopResult",
    "SimSynthPhysicsRuntime",
    "SimSynthPhysicsRuntimeConfig",
]

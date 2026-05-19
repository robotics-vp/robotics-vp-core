"""Runtime facade for the sim/synth/physics world model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional

from .backend_runtime_execution import materialize_backend_runtime_execution
from .backend_router import build_physics_execution_contract
from .calibration import (
    build_physics_adaptation_receipt,
    build_physics_calibration_receipt,
)
from .common import mapping, stable_id
from .compiler import compile_sim_synth_physics_world_state
from .diffusion_contracts import GapDrivenDiffusionPlan, compile_gap_driven_diffusion_plans
from .gen2sim_admission import build_gen2sim_admission_receipt
from .phase1x_receipts import (
    build_backend_mismatch_receipt,
    build_branch_validity_receipts,
    build_replay_validity_receipts,
    build_sensor_alignment_receipt,
    build_sim_real_gap_receipt,
    build_surrogate_calibration_receipt,
    build_surrogate_physics_receipt,
    build_task_measurement_receipt,
)
from .physics_contracts import PhysicsExecutionContract
from .promotion import HelperMode
from .render_materialization import materialize_render_provider_receipts
from .receipts import (
    BackendExecutionBindingReceipt,
    BackendRuntimeAdapterReceipt,
    BackendRuntimeBridgeReceipt,
    BackendRuntimeExecutionReceipt,
    BackendRuntimeLaunchReceipt,
    BackendRuntimeOutcomeReceipt,
    BackendRuntimeWorkOrderReceipt,
    BackendShadowExecutionReceipt,
    BranchValidityReceipt,
    BackendMismatchReceipt,
    ReplayValidityReceipt,
    SensorAlignmentReceipt,
    Gen2SimAdmissionReceipt,
    PhysicsAdaptationReceipt,
    PhysicsCalibrationReceipt,
    RenderProviderReceipt,
    RobotAssetContractReceipt,
    SimRealGapReceipt,
    SimulationOutcomeReceipt,
    SurrogateCalibrationReceipt,
    SurrogatePhysicsReceipt,
    TaskMeasurementReceipt,
)
from .runtime_evidence import summarize_runtime_evidence
from .runtime_bridge import build_backend_runtime_bridge_receipt
from .shadow_execution import materialize_backend_shadow_execution
from .runtime_work_orders import build_backend_runtime_work_orders
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
    gen2sim_admission_receipt: Gen2SimAdmissionReceipt
    backend_execution_binding_receipt: BackendExecutionBindingReceipt
    robot_asset_contract_receipt: RobotAssetContractReceipt
    backend_runtime_bridge_receipt: BackendRuntimeBridgeReceipt
    physics_calibration_receipt: PhysicsCalibrationReceipt
    task_measurement_receipt: TaskMeasurementReceipt
    sim_real_gap_receipt: SimRealGapReceipt
    backend_mismatch_receipt: BackendMismatchReceipt
    surrogate_physics_receipt: SurrogatePhysicsReceipt
    surrogate_calibration_receipt: SurrogateCalibrationReceipt
    branch_validity_receipts: list[BranchValidityReceipt] = field(default_factory=list)
    sensor_alignment_receipt: Optional[SensorAlignmentReceipt] = None
    replay_validity_receipts: list[ReplayValidityReceipt] = field(default_factory=list)
    backend_runtime_work_orders: list[BackendRuntimeWorkOrderReceipt] = field(default_factory=list)
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt] = None
    backend_runtime_adapter_receipt: Optional[BackendRuntimeAdapterReceipt] = None
    backend_runtime_launch_receipt: Optional[BackendRuntimeLaunchReceipt] = None
    backend_runtime_outcome_receipt: Optional[BackendRuntimeOutcomeReceipt] = None
    backend_shadow_execution_receipt: Optional[BackendShadowExecutionReceipt] = None
    render_provider_receipts: list[RenderProviderReceipt] = field(default_factory=list)
    outcome_receipts: list[SimulationOutcomeReceipt] = field(default_factory=list)
    training_feedback_manifest: Mapping[str, Any] = field(default_factory=dict)
    runtime_receipt_manifest: Mapping[str, Any] = field(default_factory=dict)
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    version: str = "sim_synth_physics_loop_result_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_state": self.world_state.to_dict(),
            "physics_execution_contract": self.physics_execution_contract.to_dict(),
            "physics_adaptation_receipt": self.physics_adaptation_receipt.to_dict(),
            "gen2sim_admission_receipt": self.gen2sim_admission_receipt.to_dict(),
            "backend_execution_binding_receipt": self.backend_execution_binding_receipt.to_dict(),
            "robot_asset_contract_receipt": self.robot_asset_contract_receipt.to_dict(),
            "backend_runtime_bridge_receipt": self.backend_runtime_bridge_receipt.to_dict(),
            "backend_runtime_work_orders": [
                receipt.to_dict() for receipt in self.backend_runtime_work_orders
            ],
            "backend_runtime_execution_receipt": (
                None
                if self.backend_runtime_execution_receipt is None
                else self.backend_runtime_execution_receipt.to_dict()
            ),
            "backend_runtime_adapter_receipt": (
                None
                if self.backend_runtime_adapter_receipt is None
                else self.backend_runtime_adapter_receipt.to_dict()
            ),
            "backend_runtime_launch_receipt": (
                None
                if self.backend_runtime_launch_receipt is None
                else self.backend_runtime_launch_receipt.to_dict()
            ),
            "backend_runtime_outcome_receipt": (
                None
                if self.backend_runtime_outcome_receipt is None
                else self.backend_runtime_outcome_receipt.to_dict()
            ),
            "backend_shadow_execution_receipt": (
                None
                if self.backend_shadow_execution_receipt is None
                else self.backend_shadow_execution_receipt.to_dict()
            ),
            "physics_calibration_receipt": self.physics_calibration_receipt.to_dict(),
            "task_measurement_receipt": self.task_measurement_receipt.to_dict(),
            "sim_real_gap_receipt": self.sim_real_gap_receipt.to_dict(),
            "backend_mismatch_receipt": self.backend_mismatch_receipt.to_dict(),
            "surrogate_physics_receipt": self.surrogate_physics_receipt.to_dict(),
            "surrogate_calibration_receipt": self.surrogate_calibration_receipt.to_dict(),
            "branch_validity_receipts": [
                receipt.to_dict() for receipt in self.branch_validity_receipts
            ],
            "sensor_alignment_receipt": (
                self.sensor_alignment_receipt.to_dict()
                if self.sensor_alignment_receipt is not None
                else None
            ),
            "replay_validity_receipts": [
                receipt.to_dict() for receipt in self.replay_validity_receipts
            ],
            "render_provider_receipts": [
                receipt.to_dict() for receipt in self.render_provider_receipts
            ],
            "outcome_receipts": [receipt.to_dict() for receipt in self.outcome_receipts],
            "training_feedback_manifest": mapping(self.training_feedback_manifest),
            "runtime_receipt_manifest": mapping(self.runtime_receipt_manifest),
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
        "gen2sim_admission_receipt": root / "gen2sim_admission_receipt.json",
        "backend_execution_binding_receipt": root / "backend_execution_binding_receipt.json",
        "robot_asset_contract_receipt": root / "robot_asset_contract_receipt.json",
        "backend_runtime_bridge_receipt": root / "backend_runtime_bridge_receipt.json",
        "backend_runtime_work_orders": root / "backend_runtime_work_orders.json",
        "backend_runtime_execution_receipt": root / "backend_runtime_execution_receipt.json",
        "backend_runtime_adapter_receipt": root / "backend_runtime_adapter_receipt.json",
        "backend_runtime_adapter_realization": root / "backend_runtime_adapter_realization.json",
        "backend_upstream_runtime_pack": root / "backend_upstream_runtime_pack.json",
        "backend_runtime_binding": root / "backend_runtime_binding.json",
        "backend_runtime_launch_receipt": root / "backend_runtime_launch_receipt.json",
        "backend_runtime_outcome_receipt": root / "backend_runtime_outcome_receipt.json",
        "backend_shadow_execution_receipt": root / "backend_shadow_execution_receipt.json",
        "physics_calibration_receipt": root / "physics_calibration_receipt.json",
        "task_measurement_receipt": root / "task_measurement_receipt.json",
        "sim_real_gap_receipt": root / "sim_real_gap_receipt.json",
        "backend_mismatch_receipt": root / "backend_mismatch_receipt.json",
        "surrogate_physics_receipt": root / "surrogate_physics_receipt.json",
        "surrogate_calibration_receipt": root / "surrogate_calibration_receipt.json",
        "branch_validity_receipts": root / "branch_validity_receipts.json",
        "sensor_alignment_receipt": root / "sensor_alignment_receipt.json",
        "replay_validity_receipts": root / "replay_validity_receipts.json",
        "render_provider_receipts": root / "render_provider_receipts.json",
        "simulation_outcome_receipts": root / "simulation_outcome_receipts.json",
        "training_feedback_manifest": root / "sim_synth_training_feedback.json",
        "runtime_receipt_manifest": root / "runtime_receipt_manifest.json",
        "loop_summary": root / "sim_synth_physics_loop_summary.json",
        "diffusion_plans": root / "gap_driven_diffusion_plans.json",
    }


def _is_materialized_render_status(status: str) -> bool:
    return str(status or "") in {
        "scene_materialized",
        "counterfactuals_materialized",
        "ggds_scene_materialized",
        "work_order_materialized",
        "work_order_materialized_with_preconditions",
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


def _build_backend_runtime_launch_receipt(
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt],
) -> Optional[BackendRuntimeLaunchReceipt]:
    if backend_runtime_execution_receipt is None:
        return None
    payload = mapping(backend_runtime_execution_receipt.metadata.get("launch_receipt"))
    if not payload:
        return None
    return BackendRuntimeLaunchReceipt(
        receipt_id=str(payload.get("receipt_id", "") or ""),
        backend=str(payload.get("backend", "") or ""),
        launch_profile=str(payload.get("launch_profile", "") or ""),
        launch_status=str(payload.get("launch_status", "") or ""),
        executed=bool(payload.get("executed", False)),
        command=str(payload.get("command", "") or ""),
        cwd=str(payload.get("cwd", "") or ""),
        artifact_refs=list(payload.get("artifact_refs", []) or []),
        metadata=mapping(payload.get("metadata")),
    )


def _build_backend_runtime_adapter_receipt(
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt],
) -> Optional[BackendRuntimeAdapterReceipt]:
    if backend_runtime_execution_receipt is None:
        return None
    payload = mapping(backend_runtime_execution_receipt.metadata.get("adapter_receipt"))
    if not payload:
        return None
    return BackendRuntimeAdapterReceipt(
        receipt_id=str(payload.get("receipt_id", "") or ""),
        backend=str(payload.get("backend", "") or ""),
        adapter_family=str(payload.get("adapter_family", "") or ""),
        adapter_entrypoint=str(payload.get("adapter_entrypoint", "") or ""),
        consumer_mode=str(payload.get("consumer_mode", "") or ""),
        adapter_status=str(payload.get("adapter_status", "") or ""),
        execution_path=str(payload.get("execution_path", "") or ""),
        executed=bool(payload.get("executed", False)),
        artifact_refs=list(payload.get("artifact_refs", []) or []),
        metadata=mapping(payload.get("metadata")),
    )


def _build_backend_runtime_outcome_receipt(
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt],
) -> Optional[BackendRuntimeOutcomeReceipt]:
    if backend_runtime_execution_receipt is None:
        return None
    payload = mapping(backend_runtime_execution_receipt.metadata.get("runtime_outcome_receipt"))
    if not payload:
        return None
    return BackendRuntimeOutcomeReceipt(
        receipt_id=str(payload.get("receipt_id", "") or ""),
        backend=str(payload.get("backend", "") or ""),
        outcome_profile=str(payload.get("outcome_profile", "") or ""),
        outcome_status=str(payload.get("outcome_status", "") or ""),
        executed=bool(payload.get("executed", False)),
        harvested_output_count=int(payload.get("harvested_output_count", 0) or 0),
        artifact_refs=list(payload.get("artifact_refs", []) or []),
        metadata=mapping(payload.get("metadata")),
    )


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
    robot_asset_contract_receipt: RobotAssetContractReceipt,
    backend_runtime_bridge_receipt: BackendRuntimeBridgeReceipt,
    backend_runtime_work_orders: list[BackendRuntimeWorkOrderReceipt],
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt],
    backend_runtime_adapter_receipt: Optional[BackendRuntimeAdapterReceipt],
    calibration_receipt: PhysicsCalibrationReceipt,
    *,
    backend_shadow_execution_receipt: Optional[BackendShadowExecutionReceipt] = None,
    render_provider_receipts: Optional[list[RenderProviderReceipt]] = None,
    training_feedback_path: Optional[Path] = None,
) -> list[SimulationOutcomeReceipt]:
    admissible_branch_ids = set(
        getattr(world_state.gen2sim_admission, "admissible_branch_ids", []) or []
    )
    job_by_id = {job.job_id: job for job in world_state.simulation_agenda.jobs}
    render_receipts_by_plan = {
        str(receipt.branch_plan_id): receipt for receipt in (render_provider_receipts or [])
    }
    receipts: list[SimulationOutcomeReceipt] = []
    for index, plan in enumerate(world_state.synthetic_branch_plans):
        job = job_by_id.get(plan.source_job_id)
        render_receipt = render_receipts_by_plan.get(str(plan.plan_id))
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
        adapter_realization = (
            {}
            if backend_runtime_adapter_receipt is None
            else mapping(backend_runtime_adapter_receipt.metadata.get("realization"))
        )
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
                    "artifact_materialization": (
                        "planned_only"
                        if render_receipt is None
                        else str(render_receipt.materialization_status)
                    ),
                    "render_materialization_mode": (
                        "" if render_receipt is None else str(render_receipt.materialization_mode)
                    ),
                    "render_provider_receipt_id": (
                        "" if render_receipt is None else str(render_receipt.receipt_id)
                    ),
                    "render_artifact_refs": (
                        [] if render_receipt is None else list(render_receipt.artifact_refs)
                    ),
                    "render_unsatisfied_preconditions": (
                        []
                        if render_receipt is None
                        else list(render_receipt.metadata.get("unsatisfied_preconditions", []) or [])
                    ),
                    "backend_shadow_execution_receipt_id": (
                        ""
                        if backend_shadow_execution_receipt is None
                        else str(backend_shadow_execution_receipt.receipt_id)
                    ),
                    "backend_shadow_execution_status": (
                        ""
                        if backend_shadow_execution_receipt is None
                        else str(backend_shadow_execution_receipt.execution_status)
                    ),
                    "backend_runtime_execution_receipt_id": (
                        ""
                        if backend_runtime_execution_receipt is None
                        else str(backend_runtime_execution_receipt.receipt_id)
                    ),
                    "backend_runtime_execution_status": (
                        ""
                        if backend_runtime_execution_receipt is None
                        else str(backend_runtime_execution_receipt.execution_status)
                    ),
                    "backend_runtime_adapter_receipt_id": (
                        ""
                        if backend_runtime_adapter_receipt is None
                        else str(backend_runtime_adapter_receipt.receipt_id)
                    ),
                    "backend_runtime_adapter_status": (
                        ""
                        if backend_runtime_adapter_receipt is None
                        else str(backend_runtime_adapter_receipt.adapter_status)
                    ),
                    "backend_runtime_adapter_execution_path": (
                        ""
                        if backend_runtime_adapter_receipt is None
                        else str(backend_runtime_adapter_receipt.execution_path)
                    ),
                    "backend_runtime_adapter_realization_path": str(
                        adapter_realization.get("realization_path", "") or ""
                    ),
                    "backend_runtime_adapter_realization_status": str(
                        adapter_realization.get("realization_status", "") or ""
                    ),
                    "backend_runtime_bridge_receipt_id": (
                        backend_runtime_bridge_receipt.receipt_id
                    ),
                    "backend_runtime_bridge_status": (
                        backend_runtime_bridge_receipt.bridge_status
                    ),
                    "backend_runtime_bridge_execution_authority": (
                        backend_runtime_bridge_receipt.execution_authority
                    ),
                    "bridge_transport_profile": (
                        backend_runtime_bridge_receipt.transport_profile
                    ),
                    "bridge_readiness_score": (
                        float(backend_runtime_bridge_receipt.bridge_readiness_score)
                    ),
                    "backend_runtime_work_order_receipt_ids": [
                        receipt.receipt_id for receipt in backend_runtime_work_orders
                    ],
                    "backend_runtime_work_order_statuses": [
                        receipt.status for receipt in backend_runtime_work_orders
                    ],
                    "robot_asset_contract_receipt_id": robot_asset_contract_receipt.receipt_id,
                    "robot_asset_readiness_score": float(
                        robot_asset_contract_receipt.readiness_score
                    ),
                    "inferential_learnability_contract": mapping(
                        plan.inferential_learnability_contract
                    ),
                },
            )
        )
    return receipts



def _manifest_artifact_ref(artifact_paths: Mapping[str, Path], artifact_key: str) -> str:
    artifact_path = artifact_paths.get(artifact_key)
    return "" if artifact_path is None else str(artifact_path.resolve())


def _runtime_manifest_entry(
    *,
    family: str,
    artifact_key: str,
    artifact_paths: Mapping[str, Path],
    receipt_ids: list[str],
    required: bool,
    group: str,
) -> dict[str, Any]:
    emitted = bool(receipt_ids)
    return {
        "family": family,
        "artifact_key": artifact_key,
        "artifact_path": _manifest_artifact_ref(artifact_paths, artifact_key),
        "required": bool(required),
        "group": group,
        "status": "emitted" if emitted else "not_emitted",
        "receipt_ids": list(receipt_ids),
        "receipt_count": len(receipt_ids),
    }


def _build_runtime_receipt_manifest(
    *,
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    artifact_paths: Mapping[str, Path],
    training_feedback_manifest: Mapping[str, Any],
    physics_adaptation_receipt: PhysicsAdaptationReceipt,
    gen2sim_admission_receipt: Gen2SimAdmissionReceipt,
    backend_execution_binding_receipt: BackendExecutionBindingReceipt,
    robot_asset_contract_receipt: RobotAssetContractReceipt,
    backend_runtime_bridge_receipt: BackendRuntimeBridgeReceipt,
    backend_runtime_work_orders: list[BackendRuntimeWorkOrderReceipt],
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt],
    backend_runtime_adapter_receipt: Optional[BackendRuntimeAdapterReceipt],
    backend_runtime_launch_receipt: Optional[BackendRuntimeLaunchReceipt],
    backend_runtime_outcome_receipt: Optional[BackendRuntimeOutcomeReceipt],
    backend_shadow_execution_receipt: Optional[BackendShadowExecutionReceipt],
    physics_calibration_receipt: PhysicsCalibrationReceipt,
    task_measurement_receipt: TaskMeasurementReceipt,
    sim_real_gap_receipt: SimRealGapReceipt,
    backend_mismatch_receipt: BackendMismatchReceipt,
    surrogate_physics_receipt: SurrogatePhysicsReceipt,
    surrogate_calibration_receipt: SurrogateCalibrationReceipt,
    branch_validity_receipts: list[BranchValidityReceipt],
    sensor_alignment_receipt: SensorAlignmentReceipt,
    replay_validity_receipts: list[ReplayValidityReceipt],
    render_provider_receipts: list[RenderProviderReceipt],
    outcome_receipts: list[SimulationOutcomeReceipt],
) -> dict[str, Any]:
    entries = [
        _runtime_manifest_entry(
            family="physics_adaptation_receipt_v1",
            artifact_key="physics_adaptation_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[physics_adaptation_receipt.receipt_id],
            required=True,
            group="runtime_window",
        ),
        _runtime_manifest_entry(
            family="gen2sim_admission_receipt_v1",
            artifact_key="gen2sim_admission_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[gen2sim_admission_receipt.receipt_id],
            required=True,
            group="admission",
        ),
        _runtime_manifest_entry(
            family="backend_execution_binding_receipt_v1",
            artifact_key="backend_execution_binding_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[backend_execution_binding_receipt.receipt_id],
            required=True,
            group="runtime_binding",
        ),
        _runtime_manifest_entry(
            family="robot_asset_contract_receipt_v1",
            artifact_key="robot_asset_contract_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[robot_asset_contract_receipt.receipt_id],
            required=True,
            group="runtime_binding",
        ),
        _runtime_manifest_entry(
            family="backend_runtime_bridge_receipt_v1",
            artifact_key="backend_runtime_bridge_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[backend_runtime_bridge_receipt.receipt_id],
            required=True,
            group="runtime_binding",
        ),
        _runtime_manifest_entry(
            family="backend_runtime_work_order_receipt_v1",
            artifact_key="backend_runtime_work_orders",
            artifact_paths=artifact_paths,
            receipt_ids=[receipt.receipt_id for receipt in backend_runtime_work_orders],
            required=True,
            group="runtime_work_orders",
        ),
        _runtime_manifest_entry(
            family="backend_runtime_execution_receipt_v1",
            artifact_key="backend_runtime_execution_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=(
                []
                if backend_runtime_execution_receipt is None
                else [backend_runtime_execution_receipt.receipt_id]
            ),
            required=False,
            group="optional_runtime",
        ),
        _runtime_manifest_entry(
            family="backend_runtime_adapter_receipt_v1",
            artifact_key="backend_runtime_adapter_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=(
                []
                if backend_runtime_adapter_receipt is None
                else [backend_runtime_adapter_receipt.receipt_id]
            ),
            required=False,
            group="optional_runtime",
        ),
        _runtime_manifest_entry(
            family="backend_runtime_launch_receipt_v1",
            artifact_key="backend_runtime_launch_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=(
                []
                if backend_runtime_launch_receipt is None
                else [backend_runtime_launch_receipt.receipt_id]
            ),
            required=False,
            group="optional_runtime",
        ),
        _runtime_manifest_entry(
            family="backend_runtime_outcome_receipt_v1",
            artifact_key="backend_runtime_outcome_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=(
                []
                if backend_runtime_outcome_receipt is None
                else [backend_runtime_outcome_receipt.receipt_id]
            ),
            required=False,
            group="optional_runtime",
        ),
        _runtime_manifest_entry(
            family="backend_shadow_execution_receipt_v1",
            artifact_key="backend_shadow_execution_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=(
                []
                if backend_shadow_execution_receipt is None
                else [backend_shadow_execution_receipt.receipt_id]
            ),
            required=False,
            group="optional_runtime",
        ),
        _runtime_manifest_entry(
            family="physics_calibration_receipt_v1",
            artifact_key="physics_calibration_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[physics_calibration_receipt.receipt_id],
            required=True,
            group="transfer",
        ),
        _runtime_manifest_entry(
            family="task_measurement_receipt_v1",
            artifact_key="task_measurement_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[task_measurement_receipt.receipt_id],
            required=True,
            group="task",
        ),
        _runtime_manifest_entry(
            family="sim_real_gap_receipt_v1",
            artifact_key="sim_real_gap_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[sim_real_gap_receipt.receipt_id],
            required=True,
            group="transfer",
        ),
        _runtime_manifest_entry(
            family="backend_mismatch_receipt_v1",
            artifact_key="backend_mismatch_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[backend_mismatch_receipt.receipt_id],
            required=True,
            group="transfer",
        ),
        _runtime_manifest_entry(
            family="surrogate_physics_receipt_v1",
            artifact_key="surrogate_physics_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[surrogate_physics_receipt.receipt_id],
            required=True,
            group="surrogate",
        ),
        _runtime_manifest_entry(
            family="surrogate_calibration_receipt_v1",
            artifact_key="surrogate_calibration_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[surrogate_calibration_receipt.receipt_id],
            required=True,
            group="surrogate",
        ),
        _runtime_manifest_entry(
            family="branch_validity_receipt_v1",
            artifact_key="branch_validity_receipts",
            artifact_paths=artifact_paths,
            receipt_ids=[receipt.receipt_id for receipt in branch_validity_receipts],
            required=True,
            group="per_branch_filter",
        ),
        _runtime_manifest_entry(
            family="sensor_alignment_receipt_v1",
            artifact_key="sensor_alignment_receipt",
            artifact_paths=artifact_paths,
            receipt_ids=[sensor_alignment_receipt.receipt_id],
            required=True,
            group="sensor_geometry",
        ),
        _runtime_manifest_entry(
            family="replay_validity_receipt_v1",
            artifact_key="replay_validity_receipts",
            artifact_paths=artifact_paths,
            receipt_ids=[receipt.receipt_id for receipt in replay_validity_receipts],
            required=True,
            group="per_branch_filter",
        ),
        _runtime_manifest_entry(
            family="render_provider_receipt_v1",
            artifact_key="render_provider_receipts",
            artifact_paths=artifact_paths,
            receipt_ids=[receipt.receipt_id for receipt in render_provider_receipts],
            required=True,
            group="render_materialization",
        ),
        _runtime_manifest_entry(
            family="simulation_outcome_receipt_v1",
            artifact_key="simulation_outcome_receipts",
            artifact_paths=artifact_paths,
            receipt_ids=[receipt.receipt_id for receipt in outcome_receipts],
            required=True,
            group="runtime_outcomes",
        ),
    ]
    missing_required = [
        str(entry["family"])
        for entry in entries
        if entry["required"] and entry["status"] != "emitted"
    ]
    optional_not_emitted = [
        str(entry["family"])
        for entry in entries
        if not entry["required"] and entry["status"] != "emitted"
    ]
    family_counts = {str(entry["family"]): int(entry["receipt_count"]) for entry in entries}
    emitted_receipt_ids = {
        str(entry["family"]): list(entry["receipt_ids"])
        for entry in entries
        if entry["receipt_ids"]
    }
    payload = {
        "world_state_id": world_state.state_id,
        "physics_execution_contract_id": execution_contract.contract_id,
        "emitted_receipt_ids": emitted_receipt_ids,
        "missing_required": missing_required,
    }
    manifest_id = stable_id("sim_synth_runtime_receipt_manifest", payload)
    return {
        "version": "sim_synth_runtime_receipt_manifest_v1",
        "manifest_id": manifest_id,
        "world_state_id": world_state.state_id,
        "physics_execution_contract_id": execution_contract.contract_id,
        "compiled_receipt_inventory_id": str(
            mapping(world_state.metadata.get("compiled_receipt_inventory")).get(
                "inventory_id", ""
            )
            or ""
        ),
        "manifest_status": "complete" if not missing_required else "missing_required",
        "missing_required_families": missing_required,
        "optional_not_emitted_families": optional_not_emitted,
        "receipt_family_counts": family_counts,
        "emitted_receipt_ids": emitted_receipt_ids,
        "emitted_receipt_count": sum(family_counts.values()),
        "artifact_entries": entries,
        "training_feedback_manifest_ref": _manifest_artifact_ref(
            artifact_paths, "training_feedback_manifest"
        ),
        "training_feedback_row_count": len(list(training_feedback_manifest.get("rows") or [])),
        "route_status": execution_contract.route_status,
        "requested_backend": execution_contract.requested_backend,
        "resolved_backend": execution_contract.resolved_backend,
        "metadata": {
            "artifact_root": (
                ""
                if not artifact_paths
                else str(next(iter(artifact_paths.values())).parent.resolve())
            ),
            "promotion_posture": "local_receipt_manifest_only",
            "provider_truth_claim": "no_provider_bringup_claimed",
        },
    }

def _build_training_feedback_manifest(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    adaptation_receipt: PhysicsAdaptationReceipt,
    gen2sim_admission_receipt: Gen2SimAdmissionReceipt,
    backend_binding_receipt: BackendExecutionBindingReceipt,
    robot_asset_contract_receipt: RobotAssetContractReceipt,
    backend_runtime_bridge_receipt: BackendRuntimeBridgeReceipt,
    backend_runtime_work_orders: list[BackendRuntimeWorkOrderReceipt],
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt],
    backend_runtime_adapter_receipt: Optional[BackendRuntimeAdapterReceipt],
    backend_runtime_launch_receipt: Optional[BackendRuntimeLaunchReceipt],
    backend_runtime_outcome_receipt: Optional[BackendRuntimeOutcomeReceipt],
    backend_shadow_execution_receipt: Optional[BackendShadowExecutionReceipt],
    calibration_receipt: PhysicsCalibrationReceipt,
    task_measurement_receipt: TaskMeasurementReceipt,
    sim_real_gap_receipt: SimRealGapReceipt,
    backend_mismatch_receipt: BackendMismatchReceipt,
    surrogate_physics_receipt: SurrogatePhysicsReceipt,
    surrogate_calibration_receipt: SurrogateCalibrationReceipt,
    branch_validity_receipts: list[BranchValidityReceipt],
    sensor_alignment_receipt: SensorAlignmentReceipt,
    replay_validity_receipts: list[ReplayValidityReceipt],
    render_provider_receipts: list[RenderProviderReceipt],
    outcome_receipts: list[SimulationOutcomeReceipt],
) -> dict[str, Any]:
    render_receipts_by_plan = {
        str(receipt.branch_plan_id): receipt for receipt in render_provider_receipts
    }
    transfer_evidence = {
        "sim_real_gap_receipt_id": sim_real_gap_receipt.receipt_id,
        "sim_real_gap_status": sim_real_gap_receipt.status,
        "sim_real_gap_score": float(sim_real_gap_receipt.gap_score),
        "sim_real_realism_confidence": float(sim_real_gap_receipt.realism_confidence),
        "backend_mismatch_receipt_id": backend_mismatch_receipt.receipt_id,
        "backend_mismatch_status": backend_mismatch_receipt.status,
        "backend_mismatch_score": float(backend_mismatch_receipt.mismatch_score),
        "backend_calibration_staleness_score": float(
            backend_mismatch_receipt.calibration_staleness_score
        ),
        "surrogate_physics_receipt_id": surrogate_physics_receipt.receipt_id,
        "surrogate_forecast_status": surrogate_physics_receipt.forecast_status,
        "surrogate_confidence": float(surrogate_physics_receipt.surrogate_confidence),
        "surrogate_calibration_receipt_id": surrogate_calibration_receipt.receipt_id,
        "surrogate_calibration_status": surrogate_calibration_receipt.calibration_status,
        "surrogate_calibration_score": float(
            surrogate_calibration_receipt.calibration_score
        ),
    }
    adapter_realization = (
        {}
        if backend_runtime_adapter_receipt is None
        else mapping(backend_runtime_adapter_receipt.metadata.get("realization"))
    )
    runtime_binding = (
        {}
        if backend_runtime_execution_receipt is None
        else mapping(
            mapping(backend_runtime_execution_receipt.metadata).get("runtime_binding")
            or mapping(backend_runtime_execution_receipt.metadata).get("runtime_bundle", {}).get(
                "runtime_binding"
            )
        )
    )
    rows: list[dict[str, Any]] = []
    branch_validity_by_plan = {
        receipt.branch_plan_id: receipt for receipt in branch_validity_receipts
    }
    replay_validity_by_plan = {
        receipt.branch_plan_id: receipt for receipt in replay_validity_receipts
    }
    for receipt in outcome_receipts:
        render_receipt = render_receipts_by_plan.get(str(receipt.branch_plan_id))
        branch_validity_receipt = branch_validity_by_plan.get(str(receipt.branch_plan_id))
        replay_validity_receipt = replay_validity_by_plan.get(str(receipt.branch_plan_id))
        rows.append(
            {
                "branch_plan_id": str(receipt.branch_plan_id),
                "job_id": str(receipt.job_id),
                "status": str(receipt.status),
                "training_feedback_refs": list(receipt.training_feedback_refs),
                "render_provider_receipt_ref": (
                    None if render_receipt is None else render_receipt.receipt_id
                ),
                "render_artifact_refs": (
                    [] if render_receipt is None else list(render_receipt.artifact_refs)
                ),
                "render_materialization_status": (
                    "" if render_receipt is None else str(render_receipt.materialization_status)
                ),
                "render_unsatisfied_preconditions": (
                    []
                    if render_receipt is None
                    else list(render_receipt.metadata.get("unsatisfied_preconditions", []) or [])
                ),
                "transfer_evidence": dict(transfer_evidence),
                "branch_validity": (
                    {}
                    if branch_validity_receipt is None
                    else branch_validity_receipt.to_dict()
                ),
                "sensor_alignment": sensor_alignment_receipt.to_dict(),
                "replay_validity": (
                    {}
                    if replay_validity_receipt is None
                    else replay_validity_receipt.to_dict()
                ),
                "metadata": mapping(receipt.metadata),
            }
        )
    return {
        "version": "sim_synth_training_feedback_v1",
        "world_state_id": world_state.state_id,
        "physics_execution_contract_id": execution_contract.contract_id,
        "physics_adaptation_receipt_id": adaptation_receipt.receipt_id,
        "gen2sim_admission_receipt_id": gen2sim_admission_receipt.receipt_id,
        "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
        "robot_asset_contract_receipt_id": robot_asset_contract_receipt.receipt_id,
        "backend_runtime_bridge_receipt_id": backend_runtime_bridge_receipt.receipt_id,
        "backend_runtime_work_order_receipt_ids": [
            receipt.receipt_id for receipt in backend_runtime_work_orders
        ],
        "backend_runtime_execution_receipt_id": (
            None
            if backend_runtime_execution_receipt is None
            else backend_runtime_execution_receipt.receipt_id
        ),
        "backend_runtime_adapter_receipt_id": (
            None
            if backend_runtime_adapter_receipt is None
            else backend_runtime_adapter_receipt.receipt_id
        ),
        "backend_runtime_launch_receipt_id": (
            None
            if backend_runtime_launch_receipt is None
            else backend_runtime_launch_receipt.receipt_id
        ),
        "backend_runtime_outcome_receipt_id": (
            None
            if backend_runtime_outcome_receipt is None
            else backend_runtime_outcome_receipt.receipt_id
        ),
        "backend_shadow_execution_receipt_id": (
            None
            if backend_shadow_execution_receipt is None
            else backend_shadow_execution_receipt.receipt_id
        ),
        "physics_calibration_receipt_id": calibration_receipt.receipt_id,
        "task_measurement_receipt_id": task_measurement_receipt.receipt_id,
        "sim_real_gap_receipt_id": sim_real_gap_receipt.receipt_id,
        "backend_mismatch_receipt_id": backend_mismatch_receipt.receipt_id,
        "surrogate_physics_receipt_id": surrogate_physics_receipt.receipt_id,
        "surrogate_calibration_receipt_id": surrogate_calibration_receipt.receipt_id,
        "transfer_evidence": transfer_evidence,
        "branch_validity_receipt_ids": [
            receipt.receipt_id for receipt in branch_validity_receipts
        ],
        "branch_validity_reject_count": sum(
            1 for receipt in branch_validity_receipts if not receipt.admissible
        ),
        "sensor_alignment_receipt_id": sensor_alignment_receipt.receipt_id,
        "sensor_alignment_status": sensor_alignment_receipt.status,
        "sensor_alignment_score": sensor_alignment_receipt.alignment_score,
        "replay_validity_receipt_ids": [
            receipt.receipt_id for receipt in replay_validity_receipts
        ],
        "replay_validity_reject_count": sum(
            1 for receipt in replay_validity_receipts if receipt.reject_reasons
        ),
        "route_status": execution_contract.route_status,
        "gen2sim_benchmark_gate_ready": bool(gen2sim_admission_receipt.benchmark_gate_ready),
        "gen2sim_admissible_branch_count": len(gen2sim_admission_receipt.admissible_branch_ids),
        "gen2sim_blocked_branch_count": len(gen2sim_admission_receipt.blocked_branch_ids),
        "backend_shadow_execution_status": (
            ""
            if backend_shadow_execution_receipt is None
            else backend_shadow_execution_receipt.execution_status
        ),
        "backend_runtime_execution_status": (
            ""
            if backend_runtime_execution_receipt is None
            else backend_runtime_execution_receipt.execution_status
        ),
        "backend_runtime_adapter_status": (
            ""
            if backend_runtime_adapter_receipt is None
            else backend_runtime_adapter_receipt.adapter_status
        ),
        "backend_runtime_adapter_execution_path": (
            ""
            if backend_runtime_adapter_receipt is None
            else backend_runtime_adapter_receipt.execution_path
        ),
        "backend_runtime_binding_status": str(runtime_binding.get("binding_status", "") or ""),
        "backend_runtime_binding_selected_profile": str(
            runtime_binding.get("selected_profile", "") or ""
        ),
        "backend_runtime_binding_selected_policy_ref": str(
            runtime_binding.get("selected_policy_ref", "") or ""
        ),
        "backend_runtime_adapter_realization_path": str(
            adapter_realization.get("realization_path", "") or ""
        ),
        "backend_runtime_adapter_realization_status": str(
            adapter_realization.get("realization_status", "") or ""
        ),
        "backend_runtime_launch_status": (
            ""
            if backend_runtime_launch_receipt is None
            else backend_runtime_launch_receipt.launch_status
        ),
        "backend_runtime_outcome_status": (
            ""
            if backend_runtime_outcome_receipt is None
            else backend_runtime_outcome_receipt.outcome_status
        ),
        "backend_runtime_output_count": (
            0
            if backend_runtime_outcome_receipt is None
            else backend_runtime_outcome_receipt.harvested_output_count
        ),
        "requested_backend": execution_contract.requested_backend,
        "resolved_backend": execution_contract.resolved_backend,
        "backend_runtime_bridge_status": backend_runtime_bridge_receipt.bridge_status,
        "bridge_execution_authority": backend_runtime_bridge_receipt.execution_authority,
        "bridge_transport_profile": backend_runtime_bridge_receipt.transport_profile,
        "bridge_readiness_score": float(backend_runtime_bridge_receipt.bridge_readiness_score),
        "backend_upstream_runtime_pack": mapping(
            backend_runtime_bridge_receipt.metadata.get("upstream_runtime_pack")
        ),
        "backend_upstream_runtime_pack_status": str(
            mapping(backend_runtime_bridge_receipt.metadata.get("upstream_runtime_pack")).get(
                "pack_status", ""
            )
            or ""
        ),
        "backend_runtime_work_order_count": len(backend_runtime_work_orders),
        "backend_runtime_work_order_statuses": [
            receipt.status for receipt in backend_runtime_work_orders
        ],
        "robot_asset_readiness_score": float(robot_asset_contract_receipt.readiness_score),
        "render_provider_receipt_count": len(render_provider_receipts),
        "materialized_render_provider_count": sum(
            1
            for receipt in render_provider_receipts
            if _is_materialized_render_status(str(receipt.materialization_status))
        ),
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
    adaptation_receipt: PhysicsAdaptationReceipt | None = None,
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
                "physics_adaptation_receipt_id": (
                    "" if adaptation_receipt is None else adaptation_receipt.receipt_id
                ),
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
            "physics_adaptation_receipt_id": (
                "" if adaptation_receipt is None else adaptation_receipt.receipt_id
            ),
            "executor_kind": binding.executor_kind,
            "observation_adapter_entrypoint": binding.observation_adapter_entrypoint,
            "target_runtime_stack": list(binding.target_runtime_stack),
            "required_assets": list(binding.required_assets),
            "available_assets": list(binding.available_assets),
            "missing_assets": list(binding.missing_assets),
            "runtime_target_contract": mapping(binding.metadata).get(
                "runtime_target_contract", {}
            ),
            "runtime_layout_contract": mapping(binding.metadata).get(
                "runtime_layout_contract", {}
            ),
            "policy_contract": mapping(binding.metadata).get("policy_contract", {}),
            "deployment_contract": mapping(binding.metadata).get("deployment_contract", {}),
            "upstream_runtime_pack": mapping(binding.metadata).get("upstream_runtime_pack", {}),
            "normalized_asset_manifest": mapping(binding.metadata).get(
                "normalized_asset_manifest", {}
            ),
            "binding_metadata": mapping(binding.metadata),
        },
    )


def _build_robot_asset_contract_receipt(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    backend_binding_receipt: BackendExecutionBindingReceipt,
) -> RobotAssetContractReceipt:
    contract = world_state.robot_asset_contract
    if contract is None:
        return RobotAssetContractReceipt(
            receipt_id=f"robot_asset_contract_receipt_{world_state.state_id}",
            contract_id="",
            asset_profile="",
            target_hardware_class=execution_contract.target_hardware_class,
            readiness_score=0.0,
            metadata={
                "world_state_id": world_state.state_id,
                "physics_execution_contract_id": execution_contract.contract_id,
                "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
                "binding_status": backend_binding_receipt.binding_status,
            },
        )
    return RobotAssetContractReceipt(
        receipt_id=f"robot_asset_contract_receipt_{world_state.state_id}",
        contract_id=contract.contract_id,
        asset_profile=contract.asset_profile,
        target_hardware_class=contract.target_hardware_class,
        readiness_score=float(contract.metadata.get("asset_readiness_score", 0.0) or 0.0),
        required_assets=list(contract.required_assets),
        available_assets=list(contract.available_assets),
        missing_assets=list(contract.missing_assets),
        calibration_contracts=list(contract.calibration_contracts),
        observation_contracts=list(contract.observation_contracts),
        action_contracts=list(contract.action_contracts),
        metadata={
            "world_state_id": world_state.state_id,
            "physics_execution_contract_id": execution_contract.contract_id,
            "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
            "binding_status": backend_binding_receipt.binding_status,
            "contract_metadata": mapping(contract.metadata),
        },
    )


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
            fallback_backend=self.config.fallback_backend,
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
        execute_external_runtime_launch: bool = False,
        external_launch_cwd: str | Path | None = None,
    ) -> SimSynthPhysicsLoopResult:
        artifact_paths = _artifact_paths(output_dir) if output_dir is not None else {}
        execution_contract = world_state.physics_execution_contract or build_physics_execution_contract(
            world_state,
            fallback_backend=self.config.fallback_backend,
        )
        backend_binding_receipt = _build_backend_execution_binding_receipt(
            world_state,
            execution_contract,
        )
        backend_runtime_execution_receipt = materialize_backend_runtime_execution(
            world_state,
            execution_contract,
            backend_binding_receipt,
            output_dir=output_dir,
            execute_external_launch=execute_external_runtime_launch,
            external_launch_cwd=external_launch_cwd,
        )
        backend_runtime_adapter_receipt = _build_backend_runtime_adapter_receipt(
            backend_runtime_execution_receipt
        )
        backend_runtime_launch_receipt = _build_backend_runtime_launch_receipt(
            backend_runtime_execution_receipt
        )
        backend_runtime_outcome_receipt = _build_backend_runtime_outcome_receipt(
            backend_runtime_execution_receipt
        )
        gen2sim_admission_receipt = build_gen2sim_admission_receipt(
            world_state.gen2sim_admission,
            world_state.synthetic_branch_plans,
            world_state.simulation_agenda.jobs,
        )
        backend_shadow_execution_receipt = materialize_backend_shadow_execution(
            world_state,
            execution_contract,
            backend_binding_receipt,
            backend_runtime_execution_receipt=backend_runtime_execution_receipt,
            backend_runtime_adapter_receipt=backend_runtime_adapter_receipt,
            backend_runtime_launch_receipt=backend_runtime_launch_receipt,
            backend_runtime_outcome_receipt=backend_runtime_outcome_receipt,
            output_dir=output_dir,
        )
        training_feedback_path = artifact_paths.get("training_feedback_manifest")
        runtime_evidence = summarize_runtime_evidence(
            backend_runtime_execution_receipt=backend_runtime_execution_receipt,
            backend_runtime_launch_receipt=backend_runtime_launch_receipt,
            backend_runtime_outcome_receipt=backend_runtime_outcome_receipt,
            backend_shadow_execution_receipt=backend_shadow_execution_receipt,
            render_provider_receipts=[],
            outcome_receipts=[],
        )
        adaptation_receipt = build_physics_adaptation_receipt(
            world_state,
            execution_contract,
            runtime_evidence=runtime_evidence,
        )
        render_provider_receipts = materialize_render_provider_receipts(
            world_state,
            execution_contract,
            adaptation_receipt,
            output_dir=output_dir,
        )
        runtime_evidence = summarize_runtime_evidence(
            backend_runtime_execution_receipt=backend_runtime_execution_receipt,
            backend_runtime_launch_receipt=backend_runtime_launch_receipt,
            backend_runtime_outcome_receipt=backend_runtime_outcome_receipt,
            backend_shadow_execution_receipt=backend_shadow_execution_receipt,
            render_provider_receipts=render_provider_receipts,
            outcome_receipts=[],
        )
        adaptation_receipt = build_physics_adaptation_receipt(
            world_state,
            execution_contract,
            runtime_evidence=runtime_evidence,
        )
        backend_binding_receipt = _build_backend_execution_binding_receipt(
            world_state,
            execution_contract,
            adaptation_receipt,
        )
        robot_asset_contract_receipt = _build_robot_asset_contract_receipt(
            world_state,
            execution_contract,
            backend_binding_receipt,
        )
        backend_runtime_bridge_receipt = build_backend_runtime_bridge_receipt(
            bridge_state=world_state.backend_runtime_bridge,
            backend_binding_receipt_id=backend_binding_receipt.receipt_id,
            robot_asset_contract_receipt=robot_asset_contract_receipt,
            backend_runtime_execution_receipt=backend_runtime_execution_receipt,
            backend_shadow_execution_receipt=backend_shadow_execution_receipt,
            world_state_id=world_state.state_id,
            physics_execution_contract_id=execution_contract.contract_id,
            route_status=execution_contract.route_status,
            requested_backend=execution_contract.requested_backend,
            resolved_backend=execution_contract.resolved_backend,
            fallback_reason=execution_contract.fallback_reason,
        )
        backend_runtime_work_orders = build_backend_runtime_work_orders(
            bridge_receipt=backend_runtime_bridge_receipt,
            runtime_receipt=backend_runtime_execution_receipt,
            runtime_outcome_receipt=backend_runtime_outcome_receipt,
            robot_asset_contract_receipt=robot_asset_contract_receipt,
            world_state_id=world_state.state_id,
            physics_execution_contract_id=execution_contract.contract_id,
        )
        calibration_receipt = build_physics_calibration_receipt(
            world_state,
            execution_contract,
            adaptation_receipt=adaptation_receipt,
            runtime_evidence=runtime_evidence,
        )
        outcome_receipts = _build_outcome_receipts(
            world_state,
            execution_contract,
            adaptation_receipt,
            backend_binding_receipt,
            robot_asset_contract_receipt,
            backend_runtime_bridge_receipt,
            backend_runtime_work_orders,
            backend_runtime_execution_receipt,
            backend_runtime_adapter_receipt,
            calibration_receipt,
            backend_shadow_execution_receipt=backend_shadow_execution_receipt,
            render_provider_receipts=render_provider_receipts,
            training_feedback_path=training_feedback_path,
        )
        runtime_evidence = summarize_runtime_evidence(
            backend_runtime_execution_receipt=backend_runtime_execution_receipt,
            backend_runtime_launch_receipt=backend_runtime_launch_receipt,
            backend_runtime_outcome_receipt=backend_runtime_outcome_receipt,
            backend_shadow_execution_receipt=backend_shadow_execution_receipt,
            render_provider_receipts=render_provider_receipts,
            outcome_receipts=outcome_receipts,
        )
        adaptation_receipt = build_physics_adaptation_receipt(
            world_state,
            execution_contract,
            runtime_evidence=runtime_evidence,
        )
        backend_binding_receipt = _build_backend_execution_binding_receipt(
            world_state,
            execution_contract,
            adaptation_receipt,
        )
        robot_asset_contract_receipt = _build_robot_asset_contract_receipt(
            world_state,
            execution_contract,
            backend_binding_receipt,
        )
        calibration_receipt = build_physics_calibration_receipt(
            world_state,
            execution_contract,
            adaptation_receipt=adaptation_receipt,
            runtime_evidence=runtime_evidence,
        )
        task_measurement_receipt = build_task_measurement_receipt(world_state)
        sim_real_gap_receipt = build_sim_real_gap_receipt(
            world_state,
            execution_contract,
            calibration_receipt,
        )
        backend_mismatch_receipt = build_backend_mismatch_receipt(
            world_state,
            execution_contract,
            calibration_receipt,
        )
        surrogate_physics_receipt = build_surrogate_physics_receipt(world_state)
        surrogate_calibration_receipt = build_surrogate_calibration_receipt(
            world_state,
            execution_contract,
            calibration_receipt,
            linked_receipts=[
                sim_real_gap_receipt.receipt_id,
                backend_mismatch_receipt.receipt_id,
                surrogate_physics_receipt.receipt_id,
            ],
        )
        branch_validity_receipts = build_branch_validity_receipts(world_state)
        sensor_alignment_receipt = build_sensor_alignment_receipt(world_state)
        replay_validity_receipts = build_replay_validity_receipts(
            world_state,
            task_measurement_receipt=task_measurement_receipt,
            sim_real_gap_receipt=sim_real_gap_receipt,
            sensor_alignment_receipt=sensor_alignment_receipt,
            outcome_receipts=outcome_receipts,
            branch_validity_receipts=branch_validity_receipts,
        )
        training_feedback_manifest = _build_training_feedback_manifest(
            world_state,
            execution_contract,
            adaptation_receipt,
            gen2sim_admission_receipt,
            backend_binding_receipt,
            robot_asset_contract_receipt,
            backend_runtime_bridge_receipt,
            backend_runtime_work_orders,
            backend_runtime_execution_receipt,
            backend_runtime_adapter_receipt,
            backend_runtime_launch_receipt,
            backend_runtime_outcome_receipt,
            backend_shadow_execution_receipt,
            calibration_receipt,
            task_measurement_receipt,
            sim_real_gap_receipt,
            backend_mismatch_receipt,
            surrogate_physics_receipt,
            surrogate_calibration_receipt,
            branch_validity_receipts,
            sensor_alignment_receipt,
            replay_validity_receipts,
            render_provider_receipts,
            outcome_receipts,
        )
        runtime_receipt_manifest = _build_runtime_receipt_manifest(
            world_state=world_state,
            execution_contract=execution_contract,
            artifact_paths=artifact_paths,
            training_feedback_manifest=training_feedback_manifest,
            physics_adaptation_receipt=adaptation_receipt,
            gen2sim_admission_receipt=gen2sim_admission_receipt,
            backend_execution_binding_receipt=backend_binding_receipt,
            robot_asset_contract_receipt=robot_asset_contract_receipt,
            backend_runtime_bridge_receipt=backend_runtime_bridge_receipt,
            backend_runtime_work_orders=backend_runtime_work_orders,
            backend_runtime_execution_receipt=backend_runtime_execution_receipt,
            backend_runtime_adapter_receipt=backend_runtime_adapter_receipt,
            backend_runtime_launch_receipt=backend_runtime_launch_receipt,
            backend_runtime_outcome_receipt=backend_runtime_outcome_receipt,
            backend_shadow_execution_receipt=backend_shadow_execution_receipt,
            physics_calibration_receipt=calibration_receipt,
            task_measurement_receipt=task_measurement_receipt,
            sim_real_gap_receipt=sim_real_gap_receipt,
            backend_mismatch_receipt=backend_mismatch_receipt,
            surrogate_physics_receipt=surrogate_physics_receipt,
            surrogate_calibration_receipt=surrogate_calibration_receipt,
            branch_validity_receipts=branch_validity_receipts,
            sensor_alignment_receipt=sensor_alignment_receipt,
            replay_validity_receipts=replay_validity_receipts,
            render_provider_receipts=render_provider_receipts,
            outcome_receipts=outcome_receipts,
        )
        training_feedback_manifest["runtime_receipt_manifest_id"] = (
            runtime_receipt_manifest["manifest_id"]
        )
        training_feedback_manifest["runtime_receipt_manifest_status"] = (
            runtime_receipt_manifest["manifest_status"]
        )
        training_feedback_manifest["runtime_receipt_missing_required_families"] = list(
            runtime_receipt_manifest.get("missing_required_families") or []
        )
        result = SimSynthPhysicsLoopResult(
            world_state=world_state,
            physics_execution_contract=execution_contract,
            physics_adaptation_receipt=adaptation_receipt,
            gen2sim_admission_receipt=gen2sim_admission_receipt,
            backend_execution_binding_receipt=backend_binding_receipt,
            robot_asset_contract_receipt=robot_asset_contract_receipt,
            backend_runtime_bridge_receipt=backend_runtime_bridge_receipt,
            backend_runtime_work_orders=backend_runtime_work_orders,
            backend_runtime_execution_receipt=backend_runtime_execution_receipt,
            backend_runtime_adapter_receipt=backend_runtime_adapter_receipt,
            backend_runtime_launch_receipt=backend_runtime_launch_receipt,
            backend_runtime_outcome_receipt=backend_runtime_outcome_receipt,
            backend_shadow_execution_receipt=backend_shadow_execution_receipt,
            physics_calibration_receipt=calibration_receipt,
            task_measurement_receipt=task_measurement_receipt,
            sim_real_gap_receipt=sim_real_gap_receipt,
            backend_mismatch_receipt=backend_mismatch_receipt,
            surrogate_physics_receipt=surrogate_physics_receipt,
            surrogate_calibration_receipt=surrogate_calibration_receipt,
            branch_validity_receipts=branch_validity_receipts,
            sensor_alignment_receipt=sensor_alignment_receipt,
            replay_validity_receipts=replay_validity_receipts,
            render_provider_receipts=render_provider_receipts,
            outcome_receipts=outcome_receipts,
            training_feedback_manifest=training_feedback_manifest,
            runtime_receipt_manifest=runtime_receipt_manifest,
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
                artifact_paths["gen2sim_admission_receipt"],
                gen2sim_admission_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["backend_execution_binding_receipt"],
                backend_binding_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["robot_asset_contract_receipt"],
                robot_asset_contract_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["backend_runtime_bridge_receipt"],
                backend_runtime_bridge_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["backend_runtime_work_orders"],
                {
                    "version": "backend_runtime_work_order_bundle_v1",
                    "receipts": [receipt.to_dict() for receipt in backend_runtime_work_orders],
                },
            )
            if backend_runtime_execution_receipt is not None:
                _write_json(
                    artifact_paths["backend_runtime_execution_receipt"],
                    backend_runtime_execution_receipt.to_dict(),
                )
            if backend_runtime_adapter_receipt is not None:
                _write_json(
                    artifact_paths["backend_runtime_adapter_receipt"],
                    backend_runtime_adapter_receipt.to_dict(),
                )
                _write_json(
                    artifact_paths["backend_runtime_adapter_realization"],
                    mapping(backend_runtime_adapter_receipt.metadata.get("realization")),
                )
            upstream_runtime_pack = mapping(
                backend_runtime_bridge_receipt.metadata.get("upstream_runtime_pack")
            ) or mapping(
                {}
                if backend_runtime_execution_receipt is None
                else mapping(backend_runtime_execution_receipt.metadata).get("runtime_bundle", {})
            ).get("upstream_runtime_pack", {})
            if upstream_runtime_pack:
                _write_json(
                    artifact_paths["backend_upstream_runtime_pack"],
                    upstream_runtime_pack,
                )
            runtime_binding = mapping(
                {}
                if backend_runtime_execution_receipt is None
                else mapping(backend_runtime_execution_receipt.metadata).get("runtime_binding")
            ) or mapping(
                {}
                if backend_runtime_execution_receipt is None
                else mapping(backend_runtime_execution_receipt.metadata).get("runtime_bundle", {})
            ).get("runtime_binding", {})
            if runtime_binding:
                _write_json(
                    artifact_paths["backend_runtime_binding"],
                    runtime_binding,
                )
            if backend_runtime_launch_receipt is not None:
                _write_json(
                    artifact_paths["backend_runtime_launch_receipt"],
                    backend_runtime_launch_receipt.to_dict(),
                )
            if backend_runtime_outcome_receipt is not None:
                _write_json(
                    artifact_paths["backend_runtime_outcome_receipt"],
                    backend_runtime_outcome_receipt.to_dict(),
                )
            if backend_shadow_execution_receipt is not None:
                _write_json(
                    artifact_paths["backend_shadow_execution_receipt"],
                    backend_shadow_execution_receipt.to_dict(),
                )
            _write_json(
                artifact_paths["physics_calibration_receipt"],
                calibration_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["task_measurement_receipt"],
                task_measurement_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["sim_real_gap_receipt"],
                sim_real_gap_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["backend_mismatch_receipt"],
                backend_mismatch_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["surrogate_physics_receipt"],
                surrogate_physics_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["surrogate_calibration_receipt"],
                surrogate_calibration_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["branch_validity_receipts"],
                {
                    "version": "branch_validity_receipt_bundle_v1",
                    "receipts": [receipt.to_dict() for receipt in branch_validity_receipts],
                },
            )
            _write_json(
                artifact_paths["sensor_alignment_receipt"],
                sensor_alignment_receipt.to_dict(),
            )
            _write_json(
                artifact_paths["replay_validity_receipts"],
                {
                    "version": "replay_validity_receipt_bundle_v1",
                    "receipts": [receipt.to_dict() for receipt in replay_validity_receipts],
                },
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
                artifact_paths["runtime_receipt_manifest"],
                runtime_receipt_manifest,
            )
            _write_json(
                artifact_paths["loop_summary"],
                {
                    "version": "sim_synth_physics_loop_summary_v1",
                    "world_state_id": world_state.state_id,
                    "physics_execution_contract_id": execution_contract.contract_id,
                    "physics_adaptation_receipt_id": adaptation_receipt.receipt_id,
                    "runtime_receipt_manifest_id": runtime_receipt_manifest["manifest_id"],
                    "runtime_receipt_manifest_status": runtime_receipt_manifest[
                        "manifest_status"
                    ],
                    "gen2sim_admission_receipt_id": gen2sim_admission_receipt.receipt_id,
                    "gen2sim_benchmark_gate_ready": bool(gen2sim_admission_receipt.benchmark_gate_ready),
                    "gen2sim_admissible_branch_count": len(gen2sim_admission_receipt.admissible_branch_ids),
                    "gen2sim_blocked_branch_count": len(gen2sim_admission_receipt.blocked_branch_ids),
                    "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
                    "robot_asset_contract_receipt_id": robot_asset_contract_receipt.receipt_id,
                    "backend_runtime_bridge_receipt_id": backend_runtime_bridge_receipt.receipt_id,
                    "backend_runtime_execution_receipt_id": (
                        None
                        if backend_runtime_execution_receipt is None
                        else backend_runtime_execution_receipt.receipt_id
                    ),
                    "backend_runtime_adapter_receipt_id": (
                        None
                        if backend_runtime_adapter_receipt is None
                        else backend_runtime_adapter_receipt.receipt_id
                    ),
                    "backend_runtime_launch_receipt_id": (
                        None
                        if backend_runtime_launch_receipt is None
                        else backend_runtime_launch_receipt.receipt_id
                    ),
                    "backend_runtime_outcome_receipt_id": (
                        None
                        if backend_runtime_outcome_receipt is None
                        else backend_runtime_outcome_receipt.receipt_id
                    ),
                    "backend_shadow_execution_receipt_id": (
                        None
                        if backend_shadow_execution_receipt is None
                        else backend_shadow_execution_receipt.receipt_id
                    ),
                    "physics_calibration_receipt_id": calibration_receipt.receipt_id,
                    "robot_asset_readiness_score": float(
                        robot_asset_contract_receipt.readiness_score
                    ),
                    "backend_runtime_bridge_status": backend_runtime_bridge_receipt.bridge_status,
                    "bridge_execution_authority": backend_runtime_bridge_receipt.execution_authority,
                    "bridge_transport_profile": backend_runtime_bridge_receipt.transport_profile,
                    "bridge_readiness_score": float(
                        backend_runtime_bridge_receipt.bridge_readiness_score
                    ),
                    "backend_runtime_work_order_count": len(backend_runtime_work_orders),
                    "backend_runtime_work_order_statuses": [
                        receipt.status for receipt in backend_runtime_work_orders
                    ],
                    "render_provider_receipt_count": len(render_provider_receipts),
                    "materialized_render_provider_count": training_feedback_manifest.get(
                        "materialized_render_provider_count",
                        0,
                    ),
                    "requested_backend": execution_contract.requested_backend,
                    "resolved_backend": execution_contract.resolved_backend,
                    "route_status": execution_contract.route_status,
                    "backend_runtime_execution_status": (
                        ""
                        if backend_runtime_execution_receipt is None
                        else backend_runtime_execution_receipt.execution_status
                    ),
                    "backend_runtime_adapter_status": (
                        ""
                        if backend_runtime_adapter_receipt is None
                        else backend_runtime_adapter_receipt.adapter_status
                    ),
                    "backend_runtime_adapter_execution_path": (
                        ""
                        if backend_runtime_adapter_receipt is None
                        else backend_runtime_adapter_receipt.execution_path
                    ),
                    "backend_runtime_adapter_realization_path": str(
                        mapping(
                            {}
                            if backend_runtime_adapter_receipt is None
                            else backend_runtime_adapter_receipt.metadata.get("realization")
                        ).get("realization_path", "")
                        or ""
                    ),
                    "backend_runtime_adapter_realization_status": str(
                        mapping(
                            {}
                            if backend_runtime_adapter_receipt is None
                            else backend_runtime_adapter_receipt.metadata.get("realization")
                        ).get("realization_status", "")
                        or ""
                    ),
                    "backend_upstream_runtime_pack_status": str(
                        mapping(
                            backend_runtime_bridge_receipt.metadata.get(
                                "upstream_runtime_pack"
                            )
                        ).get("pack_status", "")
                        or ""
                    ),
                    "backend_upstream_runtime_ready_surfaces": list(
                        mapping(
                            backend_runtime_bridge_receipt.metadata.get(
                                "upstream_runtime_pack"
                            )
                        ).get("ready_surfaces", [])
                        or []
                    ),
                    "backend_runtime_binding_status": str(
                        mapping(
                            {}
                            if backend_runtime_execution_receipt is None
                            else mapping(backend_runtime_execution_receipt.metadata).get(
                                "runtime_binding"
                            )
                        ).get("binding_status", "")
                        or ""
                    ),
                    "backend_runtime_binding_selected_profile": str(
                        mapping(
                            {}
                            if backend_runtime_execution_receipt is None
                            else mapping(backend_runtime_execution_receipt.metadata).get(
                                "runtime_binding"
                            )
                        ).get("selected_profile", "")
                        or ""
                    ),
                    "backend_runtime_launch_status": (
                        ""
                        if backend_runtime_launch_receipt is None
                        else backend_runtime_launch_receipt.launch_status
                    ),
                    "backend_runtime_outcome_status": (
                        ""
                        if backend_runtime_outcome_receipt is None
                        else backend_runtime_outcome_receipt.outcome_status
                    ),
                    "backend_runtime_output_count": (
                        0
                        if backend_runtime_outcome_receipt is None
                        else backend_runtime_outcome_receipt.harvested_output_count
                    ),
                    "backend_shadow_execution_status": (
                        ""
                        if backend_shadow_execution_receipt is None
                        else backend_shadow_execution_receipt.execution_status
                    ),
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
        execute_external_runtime_launch: bool = False,
        external_launch_cwd: str | Path | None = None,
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
        result = self.execute_world_state(
            world_state,
            output_dir=output_dir,
            execute_external_runtime_launch=execute_external_runtime_launch,
            external_launch_cwd=external_launch_cwd,
        )
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

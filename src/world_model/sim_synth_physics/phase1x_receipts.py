"""Receipt builders for the Phase 1.x transfer and surrogate lanes."""

from __future__ import annotations

from typing import Any, Sequence

from .common import clip01, mapping, stable_id
from .physics_contracts import PhysicsExecutionContract
from .receipts import (
    BackendMismatchReceipt,
    BranchValidityReceipt,
    PhysicsCalibrationReceipt,
    SensorAlignmentReceipt,
    SimRealGapReceipt,
    SurrogateCalibrationReceipt,
    SurrogatePhysicsReceipt,
    TaskMeasurementReceipt,
)
from .state import SimSynthPhysicsWorldState
from .utils.camera_geometry import (
    camera_intrinsics_from_mapping,
    camera_round_trip_error,
    transform_from_mapping,
)


def _runtime_evidence(calibration_receipt: PhysicsCalibrationReceipt) -> dict[str, Any]:
    return mapping(calibration_receipt.metadata.get("runtime_evidence"))


def build_sim_real_gap_receipt(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    calibration_receipt: PhysicsCalibrationReceipt,
) -> SimRealGapReceipt:
    """Estimate transfer gap honestly from currently available local evidence."""

    runtime_evidence = _runtime_evidence(calibration_receipt)
    gap_score = 1.0 - clip01(calibration_receipt.quality_score)
    if execution_contract.route_status == "fallback":
        gap_score += 0.12
    if (
        str(execution_contract.target_hardware_class) == "unitree_g1_r1_class"
        and execution_contract.resolved_backend not in {"isaac", "holosoma"}
    ):
        gap_score += 0.12
    measured = bool(runtime_evidence.get("runtime_concrete_completed", False))
    payload = {
        "state_id": world_state.state_id,
        "resolved_backend": execution_contract.resolved_backend,
        "target_hardware_class": execution_contract.target_hardware_class,
        "gap_score": clip01(gap_score),
    }
    return SimRealGapReceipt(
        receipt_id=stable_id("sim_real_gap_receipt", payload),
        source_backend=execution_contract.resolved_backend,
        target_hardware_class=execution_contract.target_hardware_class,
        comparison_scope="planning_window",
        gap_score=clip01(gap_score),
        realism_confidence=clip01(1.0 - gap_score),
        status="measured" if measured else "estimated",
        branch_plan_ids=[plan.plan_id for plan in world_state.synthetic_branch_plans],
        metadata={
            "world_state_id": world_state.state_id,
            "physics_calibration_receipt_id": calibration_receipt.receipt_id,
            "route_status": execution_contract.route_status,
            "requested_backend": execution_contract.requested_backend,
            "runtime_evidence": runtime_evidence,
        },
    )


def build_task_measurement_receipt(
    world_state: SimSynthPhysicsWorldState,
) -> TaskMeasurementReceipt:
    """Emit the first live receipt from the task / measurement protocol."""

    surface = world_state.task_measurements
    task_contract = world_state.task_definition_contract
    surface_id = "" if surface is None else str(surface.surface_id)
    task_definition_contract_id = "" if task_contract is None else str(task_contract.contract_id)
    task_family = "unknown" if surface is None else str(surface.task_family)
    payload = {
        "state_id": world_state.state_id,
        "surface_id": surface_id,
        "task_definition_contract_id": task_definition_contract_id,
    }
    return TaskMeasurementReceipt(
        receipt_id=stable_id("task_measurement_receipt", payload),
        surface_id=surface_id,
        task_definition_contract_id=task_definition_contract_id,
        task_family=task_family,
        benchmark_gate_ready=False if surface is None else bool(surface.benchmark_gate_ready),
        measurement_values={} if surface is None else dict(surface.measurement_values),
        measurement_status={} if surface is None else dict(surface.measurement_status),
        metadata={
            "world_state_id": world_state.state_id,
            "simulator_backend_contract_id": (
                ""
                if world_state.simulator_backend_contract is None
                else world_state.simulator_backend_contract.contract_id
            ),
            "measurement_dependencies": (
                {} if surface is None else dict(surface.measurement_dependencies)
            ),
        },
    )


def build_backend_mismatch_receipt(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    calibration_receipt: PhysicsCalibrationReceipt,
) -> BackendMismatchReceipt:
    """Estimate mismatch between the requested and realized backend route."""

    mismatch_score = 0.0
    if execution_contract.requested_backend != execution_contract.resolved_backend:
        mismatch_score = 0.22
    if execution_contract.route_status == "fallback":
        mismatch_score += 0.18
    calibration_staleness_score = 1.0 - clip01(calibration_receipt.quality_score)
    payload = {
        "state_id": world_state.state_id,
        "reference_backend": execution_contract.requested_backend,
        "candidate_backend": execution_contract.resolved_backend,
        "route_status": execution_contract.route_status,
    }
    return BackendMismatchReceipt(
        receipt_id=stable_id("backend_mismatch_receipt", payload),
        reference_backend=execution_contract.requested_backend,
        candidate_backend=execution_contract.resolved_backend,
        mismatch_score=clip01(mismatch_score),
        calibration_staleness_score=clip01(calibration_staleness_score),
        status="matched" if mismatch_score == 0.0 else "mismatch_estimated",
        metadata={
            "world_state_id": world_state.state_id,
            "physics_calibration_receipt_id": calibration_receipt.receipt_id,
            "route_status": execution_contract.route_status,
            "fallback_reason": execution_contract.fallback_reason,
        },
    )


def build_surrogate_physics_receipt(
    world_state: SimSynthPhysicsWorldState,
) -> SurrogatePhysicsReceipt:
    """Summarize the advisory surrogate lane without overstating authority."""

    provider = world_state.surrogate_physics_provider
    provider_id = "" if provider is None else str(provider.provider_id)
    provider_status = "missing" if provider is None else str(provider.provider_status)
    available = False if provider is None else bool(provider.available)
    confidence = 0.68 if available else 0.0
    payload = {
        "state_id": world_state.state_id,
        "provider_id": provider_id,
        "provider_status": provider_status,
    }
    return SurrogatePhysicsReceipt(
        receipt_id=stable_id("surrogate_physics_receipt", payload),
        provider_id=provider_id,
        forecast_scope="branch_preview",
        forecast_status="shadow_available" if available else "contract_reserved",
        surrogate_confidence=confidence,
        branch_plan_ids=[plan.plan_id for plan in world_state.synthetic_branch_plans],
        metadata={
            "world_state_id": world_state.state_id,
            "provider_status": provider_status,
            "provider_family": "" if provider is None else str(provider.provider_family),
            "lane_authority": "advisory_provider_only",
        },
    )


def build_surrogate_calibration_receipt(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    calibration_receipt: PhysicsCalibrationReceipt,
    *,
    linked_receipts: Sequence[str] = (),
) -> SurrogateCalibrationReceipt:
    """Emit bounded surrogate-vs-backend calibration posture."""

    provider = world_state.surrogate_physics_provider
    provider_id = "" if provider is None else str(provider.provider_id)
    available = False if provider is None else bool(provider.available)
    calibration_score = (
        clip01(0.35 + 0.55 * clip01(calibration_receipt.quality_score))
        if available
        else 0.0
    )
    payload = {
        "state_id": world_state.state_id,
        "provider_id": provider_id,
        "reference_backend": execution_contract.resolved_backend,
        "available": available,
    }
    return SurrogateCalibrationReceipt(
        receipt_id=stable_id("surrogate_calibration_receipt", payload),
        provider_id=provider_id,
        reference_backend=execution_contract.resolved_backend,
        calibration_status="shadow_calibrated" if available else "not_calibrated",
        calibration_score=calibration_score,
        staleness_score=clip01(1.0 - calibration_score),
        metadata={
            "world_state_id": world_state.state_id,
            "physics_calibration_receipt_id": calibration_receipt.receipt_id,
            "linked_receipts": [str(receipt_id) for receipt_id in linked_receipts if receipt_id],
            "lane_authority": "advisory_provider_only",
        },
    )


def build_sensor_alignment_receipt(
    world_state: SimSynthPhysicsWorldState,
) -> SensorAlignmentReceipt:
    """Emit a CPU-local camera geometry / sensor-alignment receipt."""

    scene_hierarchy = world_state.scene_hierarchy
    scene_metadata = mapping({} if scene_hierarchy is None else scene_hierarchy.metadata)
    semantic_context = mapping(scene_metadata.get("semantic_context"))
    intrinsics_source = (
        semantic_context.get("camera_intrinsics")
        or semantic_context.get("sensor_intrinsics")
        or semantic_context.get("intrinsics")
    )
    intrinsics_payload = (
        mapping(intrinsics_source)
        if isinstance(intrinsics_source, dict)
        else ({"matrix": intrinsics_source} if intrinsics_source is not None else {})
    )
    extrinsics_source = (
        semantic_context.get("camera_extrinsics")
        or semantic_context.get("camera_pose")
        or semantic_context.get("sensor_extrinsics")
    )
    checks = {
        "scene_hierarchy": "present" if scene_hierarchy is not None else "missing",
        "intrinsics": "missing",
        "extrinsics": "missing",
        "round_trip": "not_run",
    }
    metrics: dict[str, float] = {}
    metadata: dict[str, Any] = {
        "world_state_id": world_state.state_id,
        "scene_materialization_status": (
            "" if scene_hierarchy is None else scene_hierarchy.materialization_status
        ),
        "asset_contract_id": world_state.robot_asset_contract.contract_id,
        "asset_profile": world_state.robot_asset_contract.asset_profile,
        "missing_assets": list(world_state.robot_asset_contract.missing_assets),
        "semantic_context_keys": sorted(semantic_context),
    }
    status = "alignment_contract_missing"
    alignment_score = 0.0

    if intrinsics_payload:
        try:
            intrinsics = camera_intrinsics_from_mapping(intrinsics_payload)
            checks["intrinsics"] = "valid"
            metadata["intrinsics_matrix"] = intrinsics.tolist()
            camera_to_world = None
            if extrinsics_source is not None:
                camera_to_world = transform_from_mapping(extrinsics_source)
                checks["extrinsics"] = "valid"
                metadata["camera_to_world"] = camera_to_world.tolist()
            depth_sample = semantic_context.get("alignment_depth_sample") or [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
            round_trip_error = camera_round_trip_error(
                depth_sample,
                intrinsics,
                camera_to_world=camera_to_world,
            )
            checks["round_trip"] = "passed" if round_trip_error <= 1e-6 else "failed"
            metrics["round_trip_max_pixel_error"] = round_trip_error
            alignment_score = clip01(1.0 - min(round_trip_error, 1.0))
            if camera_to_world is None:
                status = "intrinsics_validated_extrinsics_missing"
                alignment_score = min(alignment_score, 0.5)
            elif checks["round_trip"] == "passed":
                status = "geometry_contract_validated"
            else:
                status = "geometry_round_trip_mismatch"
        except Exception as exc:  # pragma: no cover - exact exception type is metadata only
            status = "alignment_contract_invalid"
            checks["round_trip"] = "failed"
            metadata["validation_error"] = str(exc)
            alignment_score = 0.0
            if checks["intrinsics"] == "missing" and intrinsics_payload:
                checks["intrinsics"] = "invalid"
            if extrinsics_source is not None and checks["extrinsics"] == "missing":
                checks["extrinsics"] = "invalid"

    payload = {
        "state_id": world_state.state_id,
        "scene_hierarchy_id": "" if scene_hierarchy is None else scene_hierarchy.hierarchy_id,
        "sensor_profile": "" if scene_hierarchy is None else scene_hierarchy.sensor_profile,
        "status": status,
        "checks": checks,
    }
    return SensorAlignmentReceipt(
        receipt_id=stable_id("sensor_alignment_receipt", payload),
        scene_hierarchy_id="" if scene_hierarchy is None else scene_hierarchy.hierarchy_id,
        sensor_profile="" if scene_hierarchy is None else scene_hierarchy.sensor_profile,
        alignment_score=alignment_score,
        status=status,
        checks=checks,
        metrics=metrics,
        metadata=metadata,
    )



def _branch_reject_reasons(
    *,
    plan_metadata: dict[str, Any],
    admission_preconditions: dict[str, Any],
    benchmark_gate_ready: bool,
    semantic_grounding_non_heuristic: bool,
    admissible: bool,
) -> list[str]:
    reasons: list[str] = []
    if bool(admission_preconditions.get("requires_benchmark_ready", False)) and not benchmark_gate_ready:
        reasons.append("benchmark_gate_not_ready")
    if (
        bool(admission_preconditions.get("requires_non_heuristic_grounding", False))
        and not semantic_grounding_non_heuristic
    ):
        reasons.append("semantic_grounding_heuristic")
    if str(plan_metadata.get("scene_materialization_status", "")) == "asset_contract_incomplete":
        reasons.append("scene_asset_contract_incomplete")
    if not admissible and not reasons:
        reasons.append("admission_gate_blocked")
    return reasons


def build_branch_validity_receipts(
    world_state: SimSynthPhysicsWorldState,
) -> list[BranchValidityReceipt]:
    """Emit per-branch validity / reject-filter receipts from admission state."""

    admission = world_state.gen2sim_admission
    admissible_branch_ids = set(getattr(admission, "admissible_branch_ids", []) or [])
    admission_metadata = mapping({} if admission is None else admission.metadata)
    admission_scores = mapping(admission_metadata.get("admission_scores"))
    benchmark_signals = mapping(admission_metadata.get("benchmark_signals"))
    benchmark_gate_ready = bool(
        benchmark_signals.get("ready", False)
        or benchmark_signals.get("benchmark_eligible", False)
        or getattr(admission, "benchmark_gate_ready", False)
    )
    semantic_grounding_non_heuristic = bool(
        admission_metadata.get("semantic_grounding_non_heuristic", False)
    )
    receipts: list[BranchValidityReceipt] = []
    for plan in world_state.synthetic_branch_plans:
        plan_metadata = mapping(plan.metadata)
        admission_preconditions = mapping(plan.admission_preconditions)
        admissible = plan.plan_id in admissible_branch_ids
        admission_score = clip01(
            admission_scores.get(plan.plan_id, getattr(plan, "expected_yield_score", 0.0))
        )
        reject_reasons = _branch_reject_reasons(
            plan_metadata=plan_metadata,
            admission_preconditions=admission_preconditions,
            benchmark_gate_ready=benchmark_gate_ready,
            semantic_grounding_non_heuristic=semantic_grounding_non_heuristic,
            admissible=admissible,
        )
        payload = {
            "state_id": world_state.state_id,
            "branch_plan_id": plan.plan_id,
            "admission_score": admission_score,
            "admissible": admissible,
        }
        receipts.append(
            BranchValidityReceipt(
                receipt_id=stable_id("branch_validity_receipt", payload),
                branch_plan_id=plan.plan_id,
                job_id=plan.source_job_id,
                validity_score=admission_score,
                admission_score=admission_score,
                admissible=admissible,
                evidence_status=(
                    "benchmark_supported_estimate"
                    if benchmark_gate_ready
                    else "local_estimate"
                ),
                reject_reasons=reject_reasons,
                metadata={
                    "world_state_id": world_state.state_id,
                    "gen2sim_admission_id": (
                        "" if admission is None else str(admission.admission_id)
                    ),
                    "admission_preconditions": admission_preconditions,
                    "scene_hierarchy_ref": mapping(plan_metadata.get("scene_hierarchy_ref")),
                    "scene_materialization_status": str(
                        plan_metadata.get("scene_materialization_status", "") or ""
                    ),
                    "expected_yield_score": float(plan.expected_yield_score),
                    "benchmark_gate_ready": benchmark_gate_ready,
                    "semantic_grounding_non_heuristic": semantic_grounding_non_heuristic,
                },
            )
        )
    return receipts


__all__ = [
    "build_backend_mismatch_receipt",
    "build_branch_validity_receipts",
    "build_sensor_alignment_receipt",
    "build_sim_real_gap_receipt",
    "build_surrogate_calibration_receipt",
    "build_surrogate_physics_receipt",
    "build_task_measurement_receipt",
]

"""WM-owned backend shadow execution/materialization helpers."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Optional

from src.envs.physics.isaac_backend import IsaacBackend
from src.motor_backend.holosoma_backend import HOLOSOMA_TASK_MAP

from .common import mapping
from .physics_contracts import PhysicsExecutionContract
from .receipts import (
    BackendExecutionBindingReceipt,
    BackendRuntimeAdapterReceipt,
    BackendRuntimeExecutionReceipt,
    BackendRuntimeLaunchReceipt,
    BackendRuntimeOutcomeReceipt,
    BackendShadowExecutionReceipt,
)
from .state import SimSynthPhysicsWorldState


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _runtime_ladder_metadata(
    *,
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt],
    backend_runtime_adapter_receipt: Optional[BackendRuntimeAdapterReceipt],
    backend_runtime_launch_receipt: Optional[BackendRuntimeLaunchReceipt],
    backend_runtime_outcome_receipt: Optional[BackendRuntimeOutcomeReceipt],
) -> dict[str, Any]:
    execution_metadata = _mapping(
        {}
        if backend_runtime_execution_receipt is None
        else backend_runtime_execution_receipt.metadata
    )
    runtime_bundle = _mapping(execution_metadata.get("runtime_bundle"))
    runtime_binding = _mapping(
        execution_metadata.get("runtime_binding") or runtime_bundle.get("runtime_binding")
    )
    adapter_metadata = _mapping(
        {}
        if backend_runtime_adapter_receipt is None
        else backend_runtime_adapter_receipt.metadata
    )
    adapter_realization = _mapping(adapter_metadata.get("realization"))
    local_adapter_invocation = _mapping(adapter_metadata.get("local_adapter_invocation"))
    local_adapter_result = _mapping(adapter_metadata.get("local_adapter_result"))
    outcome_metadata = _mapping(
        {}
        if backend_runtime_outcome_receipt is None
        else backend_runtime_outcome_receipt.metadata
    )
    structured_outputs = _mapping(outcome_metadata.get("structured_outputs"))
    return {
        "backend_runtime_execution_receipt_id": (
            ""
            if backend_runtime_execution_receipt is None
            else backend_runtime_execution_receipt.receipt_id
        ),
        "backend_runtime_execution_status": (
            ""
            if backend_runtime_execution_receipt is None
            else backend_runtime_execution_receipt.execution_status
        ),
        "backend_runtime_adapter_receipt_id": (
            ""
            if backend_runtime_adapter_receipt is None
            else backend_runtime_adapter_receipt.receipt_id
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
            adapter_realization.get("realization_path", "") or ""
        ),
        "backend_runtime_adapter_realization_status": str(
            adapter_realization.get("realization_status", "") or ""
        ),
        "backend_runtime_local_adapter_invocation_status": str(
            local_adapter_invocation.get("invocation_status", "") or ""
        ),
        "backend_runtime_local_adapter_result_status": str(
            local_adapter_result.get("result_status", "") or ""
        ),
        "backend_runtime_launch_receipt_id": (
            ""
            if backend_runtime_launch_receipt is None
            else backend_runtime_launch_receipt.receipt_id
        ),
        "backend_runtime_launch_status": (
            ""
            if backend_runtime_launch_receipt is None
            else backend_runtime_launch_receipt.launch_status
        ),
        "backend_runtime_launch_executed": (
            False
            if backend_runtime_launch_receipt is None
            else bool(backend_runtime_launch_receipt.executed)
        ),
        "backend_runtime_outcome_receipt_id": (
            ""
            if backend_runtime_outcome_receipt is None
            else backend_runtime_outcome_receipt.receipt_id
        ),
        "backend_runtime_outcome_status": (
            ""
            if backend_runtime_outcome_receipt is None
            else backend_runtime_outcome_receipt.outcome_status
        ),
        "backend_runtime_output_count": (
            0
            if backend_runtime_outcome_receipt is None
            else int(backend_runtime_outcome_receipt.harvested_output_count)
        ),
        "backend_runtime_ready_surfaces": list(structured_outputs.get("ready_surfaces") or []),
        "backend_runtime_binding": runtime_binding,
        "backend_runtime_binding_status": str(runtime_binding.get("binding_status", "") or ""),
        "backend_runtime_binding_selected_profile": str(
            runtime_binding.get("selected_profile", "") or ""
        ),
        "backend_runtime_binding_selected_policy_ref": str(
            runtime_binding.get("selected_policy_ref", "") or ""
        ),
        "backend_runtime_binding_selected_launch_root": str(
            runtime_binding.get("selected_launch_root", "") or ""
        ),
        "backend_runtime_binding_missing_components": list(
            runtime_binding.get("missing_components") or []
        ),
        "runtime_ladder_threaded": any(
            [
                backend_runtime_execution_receipt is not None,
                backend_runtime_adapter_receipt is not None,
                backend_runtime_launch_receipt is not None,
                backend_runtime_outcome_receipt is not None,
            ]
        ),
    }


def _materialize_robot_asset_sidecars(
    world_state: SimSynthPhysicsWorldState,
    *,
    output_root: Optional[Path],
    backend: str,
) -> tuple[list[str], dict[str, Any]]:
    contract = world_state.robot_asset_contract
    if contract is None:
        return [], {}
    artifact_refs: list[str] = []
    contract_path = None if output_root is None else output_root / "robot_asset_contract_sidecar.json"
    calibration_path = None if output_root is None else output_root / "backend_calibration_sidecar.json"
    io_path = None if output_root is None else output_root / "backend_io_contract_sidecar.json"
    summary = {
        "robot_asset_contract_id": contract.contract_id,
        "asset_profile": contract.asset_profile,
        "target_hardware_class": contract.target_hardware_class,
        "required_assets": list(contract.required_assets),
        "available_assets": list(contract.available_assets),
        "missing_assets": list(contract.missing_assets),
        "calibration_contracts": list(contract.calibration_contracts),
        "observation_contracts": list(contract.observation_contracts),
        "action_contracts": list(contract.action_contracts),
        "asset_readiness_score": float(contract.metadata.get("asset_readiness_score", 0.0) or 0.0),
        "normalized_asset_manifest": mapping(contract.metadata.get("normalized_asset_manifest", {})),
        "recommended_assets": list(contract.metadata.get("recommended_assets", []) or []),
    }
    if contract_path is not None:
        _write_json(
            contract_path,
            {
                "version": "robot_asset_contract_sidecar_v1",
                "backend": backend,
                "world_state_id": world_state.state_id,
                "robot_asset_contract": contract.to_dict(),
            },
        )
        artifact_refs.append(str(contract_path.resolve()))
    if calibration_path is not None:
        _write_json(
            calibration_path,
            {
                "version": "backend_calibration_sidecar_v1",
                "backend": backend,
                "world_state_id": world_state.state_id,
                "target_hardware_class": contract.target_hardware_class,
                "calibration_contracts": list(contract.calibration_contracts),
                "missing_assets": list(contract.missing_assets),
            },
        )
        artifact_refs.append(str(calibration_path.resolve()))
    if io_path is not None:
        _write_json(
            io_path,
            {
                "version": "backend_io_contract_sidecar_v1",
                "backend": backend,
                "world_state_id": world_state.state_id,
                "observation_contracts": list(contract.observation_contracts),
                "action_contracts": list(contract.action_contracts),
                "missing_assets": list(contract.missing_assets),
            },
        )
        artifact_refs.append(str(io_path.resolve()))
    return artifact_refs, summary


def _derive_isaac_shadow_env_config(
    world_state: SimSynthPhysicsWorldState,
    *,
    output_dir: Optional[str | Path],
) -> dict[str, Any]:
    embodiment_context = _mapping(world_state.input_context.get("embodiment"))
    control_constraints = _mapping(embodiment_context.get("control_constraints"))
    robot_asset_manifest = _mapping(
        embodiment_context.get("robot_asset_manifest")
        or embodiment_context.get("asset_manifest")
        or embodiment_context.get("robot_assets")
    )
    first_job = (
        world_state.simulation_agenda.jobs[0]
        if world_state.simulation_agenda.jobs
        else None
    )
    env_name = (
        _mapping(world_state.input_context.get("semantic")).get("env_name")
        or world_state.simulation_agenda.metadata.get("env_name")
        or "sim_synth_shadow"
    )
    action_dim = _int(
        embodiment_context.get("action_dim", control_constraints.get("action_dim", 12)),
        12,
    )
    task_name = getattr(first_job, "objective", None) or env_name
    return {
        "env_name": str(env_name),
        "task": str(task_name),
        "robot": str(
            embodiment_context.get("primary_embodiment")
            or embodiment_context.get("robot_family")
            or "unitree_shadow"
        ),
        "backend_mode": "shadow_contract",
        "max_steps": max(2, min(4, len(world_state.synthetic_branch_plans) + 1)),
        "dt": float(control_constraints.get("dt", 1.0 / 60.0) or 1.0 / 60.0),
        "action_dim": action_dim,
        "obs_dim": max(action_dim * 4, 32),
        "seed": 0,
        "econ_params": _mapping(world_state.input_context.get("economic")),
        "robot_asset_manifest": robot_asset_manifest,
        "output_root": (
            None
            if output_dir is None
            else str(Path(output_dir) / "backend_shadow_execution" / "isaac")
        ),
    }


def _infer_holosoma_task_id(world_state: SimSynthPhysicsWorldState) -> str:
    embodiment_context = _mapping(world_state.input_context.get("embodiment"))
    for key in ("holosoma_task_id", "task_id", "task_preset"):
        candidate = str(embodiment_context.get(key, "") or "")
        if candidate in HOLOSOMA_TASK_MAP:
            return candidate
    active_embodiments = [
        str(value).lower()
        for value in (
            embodiment_context.get("active_embodiments")
            or embodiment_context.get("target_embodiments")
            or embodiment_context.get("robot_families")
            or []
        )
    ]
    if any("g1" in value for value in active_embodiments):
        fidelity = str(world_state.physics_context.fidelity_tier or "").lower()
        return "humanoid_wbt_g1" if fidelity == "high_fidelity" else "humanoid_locomotion_g1"
    return "humanoid_locomotion_g1"


def _derive_holosoma_shadow_work_order(
    world_state: SimSynthPhysicsWorldState,
    *,
    output_dir: Optional[Path],
) -> dict[str, Any]:
    embodiment_context = _mapping(world_state.input_context.get("embodiment"))
    control_constraints = _mapping(embodiment_context.get("control_constraints"))
    task_id = _infer_holosoma_task_id(world_state)
    task_spec = HOLOSOMA_TASK_MAP[task_id]
    job = world_state.simulation_agenda.jobs[0] if world_state.simulation_agenda.jobs else None
    work_order_root = (
        None
        if output_dir is None
        else str(Path(output_dir) / "backend_shadow_execution" / "holosoma")
    )
    return {
        "backend": "holosoma",
        "task_id": task_id,
        "exp_name": task_spec.exp_name,
        "simulator": task_spec.simulator,
        "task_name": task_spec.task_name,
        "description": task_spec.description,
        "fidelity_tier": world_state.physics_context.fidelity_tier,
        "target_hardware_class": (
            ""
            if world_state.physics_adaptation_policy is None
            else world_state.physics_adaptation_policy.target_hardware_class
        ),
        "robot_asset_profile": (
            ""
            if world_state.physics_adaptation_policy is None
            else world_state.physics_adaptation_policy.robot_asset_profile
        ),
        "domain_randomization_profile": (
            ""
            if world_state.physics_adaptation_policy is None
            else world_state.physics_adaptation_policy.domain_randomization_profile
        ),
        "system_identification_profile": (
            ""
            if world_state.physics_adaptation_policy is None
            else world_state.physics_adaptation_policy.system_identification_profile
        ),
        "preferred_num_envs": max(1, min(8, len(world_state.synthetic_branch_plans) or 1)),
        "preferred_max_steps": max(64, 32 * max(1, len(world_state.synthetic_branch_plans))),
        "control_dt": float(control_constraints.get("dt", 1.0 / 120.0) or 1.0 / 120.0),
        "active_embodiments": list(
            embodiment_context.get("active_embodiments")
            or embodiment_context.get("target_embodiments")
            or []
        ),
        "source_job_id": "" if job is None else str(job.job_id),
        "source_objective": "" if job is None else str(job.objective_preset),
        "work_order_root": work_order_root,
    }


def _materialize_isaac_shadow_execution(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    backend_binding_receipt: BackendExecutionBindingReceipt,
    *,
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt],
    backend_runtime_adapter_receipt: Optional[BackendRuntimeAdapterReceipt],
    backend_runtime_launch_receipt: Optional[BackendRuntimeLaunchReceipt],
    backend_runtime_outcome_receipt: Optional[BackendRuntimeOutcomeReceipt],
    output_root: Optional[Path],
) -> BackendShadowExecutionReceipt:
    asset_sidecar_refs, asset_summary = _materialize_robot_asset_sidecars(
        world_state,
        output_root=output_root,
        backend="isaac",
    )
    env_config = _derive_isaac_shadow_env_config(world_state, output_dir=output_root)
    env_config["robot_asset_contract_id"] = asset_summary.get("robot_asset_contract_id", "")
    env_config["calibration_contracts"] = list(asset_summary.get("calibration_contracts", []))
    env_config["observation_contracts"] = list(asset_summary.get("observation_contracts", []))
    env_config["action_contracts"] = list(asset_summary.get("action_contracts", []))
    env_config["missing_assets"] = list(asset_summary.get("missing_assets", []))
    backend = IsaacBackend(
        env_config=env_config,
        num_envs=max(1, min(2, len(world_state.synthetic_branch_plans) or 1)),
        device="cuda:0",
    )
    artifact_refs: list[str] = list(asset_sidecar_refs)
    episode_ids: list[str] = []
    execution_status = "shadow_executed"
    summary_path = (
        None if output_root is None else output_root / "backend_shadow_execution_receipt.json"
    )
    batch_summary_path = (
        None if output_root is None else output_root / "backend_shadow_batch_summary.json"
    )
    runtime_ladder_metadata = _runtime_ladder_metadata(
        backend_runtime_execution_receipt=backend_runtime_execution_receipt,
        backend_runtime_adapter_receipt=backend_runtime_adapter_receipt,
        backend_runtime_launch_receipt=backend_runtime_launch_receipt,
        backend_runtime_outcome_receipt=backend_runtime_outcome_receipt,
    )

    try:
        backend.reset()
        zero_action = [0.0 for _ in range(_int(env_config.get("action_dim", 12), 12))]
        for _ in range(max(1, _int(env_config.get("max_steps", 2), 2) - 1)):
            backend.step(zero_action)
        episode_ids = [episode_id for episode_id in [backend.get_current_episode_id(0)] if episode_id]
        if backend.num_envs > 1:
            for env_idx in range(1, backend.num_envs):
                backend.reset_env(env_idx)
                episode_id = backend.get_current_episode_id(env_idx)
                if episode_id:
                    episode_ids.append(episode_id)
        media_ref_sets = [backend.get_media_refs(env_idx) for env_idx in range(backend.num_envs)]
        for media_refs in media_ref_sets:
            artifact_refs.extend(str(value) for value in media_refs.values() if value)
        batch_summaries = [asdict(summary) for summary in backend.get_batch_episode_info()]
        if batch_summary_path is not None:
            _write_json(
                batch_summary_path,
                {
                    "version": "backend_shadow_batch_summary_v1",
                    "backend": "isaac",
                    "execution_mode": "shadow_contract",
                    "batch_summaries": batch_summaries,
                    "binding_status": backend_binding_receipt.binding_status,
                },
            )
            artifact_refs.append(str(batch_summary_path.resolve()))
        if backend_binding_receipt.metadata.get("missing_assets"):
            execution_status = "shadow_executed_with_asset_gaps"
    except Exception as exc:
        execution_status = "shadow_failed"
        artifact_refs = []
        episode_ids = []
        failure = {"error": repr(exc)}
        if batch_summary_path is not None:
            _write_json(
                batch_summary_path,
                {
                    "version": "backend_shadow_batch_summary_v1",
                    "backend": "isaac",
                    "execution_mode": "shadow_contract",
                    "binding_status": backend_binding_receipt.binding_status,
                    "failure": failure,
                },
            )
            artifact_refs.append(str(batch_summary_path.resolve()))
    finally:
        backend.close()
    shadow_harvest_mode = "shadow_with_data_harvest" if episode_ids else "shadow_only_preview"

    receipt = BackendShadowExecutionReceipt(
        receipt_id=f"backend_shadow_execution_receipt_{world_state.state_id}",
        backend="isaac",
        execution_mode="shadow_contract",
        execution_status=execution_status,
        episode_ids=episode_ids,
        artifact_refs=artifact_refs,
        metadata={
            "world_state_id": world_state.state_id,
            "physics_execution_contract_id": execution_contract.contract_id,
            "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
            "binding_status": backend_binding_receipt.binding_status,
            "requested_backend": execution_contract.requested_backend,
            "resolved_backend": execution_contract.resolved_backend,
            "route_status": execution_contract.route_status,
            "required_assets": list(
                backend_binding_receipt.metadata.get("required_assets", []) or []
            ),
            "available_assets": list(
                backend_binding_receipt.metadata.get("available_assets", []) or []
            ),
            "missing_assets": list(
                backend_binding_receipt.metadata.get("missing_assets", []) or []
            ),
            "robot_asset_contract_id": asset_summary.get("robot_asset_contract_id", ""),
            "asset_sidecar_refs": list(asset_sidecar_refs),
            "calibration_contracts": list(asset_summary.get("calibration_contracts", [])),
            "observation_contracts": list(asset_summary.get("observation_contracts", [])),
            "action_contracts": list(asset_summary.get("action_contracts", [])),
            "asset_readiness_score": float(asset_summary.get("asset_readiness_score", 0.0) or 0.0),
            "shadow_harvest_mode": shadow_harvest_mode,
            "env_config": mapping(env_config),
            **runtime_ladder_metadata,
        },
    )
    if summary_path is not None:
        _write_json(summary_path, receipt.to_dict())
    return receipt


def _materialize_holosoma_shadow_execution(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    backend_binding_receipt: BackendExecutionBindingReceipt,
    *,
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt],
    backend_runtime_adapter_receipt: Optional[BackendRuntimeAdapterReceipt],
    backend_runtime_launch_receipt: Optional[BackendRuntimeLaunchReceipt],
    backend_runtime_outcome_receipt: Optional[BackendRuntimeOutcomeReceipt],
    output_root: Optional[Path],
) -> BackendShadowExecutionReceipt:
    asset_sidecar_refs, asset_summary = _materialize_robot_asset_sidecars(
        world_state,
        output_root=output_root,
        backend="holosoma",
    )
    work_order = _derive_holosoma_shadow_work_order(world_state, output_dir=output_root)
    work_order["robot_asset_contract_id"] = asset_summary.get("robot_asset_contract_id", "")
    work_order["calibration_contracts"] = list(asset_summary.get("calibration_contracts", []))
    work_order["observation_contracts"] = list(asset_summary.get("observation_contracts", []))
    work_order["action_contracts"] = list(asset_summary.get("action_contracts", []))
    artifact_refs: list[str] = list(asset_sidecar_refs)
    summary_path = (
        None if output_root is None else output_root / "backend_shadow_execution_receipt.json"
    )
    work_order_path = None if output_root is None else output_root / "holosoma_shadow_work_order.json"
    runtime_available = bool(_has_module("holosoma"))
    missing_assets = list(backend_binding_receipt.metadata.get("missing_assets", []) or [])
    unsatisfied_preconditions = list(missing_assets)
    if not runtime_available:
        unsatisfied_preconditions.append("holosoma_runtime")
    if work_order_path is not None:
        _write_json(
            work_order_path,
            {
                "version": "holosoma_shadow_work_order_v1",
                "world_state_id": world_state.state_id,
                "physics_execution_contract_id": execution_contract.contract_id,
                "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
                "binding_status": backend_binding_receipt.binding_status,
                "work_order": work_order,
                "unsatisfied_preconditions": unsatisfied_preconditions,
            },
        )
        artifact_refs.append(str(work_order_path.resolve()))
    runtime_ladder_metadata = _runtime_ladder_metadata(
        backend_runtime_execution_receipt=backend_runtime_execution_receipt,
        backend_runtime_adapter_receipt=backend_runtime_adapter_receipt,
        backend_runtime_launch_receipt=backend_runtime_launch_receipt,
        backend_runtime_outcome_receipt=backend_runtime_outcome_receipt,
    )

    execution_status = (
        "shadow_work_order_materialized"
        if not unsatisfied_preconditions
        else "shadow_work_order_materialized_with_preconditions"
    )
    receipt = BackendShadowExecutionReceipt(
        receipt_id=f"backend_shadow_execution_receipt_{world_state.state_id}",
        backend="holosoma",
        execution_mode="shadow_work_order",
        execution_status=execution_status,
        episode_ids=[],
        artifact_refs=artifact_refs,
        metadata={
            "world_state_id": world_state.state_id,
            "physics_execution_contract_id": execution_contract.contract_id,
            "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
            "binding_status": backend_binding_receipt.binding_status,
            "requested_backend": execution_contract.requested_backend,
            "resolved_backend": execution_contract.resolved_backend,
            "route_status": execution_contract.route_status,
            "required_assets": list(
                backend_binding_receipt.metadata.get("required_assets", []) or []
            ),
            "available_assets": list(
                backend_binding_receipt.metadata.get("available_assets", []) or []
            ),
            "missing_assets": missing_assets,
            "concrete_runtime_available": runtime_available,
            "unsatisfied_preconditions": unsatisfied_preconditions,
            "robot_asset_contract_id": asset_summary.get("robot_asset_contract_id", ""),
            "asset_sidecar_refs": list(asset_sidecar_refs),
            "calibration_contracts": list(asset_summary.get("calibration_contracts", [])),
            "observation_contracts": list(asset_summary.get("observation_contracts", [])),
            "action_contracts": list(asset_summary.get("action_contracts", [])),
            "asset_readiness_score": float(asset_summary.get("asset_readiness_score", 0.0) or 0.0),
            "shadow_harvest_mode": "shadow_only_preview",
            "work_order": mapping(work_order),
            **runtime_ladder_metadata,
        },
    )
    if summary_path is not None:
        _write_json(summary_path, receipt.to_dict())
    return receipt


def materialize_backend_shadow_execution(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    backend_binding_receipt: BackendExecutionBindingReceipt,
    *,
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt] = None,
    backend_runtime_adapter_receipt: Optional[BackendRuntimeAdapterReceipt] = None,
    backend_runtime_launch_receipt: Optional[BackendRuntimeLaunchReceipt] = None,
    backend_runtime_outcome_receipt: Optional[BackendRuntimeOutcomeReceipt] = None,
    output_dir: str | Path | None = None,
) -> BackendShadowExecutionReceipt | None:
    """Run an explicit WM-owned shadow execution where a concrete backend is absent."""

    binding = world_state.backend_execution_binding
    if binding is None:
        return None
    output_root = None if output_dir is None else Path(output_dir) / "backend_shadow_execution" / str(binding.backend)
    if str(binding.backend) == "isaac":
        return _materialize_isaac_shadow_execution(
            world_state,
            execution_contract,
            backend_binding_receipt,
            backend_runtime_execution_receipt=backend_runtime_execution_receipt,
            backend_runtime_adapter_receipt=backend_runtime_adapter_receipt,
            backend_runtime_launch_receipt=backend_runtime_launch_receipt,
            backend_runtime_outcome_receipt=backend_runtime_outcome_receipt,
            output_root=output_root,
        )
    if str(binding.backend) == "holosoma":
        return _materialize_holosoma_shadow_execution(
            world_state,
            execution_contract,
            backend_binding_receipt,
            backend_runtime_execution_receipt=backend_runtime_execution_receipt,
            backend_runtime_adapter_receipt=backend_runtime_adapter_receipt,
            backend_runtime_launch_receipt=backend_runtime_launch_receipt,
            backend_runtime_outcome_receipt=backend_runtime_outcome_receipt,
            output_root=output_root,
        )
    return None


__all__ = ["materialize_backend_shadow_execution"]

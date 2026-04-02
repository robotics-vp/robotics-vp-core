"""WM-owned concrete backend runtime execution for sim/synth/physics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from src.economics.econ_meter import EconomicMeter
from src.motor_backend.datapacks import DatapackConfig, MotionClipSpec, load_datapack_configs
from src.motor_backend.factory import make_motor_backend
from src.objectives.economic_objective import EconomicObjectiveSpec
from src.ontology.models import Robot, Task
from src.ontology.store import OntologyStore

from .adapters.holosoma_adapter_execution import (
    build_holosoma_adapter_receipt,
    finalize_holosoma_adapter_execution,
    prepare_holosoma_adapter_execution,
)
from .adapters.holosoma_adapter_realization import (
    build_holosoma_adapter_realization,
)
from .adapters.holosoma_executable_consumer import (
    build_holosoma_executable_adapter_consumer,
)
from .adapters.holosoma_executable_adapter import (
    build_holosoma_executable_adapter_request,
)
from .adapters.holosoma_runtime_binding import build_holosoma_runtime_binding
from .adapters.isaac_unitree_adapter_execution import (
    build_isaac_unitree_adapter_receipt,
    finalize_isaac_unitree_adapter_execution,
    prepare_isaac_unitree_adapter_execution,
)
from .adapters.isaac_unitree_adapter_realization import (
    build_isaac_unitree_adapter_realization,
)
from .adapters.local_backend_factory_adapter import (
    build_local_backend_factory_invocation,
    materialize_local_backend_factory_invocation,
)
from .asset_manifest import extract_robot_asset_manifest, normalize_robot_asset_manifest
from .common import mapping, safe_float, strings
from .physics_contracts import PhysicsExecutionContract
from .receipts import (
    BackendExecutionBindingReceipt,
    BackendRuntimeExecutionReceipt,
)
from .runtime_bundles import build_backend_runtime_bundle
from .runtime_launch import (
    build_backend_runtime_launch_receipt,
    execute_backend_runtime_launch,
    prepare_backend_runtime_launch,
)
from .runtime_outcomes import (
    build_backend_runtime_outcome_receipt,
    build_backend_runtime_output_contract,
    harvest_backend_runtime_outcomes,
)
from .runtime_targets import describe_holosoma_runtime_targets, describe_isaac_runtime_targets
from .state import SimSynthPhysicsWorldState


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item not in (None, "")]
    return []


def _runtime_output_root(output_dir: str | Path | None, backend: str) -> Optional[Path]:
    if output_dir is None:
        return None
    return Path(output_dir) / "backend_runtime_execution" / backend


def _source_backend(execution_contract: PhysicsExecutionContract) -> str:
    requested = str(execution_contract.requested_backend or "")
    if requested in {"isaac", "holosoma"}:
        return requested
    resolved = str(execution_contract.resolved_backend or "")
    if resolved in {"isaac", "holosoma"}:
        return resolved
    return resolved


def _source_policy_id(backend: str, world_state: SimSynthPhysicsWorldState) -> str:
    semantic_context = _mapping(world_state.input_context.get("semantic"))
    embodiment_context = _mapping(world_state.input_context.get("embodiment"))
    economic_context = _mapping(world_state.input_context.get("economic"))
    for key in (
        f"{backend}_policy_id",
        "runtime_policy_id",
        "evaluation_policy_id",
        "policy_id",
    ):
        for payload in (embodiment_context, semantic_context, economic_context):
            candidate = str(payload.get(key, "") or "")
            if candidate:
                return candidate
    return ""


def _objective_spec(world_state: SimSynthPhysicsWorldState) -> EconomicObjectiveSpec:
    economic_context = _mapping(world_state.input_context.get("economic"))
    return EconomicObjectiveSpec(
        mpl_weight=safe_float(
            economic_context.get("mpl_weight", economic_context.get("mpl", 1.0)),
            1.0,
        ),
        energy_weight=safe_float(
            economic_context.get("energy_weight", economic_context.get("energy", 0.0)),
            0.0,
        ),
        error_weight=safe_float(
            economic_context.get("error_weight", economic_context.get("error", 0.0)),
            0.0,
        ),
        novelty_weight=safe_float(
            economic_context.get("novelty_weight", economic_context.get("novelty", 0.0)),
            0.0,
        ),
        risk_weight=safe_float(
            economic_context.get("risk_weight", economic_context.get("risk", 0.0)),
            0.0,
        ),
        extra_weights=_mapping(economic_context.get("extra_weights")),
    )


def _build_store_and_meter(
    *,
    world_state: SimSynthPhysicsWorldState,
    backend: str,
    output_root: Optional[Path],
    task_id: str,
) -> tuple[OntologyStore, EconomicMeter]:
    semantic_context = _mapping(world_state.input_context.get("semantic"))
    economic_context = _mapping(world_state.input_context.get("economic"))
    embodiment_context = _mapping(world_state.input_context.get("embodiment"))
    ontology_root = (
        Path("artifacts") / "sim_synth_runtime" / backend / "ontology"
        if output_root is None
        else output_root / "ontology"
    )
    store = OntologyStore(root_dir=str(ontology_root))
    env_name = str(
        semantic_context.get("env_name")
        or world_state.simulation_agenda.metadata.get("env_name")
        or task_id
    )
    task = Task(
        task_id=task_id,
        name=f"SimSynthPhysicsRuntime:{task_id}",
        description=f"WM-owned backend runtime request for {backend}",
        environment_id=env_name,
        human_mpl_units_per_hour=safe_float(
            economic_context.get("human_mpl_units_per_hour", 60.0),
            60.0,
        ),
        human_wage_per_hour=safe_float(
            economic_context.get("human_wage_per_hour", 18.0),
            18.0,
        ),
        default_energy_cost_per_wh=safe_float(
            economic_context.get("default_energy_cost_per_wh", 0.12),
            0.12,
        ),
        metadata={
            "world_state_id": world_state.state_id,
            "backend": backend,
            "target_hardware_class": (
                ""
                if world_state.robot_asset_contract is None
                else world_state.robot_asset_contract.target_hardware_class
            ),
        },
    )
    robot = Robot(
        robot_id=f"{backend}_runtime_robot",
        name=str(
            embodiment_context.get("primary_embodiment")
            or embodiment_context.get("robot_family")
            or f"{backend}_runtime_robot"
        ),
        energy_cost_per_wh=safe_float(
            economic_context.get("energy_cost_per_wh", task.default_energy_cost_per_wh),
            task.default_energy_cost_per_wh,
        ),
        metadata={
            "active_embodiments": _strings(embodiment_context.get("active_embodiments")),
            "world_state_id": world_state.state_id,
        },
    )
    store.upsert_task(task)
    store.upsert_robot(robot)
    return store, EconomicMeter(task=task, robot=robot, config=economic_context)


def _infer_holosoma_task_id(world_state: SimSynthPhysicsWorldState) -> str:
    embodiment_context = _mapping(world_state.input_context.get("embodiment"))
    for key in ("holosoma_task_id", "task_id", "task_preset"):
        candidate = str(embodiment_context.get(key, "") or "")
        if candidate:
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
        return (
            "humanoid_wbt_g1"
            if str(world_state.physics_context.fidelity_tier or "").lower() == "high_fidelity"
            else "humanoid_locomotion_g1"
        )
    return "humanoid_locomotion_g1"


def _infer_isaac_task_id(world_state: SimSynthPhysicsWorldState) -> str:
    semantic_context = _mapping(world_state.input_context.get("semantic"))
    first_job = world_state.simulation_agenda.jobs[0] if world_state.simulation_agenda.jobs else None
    for key in ("isaac_task_id", "task_id", "env_name"):
        candidate = str(semantic_context.get(key, "") or "")
        if candidate:
            return candidate
    if first_job is not None:
        for candidate in (
            getattr(first_job, "objective_preset", ""),
            getattr(first_job, "task_family", ""),
            getattr(first_job, "environment_id", ""),
        ):
            candidate_str = str(candidate or "")
            if candidate_str and candidate_str != "unknown":
                return candidate_str
    return "workcell_shadow_task"


def _rollout_artifact_refs(rollout_bundle: Any) -> list[str]:
    refs: list[str] = []
    scenario_id = str(getattr(rollout_bundle, "scenario_id", "") or "")
    if scenario_id:
        refs.append(scenario_id)
    for episode in list(getattr(rollout_bundle, "episodes", []) or []):
        trajectory_path = getattr(episode, "trajectory_path", None)
        if trajectory_path is not None:
            refs.append(str(Path(trajectory_path).resolve()))
        rgb_path = getattr(episode, "rgb_video_path", None)
        if rgb_path is not None:
            refs.append(str(Path(rgb_path).resolve()))
        depth_path = getattr(episode, "depth_video_path", None)
        if depth_path is not None:
            refs.append(str(Path(depth_path).resolve()))
    return refs


def _runtime_request_payload(
    *,
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    backend_binding_receipt: BackendExecutionBindingReceipt,
    backend: str,
    policy_id: str,
    task_id: str,
) -> dict[str, Any]:
    embodiment_context = _mapping(world_state.input_context.get("embodiment"))
    contract = world_state.robot_asset_contract
    return {
        "version": "backend_runtime_request_v1",
        "world_state_id": world_state.state_id,
        "physics_execution_contract_id": execution_contract.contract_id,
        "backend_execution_binding_receipt_id": backend_binding_receipt.receipt_id,
        "backend": backend,
        "requested_backend": execution_contract.requested_backend,
        "resolved_backend": execution_contract.resolved_backend,
        "route_status": execution_contract.route_status,
        "policy_id": policy_id,
        "task_id": task_id,
        "fidelity_tier": execution_contract.fidelity_tier,
        "domain_randomization_regime": execution_contract.domain_randomization_regime,
        "calibration_profile": execution_contract.calibration_profile,
        "target_hardware_class": execution_contract.target_hardware_class,
        "active_embodiments": list(embodiment_context.get("active_embodiments") or []),
        "robot_asset_contract": (
            None if contract is None else contract.to_dict()
        ),
    }


def _materialize_holosoma_binding(
    world_state: SimSynthPhysicsWorldState,
    *,
    output_root: Optional[Path],
    task_id: str,
) -> tuple[list[str], dict[str, Any]]:
    embodiment_context = _mapping(world_state.input_context.get("embodiment"))
    datapack_entries = list(embodiment_context.get("motion_clip_datapacks") or [])
    motion_clips = list(embodiment_context.get("motion_clips") or [])
    motion_clip_paths = _strings(embodiment_context.get("motion_clip_paths"))
    retargeting_contract = _mapping(
        embodiment_context.get("retargeting_contract")
        or embodiment_context.get("whole_body_retargeting")
    )
    runtime_target_contract = describe_holosoma_runtime_targets(embodiment_context)
    payload = {
        "version": "holosoma_datapack_binding_v1",
        "task_id": task_id,
        "active_embodiments": list(embodiment_context.get("active_embodiments") or []),
        "motion_clip_datapacks": datapack_entries,
        "motion_clips": motion_clips,
        "motion_clip_paths": motion_clip_paths,
        "whole_body_reward_overlay": _mapping(
            embodiment_context.get("whole_body_reward_overlay")
        ),
        "retargeting_contract": retargeting_contract,
        "motion_source_contract_present": bool(datapack_entries or motion_clips or motion_clip_paths),
        "retargeting_contract_present": bool(retargeting_contract),
        "runtime_target_contract": runtime_target_contract,
    }
    refs: list[str] = []
    if output_root is not None:
        binding_path = output_root / "holosoma_datapack_binding.json"
        _write_json(binding_path, payload)
        refs.append(str(binding_path.resolve()))
    return refs, payload


def _coerce_motion_clips(raw: Any) -> list[MotionClipSpec]:
    clips: list[MotionClipSpec] = []
    if not isinstance(raw, (list, tuple)):
        return clips
    for entry in raw:
        if isinstance(entry, str):
            clips.append(MotionClipSpec(path=entry, weight=1.0))
            continue
        if isinstance(entry, Mapping):
            path = entry.get("path")
            if not path:
                continue
            try:
                weight = float(entry.get("weight", 1.0))
            except Exception:
                weight = 1.0
            clips.append(MotionClipSpec(path=str(path), weight=weight))
    return clips


def _resolve_holosoma_datapacks(
    world_state: SimSynthPhysicsWorldState,
    *,
    task_id: str,
) -> tuple[list[str], list[DatapackConfig]]:
    embodiment_context = _mapping(world_state.input_context.get("embodiment"))
    datapack_entries = list(embodiment_context.get("motion_clip_datapacks") or [])
    datapack_ids: list[str] = []
    datapack_config_paths: list[str] = []
    for entry in datapack_entries:
        text = str(entry or "").strip()
        if not text:
            continue
        if Path(text).exists():
            datapack_config_paths.append(text)
        else:
            datapack_ids.append(text)

    datapack_configs = load_datapack_configs(datapack_config_paths) if datapack_config_paths else []
    direct_motion_clips = _coerce_motion_clips(
        embodiment_context.get("motion_clips") or embodiment_context.get("motion_clip_paths")
    )
    if direct_motion_clips:
        datapack_configs.append(
            DatapackConfig(
                id=f"{task_id}_runtime_motionpack",
                description="WM-owned Holosoma runtime datapack",
                motion_clips=direct_motion_clips,
                tags=["humanoid", "wm_runtime"],
                task_tags=[task_id],
                robot_families=list(embodiment_context.get("active_embodiments") or []),
                metadata={"source": "sim_synth_physics_backend_runtime_execution"},
            )
        )
    return datapack_ids, datapack_configs


def _materialize_isaac_binding(
    world_state: SimSynthPhysicsWorldState,
    *,
    output_root: Optional[Path],
    task_id: str,
) -> tuple[list[str], dict[str, Any]]:
    embodiment_context = _mapping(world_state.input_context.get("embodiment"))
    semantic_context = _mapping(world_state.input_context.get("semantic"))
    contract = world_state.robot_asset_contract
    robot_asset_manifest = extract_robot_asset_manifest(embodiment_context)
    runtime_target_contract = describe_isaac_runtime_targets(embodiment_context)
    payload = {
        "version": "isaaclab_backend_config_v1",
        "task_id": task_id,
        "physics_mode": "ISAAC",
        "max_steps": max(25, 8 * max(1, len(world_state.synthetic_branch_plans))),
        "time_step_s": max(1e-4, safe_float(world_state.physics_context.timestep_ms, 8.0) / 1000.0),
        "capture_sensor_bundle": True,
        "capture_rgb_frames": True,
        "render_max_frames": 50,
        "sensor_cameras": list(semantic_context.get("sensor_cameras") or ["front"]),
        "robot_asset_contract_id": "" if contract is None else contract.contract_id,
        "calibration_contracts": [] if contract is None else list(contract.calibration_contracts),
        "observation_contracts": [] if contract is None else list(contract.observation_contracts),
        "action_contracts": [] if contract is None else list(contract.action_contracts),
        "robot_asset_manifest": robot_asset_manifest,
        "normalized_robot_asset_manifest": normalize_robot_asset_manifest(embodiment_context),
        "runtime_target_contract": runtime_target_contract,
    }
    refs: list[str] = []
    if output_root is not None:
        config_path = output_root / "isaaclab_backend_config.json"
        _write_json(config_path, payload)
        refs.append(str(config_path.resolve()))
    return refs, payload


def _runtime_supports_execution(backend: str) -> bool:
    if backend == "holosoma":
        return _has_module("holosoma")
    if backend == "isaac":
        return _has_module("src.motor_backend.workcell_isaaclab_backend")
    return False


def _write_runtime_target_manifest(
    *,
    output_root: Optional[Path],
    backend: str,
    runtime_target_contract: Mapping[str, Any],
) -> list[str]:
    if output_root is None:
        return []
    path = output_root / f"{backend}_runtime_target_manifest.json"
    _write_json(path, runtime_target_contract)
    return [str(path.resolve())]


def _runtime_status_is_concrete(status: str) -> bool:
    return str(status or "") in {"runtime_execution_completed", "runtime_training_completed"}


def materialize_backend_runtime_execution(
    world_state: SimSynthPhysicsWorldState,
    execution_contract: PhysicsExecutionContract,
    backend_binding_receipt: BackendExecutionBindingReceipt,
    *,
    output_dir: str | Path | None = None,
    execute_external_launch: bool = False,
    external_launch_cwd: str | Path | None = None,
) -> Optional[BackendRuntimeExecutionReceipt]:
    """Execute or bind a concrete backend runtime when possible."""

    backend = _source_backend(execution_contract)
    if backend not in {"holosoma", "isaac"}:
        return None

    output_root = _runtime_output_root(output_dir, backend)
    policy_id = _source_policy_id(backend, world_state)
    task_id = (
        _infer_holosoma_task_id(world_state)
        if backend == "holosoma"
        else _infer_isaac_task_id(world_state)
    )
    artifact_refs: list[str] = []
    runtime_request = _runtime_request_payload(
        world_state=world_state,
        execution_contract=execution_contract,
        backend_binding_receipt=backend_binding_receipt,
        backend=backend,
        policy_id=policy_id,
        task_id=task_id,
    )
    if output_root is not None:
        request_path = output_root / "backend_runtime_request.json"
        _write_json(request_path, runtime_request)
        artifact_refs.append(str(request_path.resolve()))

    binding_refs: list[str]
    binding_payload: dict[str, Any]
    if backend == "holosoma":
        binding_refs, binding_payload = _materialize_holosoma_binding(
            world_state,
            output_root=output_root,
            task_id=task_id,
        )
    else:
        binding_refs, binding_payload = _materialize_isaac_binding(
            world_state,
            output_root=output_root,
            task_id=task_id,
        )
    artifact_refs.extend(binding_refs)
    runtime_target_contract = _mapping(binding_payload.get("runtime_target_contract"))
    binding_metadata = _mapping(backend_binding_receipt.metadata)
    binding_metadata_nested = _mapping(binding_metadata.get("binding_metadata"))
    runtime_layout_contract = _mapping(
        binding_metadata.get("runtime_layout_contract")
        or binding_metadata_nested.get("runtime_layout_contract")
    )
    policy_contract = _mapping(
        binding_metadata.get("policy_contract")
        or binding_metadata_nested.get("policy_contract")
    )
    deployment_contract = _mapping(
        binding_metadata.get("deployment_contract")
        or binding_metadata_nested.get("deployment_contract")
    )
    upstream_runtime_pack = _mapping(
        binding_metadata.get("upstream_runtime_pack")
        or binding_metadata_nested.get("upstream_runtime_pack")
    )
    runtime_bundle_refs, runtime_bundle, launch_spec = build_backend_runtime_bundle(
        backend=backend,
        task_id=task_id,
        policy_ref=policy_id,
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        robot_asset_manifest=_mapping(binding_payload.get("robot_asset_manifest")),
        normalized_robot_asset_manifest=_mapping(
            binding_payload.get("normalized_robot_asset_manifest")
        ),
        robot_contract_context={
            "robot_asset_contract_id": str(
                binding_payload.get("robot_asset_contract_id", "") or ""
            ),
            "calibration_contracts": _strings(binding_payload.get("calibration_contracts")),
            "observation_contracts": _strings(binding_payload.get("observation_contracts")),
            "action_contracts": _strings(binding_payload.get("action_contracts")),
        },
        deployment_contract=deployment_contract,
        upstream_runtime_pack=upstream_runtime_pack,
        output_root=output_root,
    )
    artifact_refs.extend(runtime_bundle_refs)
    artifact_refs.extend(
        _write_runtime_target_manifest(
            output_root=output_root,
            backend=backend,
            runtime_target_contract=runtime_target_contract,
        )
    )

    store, econ_meter = _build_store_and_meter(
        world_state=world_state,
        backend=backend,
        output_root=output_root,
        task_id=task_id,
    )
    objective = _objective_spec(world_state)
    scenario_id = f"sim_synth_{backend}_{world_state.state_id}"
    rollout_base_dir = None if output_root is None else output_root / "rollouts"

    backend_name = "holosoma" if backend == "holosoma" else "workcell_isaaclab"
    datapack_ids: list[str] = []
    datapack_configs: list[DatapackConfig] = []
    if backend == "holosoma":
        datapack_ids, datapack_configs = _resolve_holosoma_datapacks(
            world_state,
            task_id=task_id,
        )
        binding_payload["resolved_datapack_ids"] = list(datapack_ids)
        binding_payload["resolved_datapack_config_ids"] = [cfg.id for cfg in datapack_configs]

    missing_preconditions: list[str] = []
    can_train_holosoma = backend == "holosoma" and bool(datapack_ids or datapack_configs)
    require_policy = not (backend == "holosoma" and can_train_holosoma and not policy_id)
    if not policy_id and not can_train_holosoma:
        missing_preconditions.append("runtime_policy_id")
    if backend == "holosoma" and can_train_holosoma and not policy_id:
        patched_request = _mapping(
            launch_spec.get("executable_adapter_request")
            or runtime_bundle.get("executable_adapter_request")
        )
        if patched_request:
            patched_binding = build_holosoma_runtime_binding(
                task_id=task_id,
                explicit_policy_ref="",
                preferred_profile=str(
                    patched_request.get("preferred_profile") or "holosoma_motion_bank"
                ),
                launch_specs=list(runtime_bundle.get("launch_specs") or [launch_spec]),
                runtime_target_contract=runtime_target_contract,
                policy_contract=policy_contract,
                deployment_contract=deployment_contract,
                upstream_runtime_pack=mapping(runtime_bundle.get("upstream_runtime_pack")),
            )
            patched_request = build_holosoma_executable_adapter_request(
                task_id=task_id,
                policy_ref="",
                preferred_profile=str(
                    patched_request.get("preferred_profile") or "holosoma_motion_bank"
                ),
                launch_spec=launch_spec,
                runtime_target_contract=runtime_target_contract,
                policy_contract=policy_contract,
                runtime_binding=patched_binding,
                normalized_robot_asset_manifest=_mapping(
                    binding_payload.get("normalized_robot_asset_manifest")
                ),
                robot_contract_context={
                    "robot_asset_contract_id": str(
                        binding_payload.get("robot_asset_contract_id", "") or ""
                    ),
                    "calibration_contracts": _strings(binding_payload.get("calibration_contracts")),
                    "observation_contracts": _strings(binding_payload.get("observation_contracts")),
                    "action_contracts": _strings(binding_payload.get("action_contracts")),
                },
                output_contract=_mapping(runtime_bundle.get("output_contract")),
            )
            patched_consumer = build_holosoma_executable_adapter_consumer(patched_request)
            runtime_bundle["executable_adapter_request"] = patched_request
            runtime_bundle["executable_adapter_consumer"] = patched_consumer
            runtime_bundle["runtime_binding"] = patched_binding
            launch_spec["executable_adapter_request"] = patched_request
            launch_spec["executable_adapter_consumer"] = patched_consumer
            launch_spec["runtime_binding"] = patched_binding
            if not str(launch_spec.get("preferred_profile", "") or ""):
                launch_spec["preferred_profile"] = str(
                    patched_request.get("preferred_profile") or "holosoma_motion_bank"
                )
            if not str(runtime_bundle.get("preferred_profile", "") or ""):
                runtime_bundle["preferred_profile"] = str(
                    patched_request.get("preferred_profile") or "holosoma_motion_bank"
                )
    launch_plan = prepare_backend_runtime_launch(
        runtime_bundle,
        launch_spec,
        require_policy=require_policy,
    )
    executable_adapter_request = _mapping(
        launch_spec.get("executable_adapter_request")
        or runtime_bundle.get("executable_adapter_request")
    )
    executable_adapter_consumer = _mapping(
        launch_spec.get("executable_adapter_consumer")
        or runtime_bundle.get("executable_adapter_consumer")
    )
    runtime_binding = _mapping(
        launch_spec.get("runtime_binding") or runtime_bundle.get("runtime_binding")
    )
    runtime_output_contract = build_backend_runtime_output_contract(runtime_bundle, launch_spec)
    launch_report_refs: list[str] = []
    adapter_refs: list[str] = []
    launch_receipt_payload: dict[str, Any] | None = None
    adapter_receipt_payload: dict[str, Any] | None = None
    adapter_realization_payload: dict[str, Any] | None = None
    local_adapter_invocation_payload: dict[str, Any] | None = None
    local_adapter_result_payload: dict[str, Any] | None = None
    runtime_outcome_receipt_payload: dict[str, Any] | None = None
    runtime_outcome_refs: list[str] = []
    local_runtime_supported = _runtime_supports_execution(backend)
    adapter_execution: dict[str, Any] = {}
    if backend == "isaac" and executable_adapter_request and executable_adapter_consumer:
        adapter_execution = prepare_isaac_unitree_adapter_execution(
            executable_adapter_request,
            executable_adapter_consumer,
        )
        adapter_realization = build_isaac_unitree_adapter_realization(
            executable_adapter_request=executable_adapter_request,
            executable_adapter_consumer=executable_adapter_consumer,
            adapter_execution=adapter_execution,
            runtime_bundle=runtime_bundle,
            launch_spec=launch_spec,
            binding_payload={
                "executor_entrypoint": backend_binding_receipt.executor_entrypoint,
                "binding_status": backend_binding_receipt.binding_status,
            },
        )
        adapter_realization_payload = dict(adapter_realization)
        if output_root is not None:
            adapter_execution_path = output_root / "backend_runtime_adapter_execution.json"
            adapter_realization_path = output_root / "backend_runtime_adapter_realization.json"
            _write_json(adapter_execution_path, adapter_execution)
            _write_json(adapter_realization_path, adapter_realization_payload)
            adapter_refs.append(str(adapter_execution_path.resolve()))
            adapter_refs.append(str(adapter_realization_path.resolve()))
            artifact_refs.extend(adapter_refs)
    elif backend == "holosoma" and executable_adapter_request and executable_adapter_consumer:
        adapter_execution = prepare_holosoma_adapter_execution(
            executable_adapter_request,
            executable_adapter_consumer,
        )
        adapter_realization = build_holosoma_adapter_realization(
            executable_adapter_request=executable_adapter_request,
            executable_adapter_consumer=executable_adapter_consumer,
            adapter_execution=adapter_execution,
            runtime_bundle=runtime_bundle,
            launch_spec=launch_spec,
            binding_payload={
                "executor_entrypoint": backend_binding_receipt.executor_entrypoint,
                "binding_status": backend_binding_receipt.binding_status,
            },
        )
        adapter_realization_payload = dict(adapter_realization)
        if output_root is not None:
            adapter_execution_path = output_root / "backend_runtime_adapter_execution.json"
            adapter_realization_path = output_root / "backend_runtime_adapter_realization.json"
            _write_json(adapter_execution_path, adapter_execution)
            _write_json(adapter_realization_path, adapter_realization_payload)
            adapter_refs.append(str(adapter_execution_path.resolve()))
            adapter_refs.append(str(adapter_realization_path.resolve()))
            artifact_refs.extend(adapter_refs)
    if (
        adapter_realization_payload
        and str(adapter_realization_payload.get("realization_path", "") or "")
        == "local_backend_factory"
    ):
        local_adapter_invocation_payload = build_local_backend_factory_invocation(
            backend=backend,
            executable_adapter_request=executable_adapter_request,
            executable_adapter_consumer=executable_adapter_consumer,
            adapter_execution=adapter_execution,
            adapter_realization=adapter_realization_payload,
            binding_payload=binding_payload,
        )
        if output_root is not None:
            local_adapter_invocation_path = output_root / "backend_local_factory_invocation.json"
            _write_json(local_adapter_invocation_path, local_adapter_invocation_payload)
            ref = str(local_adapter_invocation_path.resolve())
            if ref not in artifact_refs:
                artifact_refs.append(ref)
    if not local_runtime_supported:
        launch_result: dict[str, Any] = dict(launch_plan)
        launch_result.setdefault("executed", False)
        if launch_plan["status"] == "ready_for_launch" and execute_external_launch:
            launch_result = execute_backend_runtime_launch(
                runtime_bundle,
                launch_spec,
                execute=True,
                cwd=external_launch_cwd,
                require_policy=require_policy,
            )
        if adapter_execution:
            if backend == "isaac":
                adapter_execution = finalize_isaac_unitree_adapter_execution(
                    adapter_execution,
                    launch_result=launch_result,
                )
                adapter_realization_payload = build_isaac_unitree_adapter_realization(
                    executable_adapter_request=executable_adapter_request,
                    executable_adapter_consumer=executable_adapter_consumer,
                    adapter_execution=adapter_execution,
                    runtime_bundle=runtime_bundle,
                    launch_spec=launch_spec,
                    binding_payload={
                        "executor_entrypoint": backend_binding_receipt.executor_entrypoint,
                        "binding_status": backend_binding_receipt.binding_status,
                    },
                )
                adapter_receipt = build_isaac_unitree_adapter_receipt(
                    adapter_execution,
                    artifact_refs=adapter_refs,
                    realization=adapter_realization_payload,
                    local_adapter_invocation=local_adapter_invocation_payload,
                    local_adapter_result=local_adapter_result_payload,
                )
            else:
                adapter_execution = finalize_holosoma_adapter_execution(
                    adapter_execution,
                    launch_result=launch_result,
                )
                adapter_realization_payload = build_holosoma_adapter_realization(
                    executable_adapter_request=executable_adapter_request,
                    executable_adapter_consumer=executable_adapter_consumer,
                    adapter_execution=adapter_execution,
                    runtime_bundle=runtime_bundle,
                    launch_spec=launch_spec,
                    binding_payload={
                        "executor_entrypoint": backend_binding_receipt.executor_entrypoint,
                        "binding_status": backend_binding_receipt.binding_status,
                    },
                )
                adapter_receipt = build_holosoma_adapter_receipt(
                    adapter_execution,
                    artifact_refs=adapter_refs,
                    realization=adapter_realization_payload,
                    local_adapter_invocation=local_adapter_invocation_payload,
                    local_adapter_result=local_adapter_result_payload,
                )
            adapter_receipt_payload = adapter_receipt.to_dict()
        launch_receipt = build_backend_runtime_launch_receipt(
            runtime_bundle,
            launch_spec,
            launch_result,
        )
        launch_receipt_payload = launch_receipt.to_dict()
        output_summary = harvest_backend_runtime_outcomes(
            runtime_output_contract,
            executed=bool(launch_receipt.executed),
        )
        outcome_receipt = build_backend_runtime_outcome_receipt(
            runtime_bundle=runtime_bundle,
            launch_receipt=launch_receipt,
            output_summary=output_summary,
        )
        runtime_outcome_receipt_payload = outcome_receipt.to_dict()
        if output_root is not None:
            report_path = output_root / "backend_runtime_launch_report.json"
            adapter_execution_path = output_root / "backend_runtime_adapter_execution.json"
            adapter_realization_path = output_root / "backend_runtime_adapter_realization.json"
            adapter_receipt_path = output_root / "backend_runtime_adapter_receipt.json"
            local_adapter_invocation_path = output_root / "backend_local_factory_invocation.json"
            local_adapter_result_path = output_root / "backend_local_factory_result.json"
            receipt_path = output_root / "backend_runtime_launch_receipt.json"
            consumer_path = output_root / "backend_executable_adapter_consumer.json"
            output_contract_path = output_root / "backend_runtime_output_contract.json"
            output_summary_path = output_root / "backend_runtime_output_summary.json"
            outcome_receipt_path = output_root / "backend_runtime_outcome_receipt.json"
            _write_json(
                report_path,
                {
                    "version": "backend_runtime_launch_report_v1",
                    "backend": backend,
                    "task_id": task_id,
                    "launch_result": launch_result,
                },
            )
            if adapter_execution:
                _write_json(adapter_execution_path, adapter_execution)
            if adapter_realization_payload is not None:
                _write_json(adapter_realization_path, adapter_realization_payload)
            if adapter_receipt_payload is not None:
                _write_json(adapter_receipt_path, adapter_receipt_payload)
            if local_adapter_invocation_payload is not None:
                _write_json(local_adapter_invocation_path, local_adapter_invocation_payload)
            if local_adapter_result_payload is not None:
                _write_json(local_adapter_result_path, local_adapter_result_payload)
            _write_json(receipt_path, launch_receipt_payload)
            _write_json(consumer_path, executable_adapter_consumer)
            _write_json(output_contract_path, runtime_output_contract)
            _write_json(output_summary_path, output_summary)
            _write_json(outcome_receipt_path, runtime_outcome_receipt_payload)
            launch_report_refs.extend(
                [
                    str(report_path.resolve()),
                    str(adapter_execution_path.resolve()) if adapter_execution else "",
                    str(adapter_realization_path.resolve()) if adapter_realization_payload else "",
                    str(adapter_receipt_path.resolve()) if adapter_receipt_payload else "",
                    str(local_adapter_invocation_path.resolve())
                    if local_adapter_invocation_payload
                    else "",
                    str(local_adapter_result_path.resolve()) if local_adapter_result_payload else "",
                    str(receipt_path.resolve()),
                    str(consumer_path.resolve()),
                    str(output_contract_path.resolve()),
                    str(output_summary_path.resolve()),
                    str(outcome_receipt_path.resolve()),
                ]
            )
            launch_report_refs = [ref for ref in launch_report_refs if ref]
            artifact_refs.extend(launch_report_refs)
        runtime_outcome_refs.extend(strings(output_summary.get("artifact_refs")))
        artifact_refs.extend(
            ref for ref in runtime_outcome_refs if ref and ref not in artifact_refs
        )
        if launch_plan["status"] == "ready_for_launch":
            execution_status = "runtime_launch_prepared"
            if execute_external_launch:
                execution_status = (
                    "runtime_external_launch_completed"
                    if str(launch_result.get("status", "")) == "launch_completed"
                    else "runtime_external_launch_failed"
                )
            return BackendRuntimeExecutionReceipt(
                receipt_id=f"backend_runtime_execution_receipt_{world_state.state_id}",
                backend=backend,
                execution_mode=(
                    f"{backend_name}_train_policy"
                    if backend == "holosoma" and can_train_holosoma and not policy_id
                    else f"{backend_name}_evaluate_policy"
                ),
                execution_status=execution_status,
                policy_id=policy_id,
                artifact_refs=artifact_refs,
                metadata={
                    "world_state_id": world_state.state_id,
                    "task_id": task_id,
                    "scenario_id": scenario_id,
                    "runtime_request": runtime_request,
                    "binding_payload": binding_payload,
                    "runtime_target_contract": runtime_target_contract,
                    "runtime_layout_contract": runtime_layout_contract,
                    "policy_contract": policy_contract,
                    "runtime_bundle": runtime_bundle,
                    "launch_spec": launch_spec,
                    "runtime_binding": runtime_binding,
                    "executable_adapter_request": executable_adapter_request,
                    "executable_adapter_consumer": executable_adapter_consumer,
                    "adapter_execution": adapter_execution,
                    "adapter_realization": adapter_realization_payload,
                    "adapter_receipt": adapter_receipt_payload,
                    "local_adapter_invocation": local_adapter_invocation_payload,
                    "local_adapter_result": local_adapter_result_payload,
                    "runtime_output_contract": runtime_output_contract,
                    "launch_plan": launch_plan,
                    "launch_receipt": launch_receipt_payload,
                    "launch_report_refs": list(launch_report_refs),
                    "runtime_outcome_receipt": runtime_outcome_receipt_payload,
                    "runtime_outcome_refs": list(runtime_outcome_refs),
                    "missing_preconditions": list(missing_preconditions),
                },
            )
        for item in strings(launch_plan.get("missing_preconditions")):
            if item not in missing_preconditions:
                missing_preconditions.append(item)
        if launch_receipt_payload is None:
            if adapter_execution:
                if backend == "isaac":
                    adapter_execution = finalize_isaac_unitree_adapter_execution(
                        adapter_execution,
                        launch_result=launch_plan,
                    )
                    adapter_realization_payload = build_isaac_unitree_adapter_realization(
                        executable_adapter_request=executable_adapter_request,
                        executable_adapter_consumer=executable_adapter_consumer,
                        adapter_execution=adapter_execution,
                        runtime_bundle=runtime_bundle,
                        launch_spec=launch_spec,
                        binding_payload={
                            "executor_entrypoint": backend_binding_receipt.executor_entrypoint,
                            "binding_status": backend_binding_receipt.binding_status,
                        },
                    )
                    adapter_receipt = build_isaac_unitree_adapter_receipt(
                        adapter_execution,
                        artifact_refs=adapter_refs,
                        realization=adapter_realization_payload,
                        local_adapter_invocation=local_adapter_invocation_payload,
                        local_adapter_result=local_adapter_result_payload,
                    )
                else:
                    adapter_execution = finalize_holosoma_adapter_execution(
                        adapter_execution,
                        launch_result=launch_plan,
                    )
                    adapter_realization_payload = build_holosoma_adapter_realization(
                        executable_adapter_request=executable_adapter_request,
                        executable_adapter_consumer=executable_adapter_consumer,
                        adapter_execution=adapter_execution,
                        runtime_bundle=runtime_bundle,
                        launch_spec=launch_spec,
                        binding_payload={
                            "executor_entrypoint": backend_binding_receipt.executor_entrypoint,
                            "binding_status": backend_binding_receipt.binding_status,
                        },
                    )
                    adapter_receipt = build_holosoma_adapter_receipt(
                        adapter_execution,
                        artifact_refs=adapter_refs,
                        realization=adapter_realization_payload,
                        local_adapter_invocation=local_adapter_invocation_payload,
                        local_adapter_result=local_adapter_result_payload,
                    )
                adapter_receipt_payload = adapter_receipt.to_dict()
            launch_receipt = build_backend_runtime_launch_receipt(
                runtime_bundle,
                launch_spec,
                launch_plan,
            )
            launch_receipt_payload = launch_receipt.to_dict()
            if output_root is not None:
                report_path = output_root / "backend_runtime_launch_report.json"
                adapter_execution_path = output_root / "backend_runtime_adapter_execution.json"
                adapter_realization_path = output_root / "backend_runtime_adapter_realization.json"
                adapter_receipt_path = output_root / "backend_runtime_adapter_receipt.json"
                local_adapter_invocation_path = output_root / "backend_local_factory_invocation.json"
                local_adapter_result_path = output_root / "backend_local_factory_result.json"
                receipt_path = output_root / "backend_runtime_launch_receipt.json"
                consumer_path = output_root / "backend_executable_adapter_consumer.json"
                output_contract_path = output_root / "backend_runtime_output_contract.json"
                output_summary_path = output_root / "backend_runtime_output_summary.json"
                outcome_receipt_path = output_root / "backend_runtime_outcome_receipt.json"
                _write_json(
                    report_path,
                    {
                        "version": "backend_runtime_launch_report_v1",
                        "backend": backend,
                        "task_id": task_id,
                        "launch_result": launch_plan,
                    },
                )
                if adapter_execution:
                    _write_json(adapter_execution_path, adapter_execution)
                if adapter_realization_payload is not None:
                    _write_json(adapter_realization_path, adapter_realization_payload)
                if adapter_receipt_payload is not None:
                    _write_json(adapter_receipt_path, adapter_receipt_payload)
                if local_adapter_invocation_payload is not None:
                    _write_json(local_adapter_invocation_path, local_adapter_invocation_payload)
                if local_adapter_result_payload is not None:
                    _write_json(local_adapter_result_path, local_adapter_result_payload)
                _write_json(receipt_path, launch_receipt_payload)
                _write_json(consumer_path, executable_adapter_consumer)
                _write_json(output_contract_path, runtime_output_contract)
                _write_json(
                    output_summary_path,
                    {
                        "version": "backend_runtime_output_summary_v1",
                        "backend": backend,
                        "profile_id": str(runtime_output_contract.get("profile_id", "") or ""),
                        "executed": False,
                        "outcome_status": "launch_not_executed",
                        "harvested_output_count": 0,
                        "artifact_kind_counts": {},
                        "source_summaries": [],
                        "artifact_refs": [],
                        "output_contract": runtime_output_contract,
                    },
                )
                _write_json(
                    outcome_receipt_path,
                    build_backend_runtime_outcome_receipt(
                        runtime_bundle=runtime_bundle,
                        launch_receipt=launch_receipt,
                        output_summary={
                            "version": "backend_runtime_output_summary_v1",
                            "backend": backend,
                            "profile_id": str(runtime_output_contract.get("profile_id", "") or ""),
                            "executed": False,
                            "outcome_status": "launch_not_executed",
                            "harvested_output_count": 0,
                            "artifact_kind_counts": {},
                            "source_summaries": [],
                            "artifact_refs": [],
                            "output_contract": runtime_output_contract,
                        },
                    ).to_dict(),
                )
                launch_report_refs.extend(
                    [
                        str(report_path.resolve()),
                        str(adapter_execution_path.resolve()) if adapter_execution else "",
                        str(adapter_realization_path.resolve()) if adapter_realization_payload else "",
                        str(adapter_receipt_path.resolve()) if adapter_receipt_payload else "",
                        str(local_adapter_invocation_path.resolve())
                        if local_adapter_invocation_payload
                        else "",
                        str(local_adapter_result_path.resolve()) if local_adapter_result_payload else "",
                        str(receipt_path.resolve()),
                        str(consumer_path.resolve()),
                        str(output_contract_path.resolve()),
                        str(output_summary_path.resolve()),
                        str(outcome_receipt_path.resolve()),
                    ]
                )
                launch_report_refs = [ref for ref in launch_report_refs if ref]
                artifact_refs.extend(launch_report_refs)
        if "backend_runtime_module" not in missing_preconditions:
            missing_preconditions.append("backend_runtime_module")

    if missing_preconditions:
        if backend == "holosoma" and not policy_id and not datapack_ids and not datapack_configs:
            missing_preconditions = [
                item for item in missing_preconditions if item != "runtime_policy_id"
            ]
            missing_preconditions.append("runtime_policy_id_or_motion_datapack")
        request_mode = (
            f"{backend_name}_train_policy"
            if backend == "holosoma" and can_train_holosoma and not policy_id
            else f"{backend_name}_evaluate_policy"
        )
        return BackendRuntimeExecutionReceipt(
            receipt_id=f"backend_runtime_execution_receipt_{world_state.state_id}",
            backend=backend,
            execution_mode=request_mode,
            execution_status="runtime_request_materialized_with_preconditions",
            policy_id=policy_id,
            artifact_refs=artifact_refs,
            metadata={
                "world_state_id": world_state.state_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "runtime_request": runtime_request,
                "binding_payload": binding_payload,
                "runtime_target_contract": runtime_target_contract,
                "runtime_layout_contract": runtime_layout_contract,
                "policy_contract": policy_contract,
                "runtime_bundle": runtime_bundle,
                "launch_spec": launch_spec,
                "runtime_binding": runtime_binding,
                "executable_adapter_request": executable_adapter_request,
                "executable_adapter_consumer": executable_adapter_consumer,
                "adapter_execution": adapter_execution,
                "adapter_realization": adapter_realization_payload,
                "adapter_receipt": adapter_receipt_payload,
                "local_adapter_invocation": local_adapter_invocation_payload,
                "local_adapter_result": local_adapter_result_payload,
                "runtime_output_contract": runtime_output_contract,
                "launch_plan": launch_plan,
                "launch_receipt": launch_receipt_payload,
                "launch_report_refs": list(launch_report_refs),
                "runtime_outcome_receipt": runtime_outcome_receipt_payload,
                "runtime_outcome_refs": list(runtime_outcome_refs),
                "missing_preconditions": missing_preconditions,
            },
        )

    execution_mode = f"{backend_name}_evaluate_policy"
    execution_status = "runtime_execution_failed"
    backend_instance = None
    if local_adapter_invocation_payload is not None:
        backend_instance, local_adapter_result_payload = materialize_local_backend_factory_invocation(
            local_adapter_invocation_payload,
            econ_meter=econ_meter,
            store=store,
        )
        if output_root is not None and local_adapter_result_payload is not None:
            local_adapter_result_path = output_root / "backend_local_factory_result.json"
            _write_json(local_adapter_result_path, local_adapter_result_payload)
            ref = str(local_adapter_result_path.resolve())
            if ref not in artifact_refs:
                artifact_refs.append(ref)
    if adapter_execution and backend in {"isaac", "holosoma"}:
        if backend == "isaac":
            adapter_execution = finalize_isaac_unitree_adapter_execution(
                adapter_execution,
                local_runtime_handoff=backend_instance is not None,
            )
            adapter_realization_payload = build_isaac_unitree_adapter_realization(
                executable_adapter_request=executable_adapter_request,
                executable_adapter_consumer=executable_adapter_consumer,
                adapter_execution=adapter_execution,
                runtime_bundle=runtime_bundle,
                launch_spec=launch_spec,
                binding_payload={
                    "executor_entrypoint": backend_binding_receipt.executor_entrypoint,
                    "binding_status": backend_binding_receipt.binding_status,
                },
            )
            adapter_receipt = build_isaac_unitree_adapter_receipt(
                adapter_execution,
                artifact_refs=adapter_refs,
                realization=adapter_realization_payload,
                local_adapter_invocation=local_adapter_invocation_payload,
                local_adapter_result=local_adapter_result_payload,
            )
        else:
            adapter_execution = finalize_holosoma_adapter_execution(
                adapter_execution,
                local_runtime_handoff=backend_instance is not None,
            )
            adapter_realization_payload = build_holosoma_adapter_realization(
                executable_adapter_request=executable_adapter_request,
                executable_adapter_consumer=executable_adapter_consumer,
                adapter_execution=adapter_execution,
                runtime_bundle=runtime_bundle,
                launch_spec=launch_spec,
                binding_payload={
                    "executor_entrypoint": backend_binding_receipt.executor_entrypoint,
                    "binding_status": backend_binding_receipt.binding_status,
                },
            )
            adapter_receipt = build_holosoma_adapter_receipt(
                adapter_execution,
                artifact_refs=adapter_refs,
                realization=adapter_realization_payload,
                local_adapter_invocation=local_adapter_invocation_payload,
                local_adapter_result=local_adapter_result_payload,
            )
        adapter_receipt_payload = adapter_receipt.to_dict()
        if output_root is not None:
            adapter_execution_path = output_root / "backend_runtime_adapter_execution.json"
            adapter_realization_path = output_root / "backend_runtime_adapter_realization.json"
            adapter_receipt_path = output_root / "backend_runtime_adapter_receipt.json"
            local_adapter_invocation_path = output_root / "backend_local_factory_invocation.json"
            local_adapter_result_path = output_root / "backend_local_factory_result.json"
            _write_json(adapter_execution_path, adapter_execution)
            _write_json(adapter_realization_path, adapter_realization_payload)
            _write_json(adapter_receipt_path, adapter_receipt_payload)
            if local_adapter_invocation_payload is not None:
                _write_json(local_adapter_invocation_path, local_adapter_invocation_payload)
            if local_adapter_result_payload is not None:
                _write_json(local_adapter_result_path, local_adapter_result_payload)
            for ref in (
                str(adapter_execution_path.resolve()),
                str(adapter_realization_path.resolve()),
                str(adapter_receipt_path.resolve()),
                (
                    str(local_adapter_invocation_path.resolve())
                    if local_adapter_invocation_payload is not None
                    else ""
                ),
                (
                    str(local_adapter_result_path.resolve())
                    if local_adapter_result_payload is not None
                    else ""
                ),
            ):
                if ref and ref not in artifact_refs:
                    artifact_refs.append(ref)
    try:
        if backend_instance is None:
            if local_adapter_invocation_payload is not None:
                raise RuntimeError(
                    str(
                        _mapping(local_adapter_result_payload).get(
                            "error",
                            _mapping(local_adapter_result_payload).get(
                                "result_status",
                                "local_backend_materialization_failed",
                            ),
                        )
                        or "local_backend_materialization_failed"
                    )
                )
            backend_instance = make_motor_backend(
                backend_name,
                econ_meter=econ_meter,
                store=store,
                backend_config=(None if backend == "holosoma" else binding_payload),
            )
        if backend_instance is None:
            raise RuntimeError(f"{backend_name} backend is not available.")
        if backend == "holosoma" and not policy_id and can_train_holosoma:
            preferred_num_envs = max(1, min(8, len(world_state.synthetic_branch_plans) or 1))
            preferred_max_steps = max(64, 32 * max(1, len(world_state.synthetic_branch_plans)))
            result = backend_instance.train_policy(
                task_id=task_id,
                objective=objective,
                datapack_ids=datapack_ids,
                datapack_configs=datapack_configs,
                num_envs=preferred_num_envs,
                max_steps=preferred_max_steps,
                scenario_id=scenario_id,
                rollout_base_dir=rollout_base_dir,
                seed=0,
            )
            policy_id = str(getattr(result, "policy_id", "") or "")
            execution_mode = f"{backend_name}_train_policy"
            execution_status = "runtime_training_completed"
        else:
            result = backend_instance.evaluate_policy(
                policy_id=policy_id,
                task_id=task_id,
                objective=objective,
                num_episodes=1,
                scenario_id=scenario_id,
                rollout_base_dir=rollout_base_dir,
                seed=0,
            )
            execution_mode = f"{backend_name}_evaluate_policy"
            execution_status = "runtime_execution_completed"
    except Exception as exc:
        if output_root is not None:
            failure_path = output_root / "backend_runtime_failure.json"
            _write_json(
                failure_path,
                {
                    "version": "backend_runtime_failure_v1",
                    "backend": backend,
                    "task_id": task_id,
                    "policy_id": policy_id,
                    "error": str(exc),
                },
            )
            artifact_refs.append(str(failure_path.resolve()))
        return BackendRuntimeExecutionReceipt(
            receipt_id=f"backend_runtime_execution_receipt_{world_state.state_id}",
            backend=backend,
            execution_mode=execution_mode,
            execution_status="runtime_execution_failed",
            policy_id=policy_id,
            artifact_refs=artifact_refs,
            metadata={
                "world_state_id": world_state.state_id,
                "task_id": task_id,
                "scenario_id": scenario_id,
                "runtime_request": runtime_request,
                "binding_payload": binding_payload,
                "runtime_target_contract": runtime_target_contract,
                "runtime_layout_contract": runtime_layout_contract,
                "policy_contract": policy_contract,
                "runtime_bundle": runtime_bundle,
                "launch_spec": launch_spec,
                "runtime_binding": runtime_binding,
                "executable_adapter_request": executable_adapter_request,
                "executable_adapter_consumer": executable_adapter_consumer,
                "adapter_execution": adapter_execution,
                "adapter_realization": adapter_realization_payload,
                "adapter_receipt": adapter_receipt_payload,
                "local_adapter_invocation": local_adapter_invocation_payload,
                "local_adapter_result": local_adapter_result_payload,
                "runtime_output_contract": runtime_output_contract,
                "launch_plan": launch_plan,
                "launch_receipt": launch_receipt_payload,
                "launch_report_refs": list(launch_report_refs),
                "runtime_outcome_receipt": runtime_outcome_receipt_payload,
                "runtime_outcome_refs": list(runtime_outcome_refs),
                "error": str(exc),
            },
        )

    rollout_bundle = getattr(result, "rollout_bundle", None)
    artifact_refs.extend(_rollout_artifact_refs(rollout_bundle))
    raw_metrics = mapping(getattr(result, "raw_metrics", {}))
    econ_metrics = mapping(getattr(result, "econ_metrics", {}))
    if output_root is not None:
        metrics_path = output_root / "backend_runtime_metrics.json"
        _write_json(
            metrics_path,
            {
                "version": "backend_runtime_metrics_v1",
                "backend": backend,
                "task_id": task_id,
                "policy_id": policy_id,
                "raw_metrics": raw_metrics,
                "econ_metrics": econ_metrics,
            },
        )
        artifact_refs.append(str(metrics_path.resolve()))
    if policy_id and Path(policy_id).exists():
        policy_ref = str(Path(policy_id).resolve())
        if policy_ref not in artifact_refs:
            artifact_refs.append(policy_ref)
    local_output_summary = harvest_backend_runtime_outcomes(
        runtime_output_contract,
        executed=True,
        explicit_artifact_refs=artifact_refs,
        explicit_policy_ref=policy_id,
    )
    local_outcome_receipt = build_backend_runtime_outcome_receipt(
        runtime_bundle=runtime_bundle,
        launch_receipt=None,
        output_summary=local_output_summary,
    )
    runtime_outcome_receipt_payload = local_outcome_receipt.to_dict()
    runtime_outcome_refs = strings(local_output_summary.get("artifact_refs"))
    for ref in runtime_outcome_refs:
        if ref and ref not in artifact_refs:
            artifact_refs.append(ref)
    if output_root is not None:
        consumer_path = output_root / "backend_executable_adapter_consumer.json"
        output_contract_path = output_root / "backend_runtime_output_contract.json"
        output_summary_path = output_root / "backend_runtime_output_summary.json"
        outcome_receipt_path = output_root / "backend_runtime_outcome_receipt.json"
        _write_json(consumer_path, executable_adapter_consumer)
        _write_json(output_contract_path, runtime_output_contract)
        _write_json(output_summary_path, local_output_summary)
        _write_json(outcome_receipt_path, runtime_outcome_receipt_payload)
        for ref in (
            str(consumer_path.resolve()),
            str(output_contract_path.resolve()),
            str(output_summary_path.resolve()),
            str(outcome_receipt_path.resolve()),
        ):
            if ref not in artifact_refs:
                artifact_refs.append(ref)
    return BackendRuntimeExecutionReceipt(
        receipt_id=f"backend_runtime_execution_receipt_{world_state.state_id}",
        backend=backend,
        execution_mode=execution_mode,
        execution_status=execution_status,
        policy_id=policy_id,
        artifact_refs=artifact_refs,
        metadata={
            "world_state_id": world_state.state_id,
            "task_id": task_id,
            "scenario_id": scenario_id,
            "runtime_request": runtime_request,
            "binding_payload": binding_payload,
            "runtime_target_contract": runtime_target_contract,
            "runtime_layout_contract": runtime_layout_contract,
            "policy_contract": policy_contract,
            "runtime_bundle": runtime_bundle,
            "launch_spec": launch_spec,
            "runtime_binding": runtime_binding,
            "executable_adapter_request": executable_adapter_request,
            "executable_adapter_consumer": executable_adapter_consumer,
            "adapter_execution": adapter_execution,
            "adapter_realization": adapter_realization_payload,
            "adapter_receipt": adapter_receipt_payload,
            "local_adapter_invocation": local_adapter_invocation_payload,
            "local_adapter_result": local_adapter_result_payload,
            "runtime_output_contract": runtime_output_contract,
            "launch_plan": launch_plan,
            "launch_receipt": launch_receipt_payload,
            "launch_report_refs": list(launch_report_refs),
            "runtime_outcome_receipt": runtime_outcome_receipt_payload,
            "runtime_outcome_refs": list(runtime_outcome_refs),
            "datapack_ids": list(datapack_ids),
            "datapack_config_ids": [cfg.id for cfg in datapack_configs],
            "raw_metrics": raw_metrics,
            "econ_metrics": econ_metrics,
            "rollout_episode_count": len(list(getattr(rollout_bundle, "episodes", []) or [])),
        },
    )


__all__ = ["materialize_backend_runtime_execution"]

"""Bridge contracts between WM slow-loop planning and backend runtimes."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .common import clip01, mapping, safe_float, stable_id, strings
from .receipts import (
    BackendRuntimeBridgeReceipt,
    BackendRuntimeExecutionReceipt,
    BackendShadowExecutionReceipt,
    RobotAssetContractReceipt,
)
from .state import (
    BackendExecutionBindingState,
    BackendRuntimeBridgeState,
    PhysicsContextState,
    RobotAssetContractState,
)


def _int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return int(default)


def _control_constraints(embodiment_context: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    payload = mapping(embodiment_context)
    return mapping(
        payload.get("control_constraints")
        or payload.get("action_constraints")
        or payload.get("latency_envelope")
    )


def _rate_hz(value: Any, default: float) -> float:
    rate = safe_float(value, default)
    if rate > 0.0:
        return float(rate)
    return float(default)


def _planner_rate_hz(control_constraints: Mapping[str, Any], control_rate_hz: float) -> float:
    explicit = safe_float(
        control_constraints.get(
            "planner_rate_hz",
            control_constraints.get(
                "policy_rate_hz",
                control_constraints.get("rollout_rate_hz", 0.0),
            ),
        ),
        0.0,
    )
    if explicit > 0.0:
        return explicit
    decimation = _int(
        control_constraints.get(
            "action_decimation",
            control_constraints.get(
                "policy_decimation",
                control_constraints.get("decimation", 1),
            ),
        ),
        1,
    )
    return max(control_rate_hz / float(decimation), 1.0)


def _transport_profile(
    backend: str,
    runtime_target_contract: Mapping[str, Any],
) -> str:
    ready_targets = set(strings(runtime_target_contract.get("ready_target_ids")))
    if backend == "isaac":
        if "unitree_sdk2_root" in ready_targets:
            return "isaaclab_unitree_dds_bridge"
        if "isaaclab_root" in ready_targets or "isaacsim_root" in ready_targets:
            return "isaac_sim_runtime_bridge"
        return "isaac_shadow_bridge"
    if backend == "holosoma":
        return "holosoma_motion_runtime_bridge"
    return "local_python_sim_bridge"


def _transport_stack(
    backend: str,
    binding: BackendExecutionBindingState,
    runtime_target_contract: Mapping[str, Any],
) -> list[str]:
    ready_targets = set(strings(runtime_target_contract.get("ready_target_ids")))
    stack: list[str] = []
    if bool(runtime_target_contract.get("python_bridge_available", False)):
        stack.append("python_bridge")
    stack.extend(strings(binding.target_runtime_stack))
    if backend == "isaac":
        if "unitree_sdk2_root" in ready_targets:
            stack.append("dds")
        if "isaaclab_root" in ready_targets or "isaacsim_root" in ready_targets:
            stack.append("sim_runtime")
    elif backend == "holosoma":
        stack.append("motion_datapack")
        if "retargeting_root" in ready_targets:
            stack.append("retargeting")
    else:
        stack.append("inproc")
    deduped: list[str] = []
    for item in stack:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _layout_ready_profiles(binding: BackendExecutionBindingState) -> list[str]:
    layout_contract = mapping(binding.metadata.get("runtime_layout_contract"))
    runtime_target_contract = mapping(binding.metadata.get("runtime_target_contract"))
    ready_profiles = strings(layout_contract.get("ready_profiles"))
    ready_targets = set(strings(runtime_target_contract.get("ready_target_ids")))
    target_profile_map = {
        "isaac": {
            "isaaclab_root": "isaaclab_core",
            "isaacsim_root": "isaaclab_core",
            "unitree_sim_isaaclab_root": "unitree_sim_isaaclab",
            "unitree_rl_gym_root": "unitree_rl_gym",
            "humanoidverse_root": "humanoidverse",
            "xr_teleoperate_root": "xr_teleoperate",
            "unitree_model_root": "unitree_model_assets",
            "unitree_asset_root": "unitree_model_assets",
            "unitree_policy_root": "unitree_rl_gym",
        },
        "holosoma": {
            "holosoma_root": "holosoma_repo",
            "holosoma_motion_root": "holosoma_motion_bank",
            "holosoma_policy_root": "holosoma_policy_bank",
            "retargeting_root": "retargeting_bundle",
        },
    }
    for target_id in ready_targets:
        profile_id = target_profile_map.get(binding.backend, {}).get(target_id, "")
        if profile_id and profile_id not in ready_profiles:
            ready_profiles.append(profile_id)
    return ready_profiles


def _policy_ready(binding: BackendExecutionBindingState) -> bool:
    return bool(
        mapping(binding.metadata.get("policy_contract")).get("policy_ready", False)
    )


def _telemetry_contracts(
    *,
    backend: str,
    target_hardware_class: str,
    observation_contracts: list[str],
) -> list[str]:
    contracts = set(observation_contracts)
    if backend == "isaac":
        contracts.add("sim_clock_state_v1")
        contracts.add("latency_trace_v1")
    if backend == "holosoma":
        contracts.add("motion_tracking_state_v1")
        contracts.add("retargeting_trace_v1")
    if target_hardware_class == "unitree_g1_r1_class":
        contracts.update(
            {
                "joint_effort_state_v1",
                "battery_thermal_state_v1",
                "watchdog_state_v1",
            }
        )
    return sorted(contract for contract in contracts if contract)


def _safety_channels(
    *,
    backend: str,
    target_hardware_class: str,
    contract_metadata: Mapping[str, Any],
) -> list[str]:
    normalized_manifest = mapping(contract_metadata.get("normalized_asset_manifest"))
    channels = {"joint_limit_guard_v1", "watchdog_v1"}
    if target_hardware_class == "unitree_g1_r1_class":
        channels.update(
            {
                "e_stop_v1",
                "support_phase_guard_v1",
                "whole_body_balance_guard_v1",
            }
        )
    if bool(mapping(normalized_manifest.get("self_collision_profile")).get("present", False)):
        channels.add("self_collision_guard_v1")
    if backend == "holosoma":
        channels.add("retargeting_guard_v1")
    return sorted(channels)


def _binding_status_score(binding_status: str) -> float:
    if binding_status in {"runtime_ready", "ready"}:
        return 1.0
    if binding_status in {"shadow_ready", "runtime_assets_missing"}:
        return 0.65
    if binding_status in {"assets_missing", "integration_pending"}:
        return 0.35
    return 0.15


def _bridge_status(
    *,
    backend: str,
    binding_status: str,
    runtime_target_contract: Mapping[str, Any],
    missing_assets: list[str],
) -> str:
    if backend == "pybullet":
        return "runtime_bridge_ready"
    runtime_targets_ready = bool(runtime_target_contract.get("runtime_targets_ready", False))
    missing_target_ids = strings(runtime_target_contract.get("missing_required_target_ids"))
    unresolved_groups = list(runtime_target_contract.get("unresolved_one_of_groups", []) or [])
    if binding_status in {"runtime_ready", "ready"} and runtime_targets_ready and not missing_assets:
        return "runtime_bridge_ready"
    if missing_assets and runtime_targets_ready:
        return "runtime_assets_missing"
    if missing_target_ids or unresolved_groups:
        return "runtime_targets_missing"
    if binding_status in {"shadow_ready", "runtime_assets_missing", "assets_missing"} or bool(
        runtime_target_contract.get("python_bridge_available", False)
    ):
        return "shadow_bridge_only"
    return "planning_only"


def compile_backend_runtime_bridge(
    physics_context: PhysicsContextState,
    backend_execution_binding: BackendExecutionBindingState,
    *,
    robot_asset_contract: Optional[RobotAssetContractState],
    embodiment_context: Optional[Mapping[str, Any]] = None,
) -> BackendRuntimeBridgeState:
    control_constraints = _control_constraints(embodiment_context)
    runtime_target_contract = mapping(
        backend_execution_binding.metadata.get("runtime_target_contract")
    )
    control_rate_hz = _rate_hz(
        control_constraints.get(
            "control_rate_hz",
            control_constraints.get(
                "servo_rate_hz",
                control_constraints.get("control_frequency_hz", 0.0),
            ),
        ),
        default=max(1.0, 1000.0 / max(1.0, safe_float(physics_context.timestep_ms, 8.0))),
    )
    planner_rate_hz = _planner_rate_hz(control_constraints, control_rate_hz)
    observation_rate_hz = _rate_hz(
        control_constraints.get(
            "observation_rate_hz",
            control_constraints.get(
                "sensor_rate_hz",
                control_constraints.get("telemetry_rate_hz", planner_rate_hz),
            ),
        ),
        planner_rate_hz,
    )
    action_decimation = _int(
        control_constraints.get(
            "action_decimation",
            control_constraints.get(
                "policy_decimation",
                control_constraints.get("decimation", 1),
            ),
        ),
        1,
    )
    contract_metadata = (
        {} if robot_asset_contract is None else mapping(robot_asset_contract.metadata)
    )
    target_hardware_class = (
        ""
        if robot_asset_contract is None
        else str(robot_asset_contract.target_hardware_class)
    )
    missing_assets = [] if robot_asset_contract is None else list(robot_asset_contract.missing_assets)
    required_targets = strings(runtime_target_contract.get("required_target_ids"))
    ready_targets = strings(runtime_target_contract.get("ready_target_ids"))
    missing_targets = strings(runtime_target_contract.get("missing_required_target_ids"))
    unresolved_groups = list(runtime_target_contract.get("unresolved_one_of_groups", []) or [])
    bridge_status = _bridge_status(
        backend=backend_execution_binding.backend,
        binding_status=backend_execution_binding.binding_status,
        runtime_target_contract=runtime_target_contract,
        missing_assets=missing_assets,
    )
    asset_score = (
        1.0
        if robot_asset_contract is None or not robot_asset_contract.required_assets
        else 1.0
        - (
            len(missing_assets)
            / float(max(1, len(robot_asset_contract.required_assets)))
        )
    )
    target_score = (
        1.0
        if not required_targets
        else 1.0 - (len(missing_targets) / float(max(1, len(required_targets))))
    )
    group_score = (
        1.0
        if not runtime_target_contract.get("one_of_target_groups")
        else 1.0
        - (
            len(unresolved_groups)
            / float(max(1, len(runtime_target_contract.get("one_of_target_groups", []) or [])))
        )
    )
    bridge_readiness_score = clip01(
        (0.3 * asset_score)
        + (0.25 * target_score)
        + (0.15 * group_score)
        + (0.15 * _binding_status_score(backend_execution_binding.binding_status))
        + (0.15 * float(bool(runtime_target_contract.get("python_bridge_available", False))))
    )
    observation_contracts = (
        [] if robot_asset_contract is None else list(robot_asset_contract.observation_contracts)
    )
    action_contracts = (
        [] if robot_asset_contract is None else list(robot_asset_contract.action_contracts)
    )
    payload = {
        "backend": backend_execution_binding.backend,
        "bridge_status": bridge_status,
        "transport_profile": _transport_profile(
            backend_execution_binding.backend,
            runtime_target_contract,
        ),
        "binding_id": backend_execution_binding.binding_id,
        "contract_id": "" if robot_asset_contract is None else robot_asset_contract.contract_id,
        "planner_rate_hz": planner_rate_hz,
        "control_rate_hz": control_rate_hz,
        "action_decimation": action_decimation,
    }
    return BackendRuntimeBridgeState(
        bridge_id=stable_id("backend_runtime_bridge", payload),
        backend=backend_execution_binding.backend,
        bridge_status=bridge_status,
        transport_profile=_transport_profile(
            backend_execution_binding.backend,
            runtime_target_contract,
        ),
        transport_stack=_transport_stack(
            backend_execution_binding.backend,
            backend_execution_binding,
            runtime_target_contract,
        ),
        required_runtime_targets=required_targets,
        ready_runtime_targets=ready_targets,
        missing_runtime_targets=missing_targets,
        planner_rate_hz=planner_rate_hz,
        control_rate_hz=control_rate_hz,
        observation_rate_hz=observation_rate_hz,
        action_decimation=action_decimation,
        latency_budget_ms=safe_float(control_constraints.get("latency_budget_ms", 0.0), 0.0),
        bridge_readiness_score=bridge_readiness_score,
        action_contracts=action_contracts,
        observation_contracts=observation_contracts,
        telemetry_contracts=_telemetry_contracts(
            backend=backend_execution_binding.backend,
            target_hardware_class=target_hardware_class,
            observation_contracts=observation_contracts,
        ),
        safety_channels=_safety_channels(
            backend=backend_execution_binding.backend,
            target_hardware_class=target_hardware_class,
            contract_metadata=contract_metadata,
        ),
        metadata={
            "binding_id": backend_execution_binding.binding_id,
            "binding_status": backend_execution_binding.binding_status,
            "executor_entrypoint": backend_execution_binding.executor_entrypoint,
            "executor_kind": backend_execution_binding.executor_kind,
            "runtime_target_contract": runtime_target_contract,
            "runtime_layout_contract": mapping(
                backend_execution_binding.metadata.get("runtime_layout_contract")
            ),
            "policy_contract": mapping(
                backend_execution_binding.metadata.get("policy_contract")
            ),
            "target_hardware_class": target_hardware_class,
            "missing_assets": missing_assets,
            "calibration_contracts": (
                [] if robot_asset_contract is None else list(robot_asset_contract.calibration_contracts)
            ),
            "normalized_asset_manifest": mapping(
                contract_metadata.get("normalized_asset_manifest")
            ),
            "control_constraints": control_constraints,
            "runtime_targets_ready": bool(runtime_target_contract.get("runtime_targets_ready", False)),
            "unresolved_one_of_groups": unresolved_groups,
            "runtime_layout_ready_profiles": _layout_ready_profiles(backend_execution_binding),
            "policy_ready": _policy_ready(backend_execution_binding),
        },
    )


def _runtime_status_is_concrete(status: str) -> bool:
    return str(status or "") in {"runtime_execution_completed", "runtime_training_completed"}


def build_backend_runtime_bridge_receipt(
    *,
    bridge_state: Optional[BackendRuntimeBridgeState],
    backend_binding_receipt_id: str,
    robot_asset_contract_receipt: Optional[RobotAssetContractReceipt],
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt],
    backend_shadow_execution_receipt: Optional[BackendShadowExecutionReceipt],
    world_state_id: str,
    physics_execution_contract_id: str,
    route_status: str,
    requested_backend: str,
    resolved_backend: str,
    fallback_reason: str,
) -> BackendRuntimeBridgeReceipt:
    if bridge_state is None:
        return BackendRuntimeBridgeReceipt(
            receipt_id=f"backend_runtime_bridge_receipt_{world_state_id}",
            bridge_id="",
            backend=resolved_backend,
            bridge_status="missing",
            execution_authority="planning_only",
            transport_profile="",
            planner_rate_hz=0.0,
            control_rate_hz=0.0,
            observation_rate_hz=0.0,
            action_decimation=1,
            latency_budget_ms=0.0,
            bridge_readiness_score=0.0,
            metadata={
                "world_state_id": world_state_id,
                "physics_execution_contract_id": physics_execution_contract_id,
                "backend_execution_binding_receipt_id": backend_binding_receipt_id,
                "requested_backend": requested_backend,
                "resolved_backend": resolved_backend,
                "route_status": route_status,
                "fallback_reason": fallback_reason,
            },
        )
    execution_authority = "planning_only"
    if backend_runtime_execution_receipt is not None and _runtime_status_is_concrete(
        backend_runtime_execution_receipt.execution_status
    ):
        execution_authority = "concrete_runtime"
    elif backend_shadow_execution_receipt is not None:
        execution_authority = "shadow_runtime"
    elif backend_runtime_execution_receipt is not None:
        execution_authority = "runtime_request_only"
    artifact_refs: list[str] = []
    for refs in (
        [] if backend_runtime_execution_receipt is None else backend_runtime_execution_receipt.artifact_refs,
        [] if backend_shadow_execution_receipt is None else backend_shadow_execution_receipt.artifact_refs,
    ):
        for ref in refs:
            if ref and ref not in artifact_refs:
                artifact_refs.append(str(ref))
    robot_asset_receipt = robot_asset_contract_receipt
    return BackendRuntimeBridgeReceipt(
        receipt_id=f"backend_runtime_bridge_receipt_{world_state_id}",
        bridge_id=bridge_state.bridge_id,
        backend=bridge_state.backend,
        bridge_status=bridge_state.bridge_status,
        execution_authority=execution_authority,
        transport_profile=bridge_state.transport_profile,
        planner_rate_hz=bridge_state.planner_rate_hz,
        control_rate_hz=bridge_state.control_rate_hz,
        observation_rate_hz=bridge_state.observation_rate_hz,
        action_decimation=bridge_state.action_decimation,
        latency_budget_ms=bridge_state.latency_budget_ms,
        bridge_readiness_score=bridge_state.bridge_readiness_score,
        action_contracts=list(bridge_state.action_contracts),
        observation_contracts=list(bridge_state.observation_contracts),
        telemetry_contracts=list(bridge_state.telemetry_contracts),
        safety_channels=list(bridge_state.safety_channels),
        artifact_refs=artifact_refs,
        metadata={
            "world_state_id": world_state_id,
            "physics_execution_contract_id": physics_execution_contract_id,
            "backend_execution_binding_receipt_id": backend_binding_receipt_id,
            "robot_asset_contract_receipt_id": (
                "" if robot_asset_receipt is None else robot_asset_receipt.receipt_id
            ),
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
            "backend_shadow_execution_receipt_id": (
                ""
                if backend_shadow_execution_receipt is None
                else backend_shadow_execution_receipt.receipt_id
            ),
            "backend_shadow_execution_status": (
                ""
                if backend_shadow_execution_receipt is None
                else backend_shadow_execution_receipt.execution_status
            ),
            "requested_backend": requested_backend,
            "resolved_backend": resolved_backend,
            "route_status": route_status,
            "fallback_reason": fallback_reason,
            "target_hardware_class": mapping(bridge_state.metadata).get("target_hardware_class", ""),
            "runtime_target_contract": mapping(bridge_state.metadata).get("runtime_target_contract", {}),
            "runtime_layout_contract": mapping(bridge_state.metadata).get(
                "runtime_layout_contract", {}
            ),
            "policy_contract": mapping(bridge_state.metadata).get("policy_contract", {}),
            "missing_assets": mapping(bridge_state.metadata).get("missing_assets", []),
            "runtime_targets_ready": mapping(bridge_state.metadata).get("runtime_targets_ready", False),
            "runtime_layout_ready_profiles": mapping(bridge_state.metadata).get(
                "runtime_layout_ready_profiles", []
            ),
            "policy_ready": mapping(bridge_state.metadata).get("policy_ready", False),
        },
    )


__all__ = [
    "BackendRuntimeBridgeReceipt",
    "BackendRuntimeBridgeState",
    "build_backend_runtime_bridge_receipt",
    "compile_backend_runtime_bridge",
]

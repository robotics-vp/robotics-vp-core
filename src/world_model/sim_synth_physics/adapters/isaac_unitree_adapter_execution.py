"""Execution mediation over Isaac/Unitree executable-adapter consumers."""

from __future__ import annotations

import importlib.util
from typing import Any, Mapping

from ..common import mapping, stable_id, strings
from ..receipts import BackendRuntimeAdapterReceipt


LOCAL_BRIDGE_MODULE = "src.motor_backend.workcell_isaaclab_backend"


def _has_local_bridge_module() -> bool:
    try:
        return importlib.util.find_spec(LOCAL_BRIDGE_MODULE) is not None
    except Exception:
        return False


def prepare_isaac_unitree_adapter_execution(
    executable_adapter_request: Mapping[str, Any],
    executable_adapter_consumer: Mapping[str, Any],
) -> dict[str, Any]:
    request = mapping(executable_adapter_request)
    consumer = mapping(executable_adapter_consumer)
    consumer_mode = str(consumer.get("consumer_mode", "") or "")
    consumer_status = str(consumer.get("consumer_status", "") or "")
    local_bridge_available = _has_local_bridge_module()
    missing_preconditions = strings(consumer.get("missing_preconditions"))
    notes = strings(consumer.get("notes")) + strings(request.get("notes"))

    execution_path = "blocked"
    adapter_status = "adapter_blocked"
    if consumer_mode == "local_python_bridge":
        execution_path = "local_python_bridge"
        if local_bridge_available and consumer_status == "local_python_bridge_ready":
            adapter_status = "local_bridge_ready"
        else:
            adapter_status = "local_bridge_missing"
            if "local_python_bridge_module" not in missing_preconditions:
                missing_preconditions.append("local_python_bridge_module")
    elif consumer_status == "external_launch_consumer_ready":
        execution_path = "external_launch"
        adapter_status = "external_launch_ready"

    payload = {
        "request_id": str(request.get("request_id", "") or ""),
        "consumer_id": str(consumer.get("consumer_id", "") or ""),
        "adapter_family": str(request.get("adapter_family", "") or "isaac_unitree"),
        "adapter_entrypoint": str(request.get("adapter_entrypoint", "") or ""),
        "deployment_mode": str(request.get("deployment_mode", "") or ""),
        "consumer_mode": consumer_mode,
        "consumer_status": consumer_status,
        "execution_path": execution_path,
        "adapter_status": adapter_status,
        "local_bridge_module": LOCAL_BRIDGE_MODULE,
        "local_bridge_available": local_bridge_available,
        "missing_preconditions": list(dict.fromkeys(missing_preconditions)),
    }
    return {
        "version": "backend_executable_adapter_execution_v1",
        "execution_id": stable_id("backend_executable_adapter_execution", payload),
        **payload,
        "task_id": str(request.get("task_id", "") or ""),
        "policy_ref": str(request.get("policy_ref", "") or ""),
        "robot_variant": str(request.get("robot_variant", "") or ""),
        "placement_class": str(request.get("placement_class", "") or ""),
        "command": str(consumer.get("command", request.get("command", "")) or ""),
        "cwd": str(consumer.get("cwd", request.get("cwd", "")) or ""),
        "env_overrides": mapping(consumer.get("env_overrides") or request.get("env_overrides")),
        "notes": list(dict.fromkeys(notes)),
    }


def finalize_isaac_unitree_adapter_execution(
    adapter_execution: Mapping[str, Any],
    *,
    local_runtime_handoff: bool = False,
    launch_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = mapping(adapter_execution)
    execution_path = str(payload.get("execution_path", "") or "")
    result = mapping(launch_result)
    adapter_status = str(payload.get("adapter_status", "") or "adapter_blocked")
    executed = bool(result.get("executed", False))
    if execution_path == "local_python_bridge" and local_runtime_handoff:
        adapter_status = "local_bridge_handed_off"
    elif execution_path == "external_launch":
        raw_status = str(result.get("status", "") or "")
        if executed and raw_status == "launch_completed":
            adapter_status = "external_launch_completed"
        elif executed:
            adapter_status = "external_launch_failed"
    return {
        **payload,
        "adapter_status": adapter_status,
        "executed": executed,
        "launch_status": str(result.get("status", "") or ""),
        "returncode": result.get("returncode"),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }


def build_isaac_unitree_adapter_receipt(
    adapter_execution: Mapping[str, Any],
    *,
    artifact_refs: list[str] | None = None,
) -> BackendRuntimeAdapterReceipt:
    payload = mapping(adapter_execution)
    receipt_payload = {
        "backend": "isaac",
        "adapter_family": str(payload.get("adapter_family", "") or "isaac_unitree"),
        "adapter_entrypoint": str(payload.get("adapter_entrypoint", "") or ""),
        "consumer_mode": str(payload.get("consumer_mode", "") or ""),
        "adapter_status": str(payload.get("adapter_status", "") or ""),
        "execution_path": str(payload.get("execution_path", "") or ""),
        "executed": bool(payload.get("executed", False)),
        "request_id": str(payload.get("request_id", "") or ""),
        "consumer_id": str(payload.get("consumer_id", "") or ""),
        "execution_id": str(payload.get("execution_id", "") or ""),
    }
    return BackendRuntimeAdapterReceipt(
        receipt_id=stable_id("backend_runtime_adapter_receipt", receipt_payload),
        backend="isaac",
        adapter_family=str(payload.get("adapter_family", "") or "isaac_unitree"),
        adapter_entrypoint=str(payload.get("adapter_entrypoint", "") or ""),
        consumer_mode=str(payload.get("consumer_mode", "") or ""),
        adapter_status=str(payload.get("adapter_status", "") or ""),
        execution_path=str(payload.get("execution_path", "") or ""),
        executed=bool(payload.get("executed", False)),
        artifact_refs=strings(artifact_refs or []),
        metadata={
            "execution_id": str(payload.get("execution_id", "") or ""),
            "request_id": str(payload.get("request_id", "") or ""),
            "consumer_id": str(payload.get("consumer_id", "") or ""),
            "deployment_mode": str(payload.get("deployment_mode", "") or ""),
            "task_id": str(payload.get("task_id", "") or ""),
            "policy_ref": str(payload.get("policy_ref", "") or ""),
            "robot_variant": str(payload.get("robot_variant", "") or ""),
            "placement_class": str(payload.get("placement_class", "") or ""),
            "consumer_status": str(payload.get("consumer_status", "") or ""),
            "local_bridge_module": str(payload.get("local_bridge_module", "") or ""),
            "local_bridge_available": bool(payload.get("local_bridge_available", False)),
            "missing_preconditions": strings(payload.get("missing_preconditions")),
            "launch_status": str(payload.get("launch_status", "") or ""),
            "returncode": payload.get("returncode"),
            "notes": strings(payload.get("notes")),
        },
    )


__all__ = [
    "LOCAL_BRIDGE_MODULE",
    "build_isaac_unitree_adapter_receipt",
    "finalize_isaac_unitree_adapter_execution",
    "prepare_isaac_unitree_adapter_execution",
]

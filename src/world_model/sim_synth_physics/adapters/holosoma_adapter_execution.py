"""Execution mediation over Holosoma executable-adapter consumers."""

from __future__ import annotations

import importlib.util
from typing import Any, Mapping

from ..common import mapping, stable_id, strings
from ..receipts import BackendRuntimeAdapterReceipt


LOCAL_RUNTIME_MODULE = "holosoma"


def _has_local_runtime_module() -> bool:
    try:
        return importlib.util.find_spec(LOCAL_RUNTIME_MODULE) is not None
    except Exception:
        return False


def prepare_holosoma_adapter_execution(
    executable_adapter_request: Mapping[str, Any],
    executable_adapter_consumer: Mapping[str, Any],
) -> dict[str, Any]:
    request = mapping(executable_adapter_request)
    consumer = mapping(executable_adapter_consumer)
    consumer_mode = str(consumer.get("consumer_mode", "") or "")
    consumer_status = str(consumer.get("consumer_status", "") or "")
    local_runtime_available = _has_local_runtime_module()
    missing_preconditions = strings(consumer.get("missing_preconditions"))
    notes = strings(consumer.get("notes")) + strings(request.get("notes"))

    execution_path = "blocked"
    adapter_status = "adapter_blocked"
    if consumer_mode == "local_runtime_binding":
        execution_path = "local_runtime_binding"
        if local_runtime_available and consumer_status == "local_runtime_binding_ready":
            adapter_status = "local_runtime_binding_ready"
        else:
            adapter_status = "local_runtime_binding_missing"
            if "holosoma_runtime_module" not in missing_preconditions:
                missing_preconditions.append("holosoma_runtime_module")
    elif consumer_status == "external_launch_consumer_ready":
        execution_path = "external_launch"
        adapter_status = "external_launch_ready"

    payload = {
        "request_id": str(request.get("request_id", "") or ""),
        "consumer_id": str(consumer.get("consumer_id", "") or ""),
        "adapter_family": str(request.get("adapter_family", "") or "holosoma"),
        "adapter_entrypoint": str(request.get("adapter_entrypoint", "") or ""),
        "deployment_mode": str(request.get("deployment_mode", "") or ""),
        "consumer_mode": consumer_mode,
        "consumer_status": consumer_status,
        "execution_path": execution_path,
        "adapter_status": adapter_status,
        "local_runtime_module": LOCAL_RUNTIME_MODULE,
        "local_runtime_available": local_runtime_available,
        "missing_preconditions": list(dict.fromkeys(missing_preconditions)),
    }
    return {
        "version": "backend_executable_adapter_execution_v1",
        "execution_id": stable_id("backend_executable_adapter_execution", payload),
        **payload,
        "task_id": str(request.get("task_id", "") or ""),
        "policy_ref": str(request.get("policy_ref", "") or ""),
        "command": str(consumer.get("command", request.get("command", "")) or ""),
        "cwd": str(consumer.get("cwd", request.get("cwd", "")) or ""),
        "env_overrides": mapping(consumer.get("env_overrides") or request.get("env_overrides")),
        "notes": list(dict.fromkeys(notes)),
    }


def finalize_holosoma_adapter_execution(
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
    if execution_path == "local_runtime_binding" and local_runtime_handoff:
        adapter_status = "local_runtime_binding_handed_off"
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


def build_holosoma_adapter_receipt(
    adapter_execution: Mapping[str, Any],
    *,
    artifact_refs: list[str] | None = None,
    realization: Mapping[str, Any] | None = None,
    local_adapter_invocation: Mapping[str, Any] | None = None,
    local_adapter_result: Mapping[str, Any] | None = None,
) -> BackendRuntimeAdapterReceipt:
    payload = mapping(adapter_execution)
    realization_payload = mapping(realization)
    local_invocation_payload = mapping(local_adapter_invocation)
    local_result_payload = mapping(local_adapter_result)
    receipt_payload = {
        "backend": "holosoma",
        "adapter_family": str(payload.get("adapter_family", "") or "holosoma"),
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
        backend="holosoma",
        adapter_family=str(payload.get("adapter_family", "") or "holosoma"),
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
            "consumer_status": str(payload.get("consumer_status", "") or ""),
            "local_runtime_module": str(payload.get("local_runtime_module", "") or ""),
            "local_runtime_available": bool(payload.get("local_runtime_available", False)),
            "missing_preconditions": strings(payload.get("missing_preconditions")),
            "launch_status": str(payload.get("launch_status", "") or ""),
            "returncode": payload.get("returncode"),
            "notes": strings(payload.get("notes")),
            "realization": realization_payload,
            "local_adapter_invocation": local_invocation_payload,
            "local_adapter_result": local_result_payload,
        },
    )


__all__ = [
    "LOCAL_RUNTIME_MODULE",
    "build_holosoma_adapter_receipt",
    "finalize_holosoma_adapter_execution",
    "prepare_holosoma_adapter_execution",
]

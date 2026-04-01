"""Concrete realization over Isaac/Unitree adapter execution mediation."""

from __future__ import annotations

from typing import Any, Mapping

from ..common import mapping, stable_id, strings


LOCAL_BACKEND_NAME = "workcell_isaaclab"
LOCAL_BACKEND_FACTORY = "src.motor_backend.factory:make_motor_backend"
LOCAL_BRIDGE_MODULE = "src.motor_backend.workcell_isaaclab_backend"
EXTERNAL_DELEGATE_RUNNER = "scripts/run_isaac_unitree_executable_adapter.py"


def build_isaac_unitree_adapter_realization(
    *,
    executable_adapter_request: Mapping[str, Any],
    executable_adapter_consumer: Mapping[str, Any],
    adapter_execution: Mapping[str, Any],
    runtime_bundle: Mapping[str, Any],
    launch_spec: Mapping[str, Any],
    binding_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = mapping(executable_adapter_request)
    consumer = mapping(executable_adapter_consumer)
    execution = mapping(adapter_execution)
    bundle = mapping(runtime_bundle)
    spec = mapping(launch_spec)
    binding = mapping(binding_payload)
    execution_path = str(execution.get("execution_path", "") or "")
    adapter_status = str(execution.get("adapter_status", "") or "")
    missing_preconditions = strings(execution.get("missing_preconditions"))
    notes = strings(execution.get("notes"))

    realization_path = "blocked"
    realization_status = "realization_blocked"
    if execution_path == "local_python_bridge":
        realization_path = "local_backend_factory"
        if adapter_status in {"local_bridge_ready", "local_bridge_handed_off"}:
            realization_status = "local_backend_factory_ready"
        else:
            realization_status = "local_backend_factory_missing"
            if "local_python_bridge_module" not in missing_preconditions:
                missing_preconditions.append("local_python_bridge_module")
    elif execution_path == "external_launch":
        realization_path = "external_launch_delegate"
        if adapter_status in {
            "external_launch_ready",
            "external_launch_completed",
            "external_launch_failed",
        }:
            realization_status = "external_launch_delegate_ready"
        else:
            realization_status = "external_launch_delegate_blocked"

    preferred_profile = str(
        request.get("preferred_profile")
        or spec.get("preferred_profile")
        or bundle.get("preferred_profile")
        or ""
    )
    upstream_profile = mapping(spec.get("upstream_profile"))
    payload = {
        "request_id": str(request.get("request_id", "") or ""),
        "consumer_id": str(consumer.get("consumer_id", "") or ""),
        "execution_id": str(execution.get("execution_id", "") or ""),
        "adapter_family": str(request.get("adapter_family", "") or "isaac_unitree"),
        "deployment_mode": str(request.get("deployment_mode", "") or ""),
        "realization_path": realization_path,
        "realization_status": realization_status,
        "preferred_profile": preferred_profile,
    }
    upstream_notes = []
    for item in (
        str(upstream_profile.get("repo", "") or ""),
        str(upstream_profile.get("url", "") or ""),
    ):
        if item:
            upstream_notes.append(item)

    return {
        "version": "backend_executable_adapter_realization_v1",
        "realization_id": stable_id("backend_executable_adapter_realization", payload),
        **payload,
        "adapter_entrypoint": str(request.get("adapter_entrypoint", "") or ""),
        "backend_name": LOCAL_BACKEND_NAME if realization_path == "local_backend_factory" else "",
        "factory_entrypoint": (
            LOCAL_BACKEND_FACTORY if realization_path == "local_backend_factory" else ""
        ),
        "local_bridge_module": LOCAL_BRIDGE_MODULE,
        "delegate_runner": (
            EXTERNAL_DELEGATE_RUNNER
            if realization_path == "external_launch_delegate"
            else ""
        ),
        "command": str(
            consumer.get("command") or request.get("command") or spec.get("command") or ""
        ),
        "cwd": str(consumer.get("cwd") or request.get("cwd") or spec.get("root") or ""),
        "env_overrides": mapping(
            consumer.get("env_overrides")
            or request.get("env_overrides")
            or execution.get("env_overrides")
        ),
        "output_expectations": mapping(request.get("output_expectations")),
        "runtime_bundle_profile": str(bundle.get("preferred_profile", "") or ""),
        "upstream_profile": upstream_profile,
        "binding_executor_entrypoint": str(binding.get("executor_entrypoint", "") or ""),
        "binding_status": str(binding.get("binding_status", "") or ""),
        "missing_preconditions": list(dict.fromkeys(missing_preconditions)),
        "notes": list(dict.fromkeys(notes + upstream_notes)),
    }


__all__ = [
    "EXTERNAL_DELEGATE_RUNNER",
    "LOCAL_BACKEND_FACTORY",
    "LOCAL_BACKEND_NAME",
    "LOCAL_BRIDGE_MODULE",
    "build_isaac_unitree_adapter_realization",
]

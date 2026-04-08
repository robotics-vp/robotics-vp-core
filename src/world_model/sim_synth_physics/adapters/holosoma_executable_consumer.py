"""Consumer selection over Holosoma executable-adapter requests."""

from __future__ import annotations

from typing import Any, Mapping

from ..common import mapping, stable_id, strings


EXTERNAL_CONSUMER_MODES = {
    "sim_eval": "external_runtime_launch",
    "motion_train": "external_motion_launch",
}


def build_holosoma_executable_adapter_consumer(
    executable_adapter_request: Mapping[str, Any],
) -> dict[str, Any]:
    request = mapping(executable_adapter_request)
    deployment_mode = str(request.get("deployment_mode", "") or "sim_eval")
    supports_local_runtime_binding = bool(request.get("supports_local_runtime_binding", False))
    missing_preconditions = strings(request.get("missing_preconditions"))
    command = str(request.get("command", "") or "")
    cwd = str(request.get("cwd", "") or "")

    uses_local_runtime_binding = supports_local_runtime_binding
    consumer_mode = (
        "local_runtime_binding"
        if uses_local_runtime_binding
        else EXTERNAL_CONSUMER_MODES.get(deployment_mode, "external_runtime_launch")
    )
    if not uses_local_runtime_binding:
        if not command:
            missing_preconditions.append("launch_command")
        if not cwd:
            missing_preconditions.append("launch_cwd")

    if missing_preconditions:
        consumer_status = "consumer_blocked"
    elif uses_local_runtime_binding:
        consumer_status = "local_runtime_binding_ready"
    else:
        consumer_status = "external_launch_consumer_ready"

    payload = {
        "adapter_request_id": str(request.get("request_id", "") or ""),
        "preferred_profile": str(request.get("preferred_profile", "") or ""),
        "adapter_entrypoint": str(request.get("adapter_entrypoint", "") or ""),
        "deployment_mode": deployment_mode,
        "consumer_mode": consumer_mode,
        "consumer_status": consumer_status,
        "uses_local_runtime_binding": uses_local_runtime_binding,
        "external_runtime_required": not uses_local_runtime_binding,
        "task_id": str(request.get("task_id", "") or ""),
        "policy_ref": str(request.get("policy_ref", "") or ""),
        "command": command,
        "cwd": cwd,
        "missing_preconditions": list(dict.fromkeys(missing_preconditions)),
    }
    return {
        "version": "backend_executable_adapter_consumer_v1",
        "consumer_id": stable_id("backend_executable_adapter_consumer", payload),
        **payload,
        "env_overrides": mapping(request.get("env_overrides")),
        "required_target_ids": strings(request.get("required_target_ids")),
        "required_asset_ids": strings(request.get("required_asset_ids")),
        "available_asset_ids": strings(request.get("available_asset_ids")),
        "asset_refs": mapping(request.get("asset_refs")),
        "calibration_contracts": strings(request.get("calibration_contracts")),
        "observation_contracts": strings(request.get("observation_contracts")),
        "action_contracts": strings(request.get("action_contracts")),
        "output_expectations": mapping(request.get("output_expectations")),
        "notes": strings(request.get("notes")),
    }


__all__ = ["build_holosoma_executable_adapter_consumer"]

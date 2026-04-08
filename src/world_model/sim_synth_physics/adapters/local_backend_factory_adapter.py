"""Generic local backend-factory invocation over executable-adapter realization."""

from __future__ import annotations

from typing import Any, Mapping

from src.motor_backend.factory import make_motor_backend

from ..common import mapping, stable_id, strings


SUPPORTED_FACTORY_ENTRYPOINT = "src.motor_backend.factory:make_motor_backend"


def build_local_backend_factory_invocation(
    *,
    backend: str,
    executable_adapter_request: Mapping[str, Any],
    executable_adapter_consumer: Mapping[str, Any],
    adapter_execution: Mapping[str, Any],
    adapter_realization: Mapping[str, Any],
    binding_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    request = mapping(executable_adapter_request)
    consumer = mapping(executable_adapter_consumer)
    execution = mapping(adapter_execution)
    realization = mapping(adapter_realization)
    binding = mapping(binding_payload)
    missing_preconditions = strings(realization.get("missing_preconditions")) + strings(
        execution.get("missing_preconditions")
    )
    factory_entrypoint = str(realization.get("factory_entrypoint", "") or "")
    backend_name = str(realization.get("backend_name", "") or backend)
    realization_path = str(realization.get("realization_path", "") or "")
    realization_status = str(realization.get("realization_status", "") or "")

    invocation_status = "local_backend_invocation_blocked"
    if realization_path == "local_backend_factory":
        if (
            realization_status.endswith("_ready")
            and factory_entrypoint == SUPPORTED_FACTORY_ENTRYPOINT
        ):
            invocation_status = "local_backend_invocation_ready"
        elif factory_entrypoint != SUPPORTED_FACTORY_ENTRYPOINT:
            missing_preconditions.append("local_backend_factory_entrypoint")

    payload = {
        "backend": backend,
        "adapter_family": str(request.get("adapter_family", "") or backend),
        "request_id": str(request.get("request_id", "") or ""),
        "consumer_id": str(consumer.get("consumer_id", "") or ""),
        "execution_id": str(execution.get("execution_id", "") or ""),
        "realization_id": str(realization.get("realization_id", "") or ""),
        "backend_name": backend_name,
        "factory_entrypoint": factory_entrypoint,
        "invocation_status": invocation_status,
        "task_id": str(request.get("task_id", "") or ""),
        "policy_ref": str(request.get("policy_ref", "") or ""),
    }
    return {
        "version": "backend_local_factory_invocation_v1",
        "invocation_id": stable_id("backend_local_factory_invocation", payload),
        **payload,
        "deployment_mode": str(request.get("deployment_mode", "") or ""),
        "adapter_entrypoint": str(request.get("adapter_entrypoint", "") or ""),
        "consumer_mode": str(consumer.get("consumer_mode", "") or ""),
        "execution_path": str(execution.get("execution_path", "") or ""),
        "command": str(consumer.get("command") or request.get("command") or ""),
        "cwd": str(consumer.get("cwd") or request.get("cwd") or ""),
        "env_overrides": mapping(
            consumer.get("env_overrides") or request.get("env_overrides")
        ),
        "backend_config": binding,
        "missing_preconditions": list(dict.fromkeys(missing_preconditions)),
        "notes": list(
            dict.fromkeys(
                strings(execution.get("notes"))
                + strings(realization.get("notes"))
                + ["Local backend factory invocation is explicit, not implicit."]
            )
        ),
    }


def materialize_local_backend_factory_invocation(
    invocation: Mapping[str, Any],
    *,
    econ_meter: Any,
    store: Any,
) -> tuple[Any | None, dict[str, Any]]:
    payload = mapping(invocation)
    invocation_status = str(payload.get("invocation_status", "") or "")
    backend_name = str(payload.get("backend_name", "") or "")
    factory_entrypoint = str(payload.get("factory_entrypoint", "") or "")
    backend_config = mapping(payload.get("backend_config"))
    missing_preconditions = strings(payload.get("missing_preconditions"))

    result_status = "local_backend_materialization_blocked"
    error = ""
    backend_instance = None
    if invocation_status == "local_backend_invocation_ready":
        if factory_entrypoint != SUPPORTED_FACTORY_ENTRYPOINT:
            missing_preconditions.append("local_backend_factory_entrypoint")
        else:
            try:
                backend_instance = make_motor_backend(
                    backend_name,
                    econ_meter=econ_meter,
                    store=store,
                    backend_config=backend_config,
                )
                if backend_instance is None:
                    raise RuntimeError(f"{backend_name} backend factory returned None.")
                result_status = "local_backend_materialized"
            except Exception as exc:
                error = str(exc)
                result_status = "local_backend_materialization_failed"

    result_payload = {
        "backend": str(payload.get("backend", "") or ""),
        "invocation_id": str(payload.get("invocation_id", "") or ""),
        "backend_name": backend_name,
        "factory_entrypoint": factory_entrypoint,
        "result_status": result_status,
    }
    result = {
        "version": "backend_local_factory_result_v1",
        "result_id": stable_id("backend_local_factory_result", result_payload),
        **result_payload,
        "task_id": str(payload.get("task_id", "") or ""),
        "policy_ref": str(payload.get("policy_ref", "") or ""),
        "deployment_mode": str(payload.get("deployment_mode", "") or ""),
        "adapter_entrypoint": str(payload.get("adapter_entrypoint", "") or ""),
        "consumer_mode": str(payload.get("consumer_mode", "") or ""),
        "execution_path": str(payload.get("execution_path", "") or ""),
        "missing_preconditions": list(dict.fromkeys(missing_preconditions)),
        "error": error,
        "notes": strings(payload.get("notes")),
    }
    return backend_instance, result


__all__ = [
    "SUPPORTED_FACTORY_ENTRYPOINT",
    "build_local_backend_factory_invocation",
    "materialize_local_backend_factory_invocation",
]

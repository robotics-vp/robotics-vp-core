"""Launch preparation/execution over WM-owned backend runtime bundles."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .common import mapping, stable_id, strings
from .receipts import BackendRuntimeLaunchReceipt

TARGET_ENV_VARS = {
    "isaaclab_root": "ISAACLAB_ROOT",
    "isaacsim_root": "ISAACSIM_ROOT",
    "unitree_sdk2_root": "UNITREE_SDK2_ROOT",
    "unitree_asset_root": "UNITREE_ASSET_ROOT",
    "unitree_sim_isaaclab_root": "UNITREE_SIM_ISAACLAB_ROOT",
    "unitree_rl_gym_root": "UNITREE_RL_GYM_ROOT",
    "humanoidverse_root": "HUMANOIDVERSE_ROOT",
    "xr_teleoperate_root": "XR_TELEOPERATE_ROOT",
    "unitree_model_root": "UNITREE_MODEL_ROOT",
    "unitree_policy_root": "UNITREE_POLICY_ROOT",
    "unitree_sdk2_python_root": "UNITREE_SDK2_PYTHON_ROOT",
    "teleimager_root": "TELEIMAGER_ROOT",
    "unitree_il_lerobot_root": "UNITREE_IL_LEROBOT_ROOT",
    "holosoma_root": "HOLOSOMA_ROOT",
    "holosoma_motion_root": "HOLOSOMA_MOTION_ROOT",
    "holosoma_policy_root": "HOLOSOMA_POLICY_ROOT",
    "retargeting_root": "RETARGETING_ROOT",
}


def _target_env_overrides(runtime_target_contract: Mapping[str, Any]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for row in list(runtime_target_contract.get("targets", []) or []):
        row_mapping = mapping(row)
        target_id = str(row_mapping.get("target_id", "") or "")
        ref = str(row_mapping.get("ref", "") or "")
        env_var = TARGET_ENV_VARS.get(target_id, "")
        if env_var and ref:
            overrides[env_var] = ref
    return overrides


def _cuda_ready() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def prepare_backend_runtime_launch(
    runtime_bundle: Mapping[str, Any],
    launch_spec: Mapping[str, Any],
    *,
    require_policy: bool = True,
) -> dict[str, Any]:
    bundle = mapping(runtime_bundle)
    spec = mapping(launch_spec)
    executable_adapter_request = mapping(
        spec.get("executable_adapter_request") or bundle.get("executable_adapter_request")
    )
    executable_adapter_consumer = mapping(
        spec.get("executable_adapter_consumer") or bundle.get("executable_adapter_consumer")
    )
    runtime_binding = mapping(spec.get("runtime_binding") or bundle.get("runtime_binding"))
    backend = str(bundle.get("backend", spec.get("backend", "")) or "")
    runtime_target_contract = mapping(bundle.get("runtime_target_contract"))
    policy_contract = mapping(bundle.get("policy_contract"))
    missing_preconditions: list[str] = []
    notes: list[str] = []
    if platform.system().lower() != "linux":
        missing_preconditions.append("linux_host")
    if not _cuda_ready():
        missing_preconditions.append("cuda_gpu")
    if not bool(runtime_target_contract.get("runtime_targets_ready", False)):
        missing_preconditions.extend(
            strings(runtime_target_contract.get("missing_required_target_ids"))
        )
        if list(runtime_target_contract.get("unresolved_one_of_groups", []) or []):
            missing_preconditions.append("runtime_profile_root")
    if not str(spec.get("command", "") or ""):
        missing_preconditions.append("launch_command")
    if require_policy and not bool(
        spec.get("policy_ready", policy_contract.get("policy_ready", False))
    ):
        missing_preconditions.append("policy_checkpoint")
    for item in strings(executable_adapter_request.get("missing_preconditions")):
        if item not in missing_preconditions:
            missing_preconditions.append(item)
    for item in strings(executable_adapter_consumer.get("missing_preconditions")):
        if item not in missing_preconditions:
            missing_preconditions.append(item)
    for item in strings(runtime_binding.get("missing_components")):
        if item not in missing_preconditions:
            missing_preconditions.append(item)
    for item in strings(runtime_binding.get("host_preflight_missing_components")):
        if str(item).startswith("asset::"):
            continue
        if item not in missing_preconditions:
            missing_preconditions.append(item)
    host_preflight_status = str(runtime_binding.get("host_preflight_status", "") or "")
    if host_preflight_status and host_preflight_status not in {"preflight_ready", "ready"}:
        notes.append(
            f"Runtime binding host preflight is {host_preflight_status}."
        )
    if backend == "isaac":
        notes.append("Prefer Unitree/IsaacLab-style launch profiles when available.")
    elif backend == "holosoma":
        notes.append("Prefer Holosoma repo plus motion/policy/retargeting roots when available.")
    notes.extend(
        item
        for item in strings(executable_adapter_request.get("notes"))
        if item not in notes
    )
    notes.extend(
        item
        for item in strings(executable_adapter_consumer.get("notes"))
        if item not in notes
    )
    notes.extend(item for item in strings(runtime_binding.get("notes")) if item not in notes)
    env_overrides = _target_env_overrides(runtime_target_contract)
    env_overrides.update(
        {
            str(key): str(value)
            for key, value in mapping(executable_adapter_request.get("env_overrides")).items()
            if str(key) and str(value)
        }
    )
    env_overrides.update(
        {
            str(key): str(value)
            for key, value in mapping(executable_adapter_consumer.get("env_overrides")).items()
            if str(key) and str(value)
        }
    )
    status = "ready_for_launch" if not missing_preconditions else "blocked"
    payload = {
        "backend": backend,
        "preferred_profile": str(spec.get("preferred_profile", "") or ""),
        "status": status,
        "command": str(
            executable_adapter_consumer.get(
                "command",
                executable_adapter_request.get(
                    "command",
                    runtime_binding.get("selected_command", spec.get("command", "")),
                ),
            )
            or ""
        ),
        "cwd": str(
            executable_adapter_consumer.get(
                "cwd",
                executable_adapter_request.get(
                    "cwd",
                    runtime_binding.get("selected_launch_root", spec.get("root", "")),
                ),
            )
            or ""
        ),
        "policy_ref": str(spec.get("policy_ref", "") or ""),
        "env_overrides": env_overrides,
        "missing_preconditions": missing_preconditions,
        "notes": notes,
        "executable_adapter_request": executable_adapter_request,
        "executable_adapter_consumer": executable_adapter_consumer,
        "runtime_binding": runtime_binding,
        "host_preflight_status": host_preflight_status,
    }
    return {
        "launch_id": stable_id("backend_runtime_launch", payload),
        **payload,
    }


def execute_backend_runtime_launch(
    runtime_bundle: Mapping[str, Any],
    launch_spec: Mapping[str, Any],
    *,
    execute: bool = False,
    cwd: str | Path | None = None,
    require_policy: bool = True,
) -> dict[str, Any]:
    plan = prepare_backend_runtime_launch(
        runtime_bundle,
        launch_spec,
        require_policy=require_policy,
    )
    if not execute:
        plan["executed"] = False
        return plan
    if plan["status"] != "ready_for_launch":
        plan["executed"] = False
        return plan
    env = dict(os.environ)
    env.update({str(key): str(value) for key, value in dict(plan["env_overrides"]).items()})
    run_cwd = str(cwd or plan["cwd"] or ".")
    proc = subprocess.run(
        str(plan["command"]),
        shell=True,
        cwd=run_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        **plan,
        "executed": True,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": "launch_completed" if proc.returncode == 0 else "launch_failed",
    }


def build_backend_runtime_launch_receipt(
    runtime_bundle: Mapping[str, Any],
    launch_spec: Mapping[str, Any],
    launch_result: Mapping[str, Any],
    *,
    artifact_refs: list[str] | None = None,
) -> BackendRuntimeLaunchReceipt:
    bundle = mapping(runtime_bundle)
    spec = mapping(launch_spec)
    result = mapping(launch_result)
    backend = str(result.get("backend", bundle.get("backend", spec.get("backend", ""))) or "")
    preferred_profile = str(
        result.get("preferred_profile", spec.get("preferred_profile", bundle.get("preferred_profile", "")))
        or ""
    )
    raw_status = str(result.get("status", "") or "")
    if raw_status == "ready_for_launch":
        launch_status = "launch_prepared"
    elif raw_status == "blocked":
        launch_status = "launch_blocked"
    else:
        launch_status = raw_status or "launch_unknown"
    payload = {
        "backend": backend,
        "preferred_profile": preferred_profile,
        "launch_status": launch_status,
        "executed": bool(result.get("executed", False)),
        "command": str(result.get("command", spec.get("command", "")) or ""),
        "cwd": str(result.get("cwd", spec.get("root", "")) or ""),
        "policy_ref": str(result.get("policy_ref", spec.get("policy_ref", "")) or ""),
        "runtime_targets_ready": bool(
            bundle.get("runtime_target_contract", {}).get("runtime_targets_ready", False)
        ),
    }
    return BackendRuntimeLaunchReceipt(
        receipt_id=stable_id("backend_runtime_launch_receipt", payload),
        backend=backend,
        launch_profile=preferred_profile,
        launch_status=launch_status,
        executed=bool(result.get("executed", False)),
        command=str(result.get("command", spec.get("command", "")) or ""),
        cwd=str(result.get("cwd", spec.get("root", "")) or ""),
        artifact_refs=strings(artifact_refs or []),
        metadata={
            "runtime_bundle": bundle,
            "launch_spec": spec,
            "launch_result": result,
            "missing_preconditions": strings(result.get("missing_preconditions")),
            "notes": strings(result.get("notes")),
            "env_overrides": mapping(result.get("env_overrides")),
            "executable_adapter_request": mapping(result.get("executable_adapter_request")),
            "executable_adapter_consumer": mapping(result.get("executable_adapter_consumer")),
            "runtime_binding": mapping(result.get("runtime_binding")),
            "host_preflight_status": str(result.get("host_preflight_status", "") or ""),
            "returncode": result.get("returncode"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        },
    )


def load_runtime_artifacts(
    *,
    runtime_bundle_path: str | Path,
    launch_spec_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_bundle = mapping(
        json.loads(Path(runtime_bundle_path).read_text(encoding="utf-8"))
    )
    launch_spec = mapping(
        json.loads(Path(launch_spec_path).read_text(encoding="utf-8"))
    )
    return runtime_bundle, launch_spec


__all__ = [
    "build_backend_runtime_launch_receipt",
    "execute_backend_runtime_launch",
    "load_runtime_artifacts",
    "prepare_backend_runtime_launch",
]

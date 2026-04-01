"""Executable-adapter requests for Isaac/Unitree Phase-1 runtime launches."""

from __future__ import annotations

from typing import Any, Mapping

from ..common import mapping, stable_id, strings

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
}

PROFILE_TO_MODE = {
    "unitree_sim_isaaclab": "sim_eval",
    "unitree_rl_gym": "sim_eval",
    "humanoidverse": "sim_eval",
    "isaaclab_core": "sim_eval",
    "xr_teleoperate": "teleop_bridge",
    "unitree_lerobot": "lerobot_eval",
}

PROFILE_TO_ENTRYPOINT = {
    "unitree_sim_isaaclab": "isaaclab_unitree_sim",
    "unitree_rl_gym": "unitree_rl_gym_deploy",
    "humanoidverse": "humanoidverse_eval",
    "isaaclab_core": "isaaclab_eval",
    "xr_teleoperate": "unitree_xr_teleop_bridge",
    "unitree_lerobot": "unitree_lerobot_eval",
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


def _mode_contract(
    deployment_contract: Mapping[str, Any],
    deployment_mode: str,
) -> dict[str, Any]:
    for row in list(deployment_contract.get("deployment_modes", []) or []):
        row_mapping = mapping(row)
        if str(row_mapping.get("mode_id", "") or "") == deployment_mode:
            return row_mapping
    return {}


def _asset_rows(normalized_robot_asset_manifest: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    refs: dict[str, str] = {}
    available: list[str] = []
    for asset_id, row in mapping(normalized_robot_asset_manifest).items():
        row_mapping = mapping(row if isinstance(row, Mapping) else {})
        if bool(row_mapping.get("present", False)):
            available.append(str(asset_id))
            ref = str(row_mapping.get("value", "") or "")
            if ref:
                refs[str(asset_id)] = ref
    return refs, sorted(available)


def _output_expectations(output_contract: Mapping[str, Any]) -> dict[str, Any]:
    source_specs = list(output_contract.get("source_specs", []) or [])
    artifact_kinds = sorted(
        {
            str(mapping(spec).get("artifact_kind", "") or "")
            for spec in source_specs
            if str(mapping(spec).get("artifact_kind", "") or "")
        }
    )
    return {
        "profile_id": str(output_contract.get("profile_id", "") or ""),
        "artifact_kinds": artifact_kinds,
        "source_ids": [
            str(mapping(spec).get("source_id", "") or "")
            for spec in source_specs
            if str(mapping(spec).get("source_id", "") or "")
        ],
    }


def build_isaac_unitree_executable_adapter_request(
    *,
    task_id: str,
    policy_ref: str,
    preferred_profile: str,
    launch_spec: Mapping[str, Any],
    runtime_target_contract: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    normalized_robot_asset_manifest: Mapping[str, Any],
    robot_contract_context: Mapping[str, Any] | None = None,
    output_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    deployment_mode = PROFILE_TO_MODE.get(preferred_profile, "sim_eval")
    mode_contract = _mode_contract(deployment_contract, deployment_mode)
    env_overrides = _target_env_overrides(runtime_target_contract)
    robot_variant = str(deployment_contract.get("robot_variant", "") or "")
    placement_class = str(deployment_contract.get("placement_class", "") or "")
    asset_refs, available_asset_ids = _asset_rows(normalized_robot_asset_manifest)
    robot_context = mapping(robot_contract_context)
    missing_preconditions = strings(mode_contract.get("missing_preconditions"))
    policy_required = deployment_mode != "teleop_bridge"

    env_overrides.update(
        {
            "UNITREE_ROBOT_VARIANT": robot_variant,
            "UNITREE_DEPLOYMENT_MODE": deployment_mode,
            "UNITREE_TASK_ID": task_id,
            "UNITREE_POLICY_REF": policy_ref,
            "UNITREE_PREFERRED_PROFILE": preferred_profile,
            "UNITREE_PLACEMENT_CLASS": placement_class,
            "UNITREE_TELEOP_ENABLED": "1" if deployment_mode == "teleop_bridge" else "0",
            "UNITREE_LEROBOT_EVAL_ENABLED": "1" if deployment_mode == "lerobot_eval" else "0",
            "UNITREE_PHYSICAL_DEPLOY_READY": (
                "1" if bool(deployment_contract.get("physical_deploy_ready", False)) else "0"
            ),
        }
    )
    for asset_id, ref in asset_refs.items():
        env_overrides[f"UNITREE_ASSET_{asset_id.upper()}"] = ref

    payload = {
        "backend": "isaac",
        "adapter_family": "isaac_unitree",
        "preferred_profile": preferred_profile,
        "adapter_entrypoint": PROFILE_TO_ENTRYPOINT.get(preferred_profile, "isaac_unitree_runtime"),
        "deployment_mode": deployment_mode,
        "robot_variant": robot_variant,
        "placement_class": placement_class,
        "task_id": task_id,
        "policy_ref": policy_ref,
        "policy_required": policy_required,
        "cwd": str(mapping(launch_spec).get("root", "") or ""),
        "command": str(mapping(launch_spec).get("command", "") or ""),
        "required_target_ids": strings(mode_contract.get("required_target_ids")),
        "required_asset_ids": strings(mode_contract.get("required_asset_ids")),
        "available_asset_ids": available_asset_ids,
        "asset_refs": asset_refs,
        "calibration_contracts": strings(robot_context.get("calibration_contracts")),
        "observation_contracts": strings(robot_context.get("observation_contracts")),
        "action_contracts": strings(robot_context.get("action_contracts")),
        "robot_asset_contract_id": str(robot_context.get("robot_asset_contract_id", "") or ""),
        "output_expectations": _output_expectations(mapping(output_contract)),
        "runtime_target_ids": strings(runtime_target_contract.get("ready_target_ids")),
        "supports_local_python_bridge": bool(
            runtime_target_contract.get("python_bridge_available", False)
        ),
        "missing_preconditions": missing_preconditions,
        "env_overrides": env_overrides,
        "notes": [
            "This request is the WM-owned executable adapter surface for Isaac/Unitree runtime launch.",
            "It stays real-or-unavailable: missing roots/assets/calibration remain explicit preconditions.",
        ],
    }
    return {
        "version": "backend_executable_adapter_request_v1",
        "request_id": stable_id("backend_executable_adapter_request", payload),
        **payload,
    }


__all__ = ["build_isaac_unitree_executable_adapter_request"]

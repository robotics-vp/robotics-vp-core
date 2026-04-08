"""Executable-adapter requests for Holosoma Phase-1 runtime launches."""

from __future__ import annotations

from typing import Any, Mapping

from ..common import mapping, stable_id, strings


TARGET_ENV_VARS = {
    "holosoma_root": "HOLOSOMA_ROOT",
    "holosoma_motion_root": "HOLOSOMA_MOTION_ROOT",
    "holosoma_policy_root": "HOLOSOMA_POLICY_ROOT",
    "retargeting_root": "RETARGETING_ROOT",
}

PROFILE_TO_MODE = {
    "holosoma_repo": "sim_eval",
    "holosoma_policy_bank": "sim_eval",
    "retargeting_bundle": "sim_eval",
    "holosoma_motion_bank": "motion_train",
}

PROFILE_TO_ENTRYPOINT = {
    "holosoma_repo": "holosoma_repo_eval",
    "holosoma_policy_bank": "holosoma_policy_eval",
    "retargeting_bundle": "holosoma_retargeting_eval",
    "holosoma_motion_bank": "holosoma_motion_train",
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


def build_holosoma_executable_adapter_request(
    *,
    task_id: str,
    policy_ref: str,
    preferred_profile: str,
    launch_spec: Mapping[str, Any],
    runtime_target_contract: Mapping[str, Any],
    policy_contract: Mapping[str, Any],
    runtime_binding: Mapping[str, Any] | None = None,
    normalized_robot_asset_manifest: Mapping[str, Any],
    robot_contract_context: Mapping[str, Any] | None = None,
    output_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding = mapping(runtime_binding)
    deployment_mode = PROFILE_TO_MODE.get(preferred_profile, "sim_eval")
    policy_required = deployment_mode != "motion_train"
    asset_refs, available_asset_ids = _asset_rows(normalized_robot_asset_manifest)
    robot_context = mapping(robot_contract_context)
    selected_policy_ref = str(binding.get("selected_policy_ref", "") or policy_ref)
    selected_command = str(binding.get("selected_command", "") or mapping(launch_spec).get("command", "") or "")
    selected_launch_root = str(binding.get("selected_launch_root", "") or mapping(launch_spec).get("root", "") or "")
    selected_motion_sources = strings(binding.get("selected_motion_sources"))
    selected_retargeting_root = str(binding.get("selected_retargeting_root", "") or "")
    selected_deploy_config = str(binding.get("selected_deploy_config", "") or "")
    selected_runtime_report = str(binding.get("selected_runtime_report", "") or "")
    missing_preconditions: list[str] = strings(binding.get("missing_components"))
    if policy_required and not (selected_policy_ref or bool(policy_contract.get("policy_ready", False))):
        missing_preconditions.append("policy_checkpoint")
    env_overrides = _target_env_overrides(runtime_target_contract)
    env_overrides.update(
        {
            "HOLOSOMA_TASK_ID": task_id,
            "HOLOSOMA_POLICY_REF": selected_policy_ref,
            "HOLOSOMA_DEPLOYMENT_MODE": deployment_mode,
            "HOLOSOMA_PREFERRED_PROFILE": preferred_profile,
            "HOLOSOMA_RUNTIME_BINDING_STATUS": str(binding.get("binding_status", "") or ""),
            "HOLOSOMA_HOST_PREFLIGHT_STATUS": str(
                binding.get("host_preflight_status", "") or ""
            ),
            "HOLOSOMA_MOTION_TRAIN_ENABLED": "1" if deployment_mode == "motion_train" else "0",
        }
    )
    if selected_retargeting_root:
        env_overrides["HOLOSOMA_RETARGETING_REF"] = selected_retargeting_root
    if selected_deploy_config:
        env_overrides["HOLOSOMA_DEPLOY_CONFIG_REF"] = selected_deploy_config
    if selected_runtime_report:
        env_overrides["HOLOSOMA_RUNTIME_REPORT_REF"] = selected_runtime_report
    for asset_id, ref in asset_refs.items():
        env_overrides[f"HOLOSOMA_ASSET_{asset_id.upper()}"] = ref

    payload = {
        "backend": "holosoma",
        "adapter_family": "holosoma",
        "preferred_profile": preferred_profile,
        "adapter_entrypoint": PROFILE_TO_ENTRYPOINT.get(preferred_profile, "holosoma_runtime"),
        "deployment_mode": deployment_mode,
        "task_id": task_id,
        "policy_ref": selected_policy_ref,
        "policy_required": policy_required,
        "deploy_config_ref": selected_deploy_config,
        "runtime_report_ref": selected_runtime_report,
        "cwd": selected_launch_root,
        "command": selected_command,
        "required_target_ids": strings(runtime_target_contract.get("required_target_ids")),
        "required_asset_ids": [],
        "available_asset_ids": available_asset_ids,
        "asset_refs": asset_refs,
        "calibration_contracts": strings(robot_context.get("calibration_contracts")),
        "observation_contracts": strings(robot_context.get("observation_contracts")),
        "action_contracts": strings(robot_context.get("action_contracts")),
        "robot_asset_contract_id": str(robot_context.get("robot_asset_contract_id", "") or ""),
        "output_expectations": _output_expectations(mapping(output_contract)),
        "runtime_target_ids": strings(runtime_target_contract.get("ready_target_ids")),
        "supports_local_runtime_binding": bool(
            runtime_target_contract.get("python_bridge_available", False)
        ),
        "runtime_binding_id": str(binding.get("binding_id", "") or ""),
        "runtime_binding_status": str(binding.get("binding_status", "") or ""),
        "runtime_binding_host_preflight_status": str(
            binding.get("host_preflight_status", "") or ""
        ),
        "runtime_binding": binding,
        "selected_motion_sources": selected_motion_sources,
        "selected_retargeting_root": selected_retargeting_root,
        "missing_preconditions": list(dict.fromkeys(missing_preconditions)),
        "env_overrides": env_overrides,
        "notes": [
            "This request is the WM-owned executable adapter surface for Holosoma runtime launch.",
            "Holosoma stays real-or-unavailable: repo, motion, policy, and retargeting roots remain explicit.",
        ],
    }
    return {
        "version": "backend_executable_adapter_request_v1",
        "request_id": stable_id("backend_executable_adapter_request", payload),
        **payload,
    }


__all__ = ["build_holosoma_executable_adapter_request"]

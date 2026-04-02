"""Unitree-aware Isaac deployment contracts for the sim/synth/physics WM."""

from __future__ import annotations

from typing import Any, Mapping

from ..common import mapping, strings


SIM_EVAL_PROFILES = {
    "unitree_sim_isaaclab",
    "unitree_rl_gym",
    "humanoidverse",
    "isaaclab_core",
}


def _robot_variant(embodiment_context: Mapping[str, Any]) -> str:
    embodiment = mapping(embodiment_context)
    for key in ("robot_variant", "unitree_robot_variant", "primary_embodiment", "robot_family"):
        candidate = str(embodiment.get(key, "") or "").strip().lower()
        if not candidate:
            continue
        if "g1" in candidate:
            return "unitree_g1"
        if "r1" in candidate:
            return "unitree_r1"
    for candidate in strings(
        embodiment.get("active_embodiments")
        or embodiment.get("target_embodiments")
        or embodiment.get("robot_families")
    ):
        lowered = str(candidate).lower()
        if "g1" in lowered:
            return "unitree_g1"
        if "r1" in lowered:
            return "unitree_r1"
    return "unitree_g1_r1_class"


def _asset_present(normalized_asset_manifest: Mapping[str, Any], asset_id: str) -> bool:
    return bool(mapping(normalized_asset_manifest.get(asset_id)).get("present", False))


def _missing_assets(
    normalized_asset_manifest: Mapping[str, Any],
    asset_ids: list[str],
) -> list[str]:
    return [asset_id for asset_id in asset_ids if not _asset_present(normalized_asset_manifest, asset_id)]


def _usable_profiles(runtime_layout_contract: Mapping[str, Any]) -> list[str]:
    usable: list[str] = []
    for row in list(runtime_layout_contract.get("profiles", []) or []):
        row_mapping = mapping(row)
        profile_id = str(row_mapping.get("profile_id", "") or "")
        if not profile_id:
            continue
        if not bool(row_mapping.get("root_exists", False)):
            continue
        if str(row_mapping.get("install_preflight_status", "") or "") == "install_blocked":
            continue
        usable.append(profile_id)
    return usable


def _verified_targets(runtime_target_contract: Mapping[str, Any]) -> set[str]:
    verified = strings(runtime_target_contract.get("verified_target_ids"))
    if verified:
        return set(verified)
    return set(strings(runtime_target_contract.get("ready_target_ids")))


def _preferred_profile_from_context(
    embodiment_context: Mapping[str, Any],
    ready_profiles: list[str],
) -> str:
    embodiment = mapping(embodiment_context)
    explicit_profile_keys = [
        ("unitree_sim_isaaclab", ("unitree_sim_isaaclab_root",)),
        ("unitree_rl_gym", ("unitree_rl_gym_root", "unitree_runtime_repo_root")),
        ("unitree_lerobot", ("unitree_lerobot_root", "unitree_il_lerobot_root")),
        ("humanoidverse", ("humanoidverse_root",)),
        ("isaaclab_core", ("isaaclab_root", "isaac_lab_root", "isaac_repo_root")),
        ("xr_teleoperate", ("xr_teleoperate_root",)),
        ("unitree_model_assets", ("unitree_model_root", "unitree_asset_root", "robot_asset_root")),
    ]
    for profile_id, keys in explicit_profile_keys:
        if profile_id not in ready_profiles:
            continue
        for key in keys:
            if str(embodiment.get(key, "") or "").strip():
                return profile_id
    return ""


def _mode_contract(
    *,
    mode_id: str,
    label: str,
    profile_candidates: list[str],
    required_target_ids: list[str],
    policy_required: bool,
    required_asset_ids: list[str],
    ready_profiles: list[str],
    ready_targets: set[str],
    policy_ready: bool,
    normalized_asset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    missing_preconditions: list[str] = []
    if not any(profile in ready_profiles for profile in profile_candidates):
        missing_preconditions.append("runtime_profile")
    missing_preconditions.extend(
        target_id for target_id in required_target_ids if target_id not in ready_targets
    )
    if policy_required and not policy_ready:
        missing_preconditions.append("policy_checkpoint")
    missing_preconditions.extend(_missing_assets(normalized_asset_manifest, required_asset_ids))
    return {
        "mode_id": mode_id,
        "label": label,
        "profile_candidates": list(profile_candidates),
        "required_target_ids": list(required_target_ids),
        "required_asset_ids": list(required_asset_ids),
        "ready": not missing_preconditions,
        "missing_preconditions": missing_preconditions,
    }


def build_isaac_unitree_deployment_contract(
    *,
    embodiment_context: Mapping[str, Any],
    runtime_target_contract: Mapping[str, Any],
    runtime_layout_contract: Mapping[str, Any],
    policy_contract: Mapping[str, Any],
    normalized_asset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    ready_profiles = _usable_profiles(runtime_layout_contract)
    ready_targets = _verified_targets(runtime_target_contract)
    policy_ready = bool(policy_contract.get("policy_ready", False))
    checkpoint_candidates = strings(policy_contract.get("checkpoint_candidates"))
    deploy_config_candidates = strings(policy_contract.get("deploy_config_candidates"))
    deployment_modes = [
        _mode_contract(
            mode_id="sim_eval",
            label="Isaac/Unitree sim evaluation",
            profile_candidates=["unitree_sim_isaaclab", "unitree_rl_gym", "humanoidverse", "isaaclab_core"],
            required_target_ids=["unitree_sdk2_root", "unitree_asset_root"],
            policy_required=True,
            required_asset_ids=[
                "unitree_robot_description",
                "whole_body_joint_map",
                "actuator_latency_profile",
                "joint_limit_profile",
                "safety_watchdog_profile",
            ],
            ready_profiles=ready_profiles,
            ready_targets=ready_targets,
            policy_ready=policy_ready,
            normalized_asset_manifest=normalized_asset_manifest,
        ),
        _mode_contract(
            mode_id="teleop_bridge",
            label="XR teleoperation bridge",
            profile_candidates=["xr_teleoperate"],
            required_target_ids=[
                "unitree_sdk2_root",
                "unitree_sdk2_python_root",
                "teleimager_root",
                "xr_teleoperate_root",
            ],
            policy_required=False,
            required_asset_ids=[
                "unitree_robot_description",
                "camera_extrinsics",
                "imu_extrinsics",
                "safety_watchdog_profile",
            ],
            ready_profiles=ready_profiles,
            ready_targets=ready_targets,
            policy_ready=policy_ready,
            normalized_asset_manifest=normalized_asset_manifest,
        ),
        _mode_contract(
            mode_id="lerobot_eval",
            label="LeRobot evaluation and data replay",
            profile_candidates=["unitree_lerobot"],
            required_target_ids=[
                "unitree_sdk2_root",
                "unitree_il_lerobot_root",
                "unitree_asset_root",
            ],
            policy_required=True,
            required_asset_ids=[
                "unitree_robot_description",
                "whole_body_joint_map",
                "camera_extrinsics",
                "imu_extrinsics",
            ],
            ready_profiles=ready_profiles,
            ready_targets=ready_targets,
            policy_ready=policy_ready,
            normalized_asset_manifest=normalized_asset_manifest,
        ),
        _mode_contract(
            mode_id="physical_deploy",
            label="Physical Unitree deployment",
            profile_candidates=["xr_teleoperate", "unitree_lerobot", "unitree_rl_gym"],
            required_target_ids=[
                "unitree_sdk2_root",
                "unitree_sdk2_python_root",
                "unitree_asset_root",
            ],
            policy_required=True,
            required_asset_ids=[
                "unitree_robot_description",
                "whole_body_joint_map",
                "camera_extrinsics",
                "imu_extrinsics",
                "force_torque_calibration",
                "actuator_latency_profile",
                "joint_limit_profile",
                "safety_watchdog_profile",
            ],
            ready_profiles=ready_profiles,
            ready_targets=ready_targets,
            policy_ready=policy_ready,
            normalized_asset_manifest=normalized_asset_manifest,
        ),
    ]
    ready_modes = [row["mode_id"] for row in deployment_modes if bool(row.get("ready", False))]
    preferred_profile_order = [
        "unitree_sim_isaaclab",
        "unitree_rl_gym",
        "unitree_lerobot",
        "humanoidverse",
        "isaaclab_core",
        "xr_teleoperate",
        "unitree_model_assets",
    ]
    preferred_profile = _preferred_profile_from_context(
        embodiment_context,
        ready_profiles,
    ) or next(
        (profile_id for profile_id in preferred_profile_order if profile_id in ready_profiles),
        (ready_profiles[0] if ready_profiles else ""),
    )
    return {
        "version": "isaac_unitree_deployment_contract_v1",
        "robot_variant": _robot_variant(embodiment_context),
        "placement_class": (
            "unitree_onboard_plus_companion"
            if "unitree_sdk2_root" in ready_targets
            else "companion_gpu_shadow"
        ),
        "ready_profiles": ready_profiles,
        "verified_target_ids": sorted(ready_targets),
        "runtime_target_preflight_status": str(
            runtime_target_contract.get("runtime_target_preflight_status", "") or ""
        ),
        "preferred_profile_order": preferred_profile_order,
        "preferred_profile": preferred_profile,
        "policy_ready": policy_ready,
        "checkpoint_candidates": checkpoint_candidates,
        "deploy_config_candidates": deploy_config_candidates,
        "ready_modes": ready_modes,
        "sim_launch_ready": "sim_eval" in ready_modes,
        "teleop_launch_ready": "teleop_bridge" in ready_modes,
        "lerobot_eval_ready": "lerobot_eval" in ready_modes,
        "physical_deploy_ready": "physical_deploy" in ready_modes,
        "deployment_modes": deployment_modes,
    }


__all__ = ["build_isaac_unitree_deployment_contract"]

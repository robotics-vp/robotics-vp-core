"""Canonical humanoid asset-manifest helpers for sim/synth/physics."""

from __future__ import annotations

from typing import Any, Mapping

from .common import mapping


UNITREE_REQUIRED_ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "unitree_robot_description": (
        "unitree_robot_description",
        "unitree_urdf",
        "unitree_usd",
        "robot_description",
        "robot_urdf",
        "robot_usd",
    ),
    "whole_body_joint_map": (
        "whole_body_joint_map",
        "joint_mapping_contract",
        "joint_map",
        "joint_map_path",
    ),
    "camera_extrinsics": (
        "camera_extrinsics",
        "sensor_extrinsics",
        "rgb_camera_extrinsics",
    ),
    "imu_extrinsics": (
        "imu_extrinsics",
        "sensor_extrinsics",
    ),
    "force_torque_calibration": (
        "force_torque_calibration",
        "ft_sensor_calibration",
        "force_torque_extrinsics",
    ),
    "actuator_latency_profile": (
        "actuator_latency_profile",
        "actuator_profile",
        "latency_profile",
        "control_latency_profile",
    ),
    "joint_limit_profile": (
        "joint_limit_profile",
        "joint_limits",
        "joint_limit_config",
        "safety_limits",
    ),
    "safety_watchdog_profile": (
        "safety_watchdog_profile",
        "safety_profile",
        "watchdog_profile",
        "e_stop_profile",
    ),
}

UNITREE_RECOMMENDED_ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "self_collision_profile": ("self_collision_profile", "collision_profile"),
    "teleop_recovery_contract": ("teleop_recovery_contract", "operator_override_contract"),
    "support_phase_contract": ("support_phase_contract", "contact_schedule_profile"),
    "control_frequency_profile": ("control_frequency_profile", "servo_profile"),
}


def extract_robot_asset_manifest(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = mapping(embodiment_context)
    return mapping(
        payload.get("robot_asset_manifest")
        or payload.get("asset_manifest")
        or payload.get("robot_assets")
    )


def normalize_robot_asset_manifest(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    manifest = extract_robot_asset_manifest(embodiment_context)
    normalized: dict[str, dict[str, Any]] = {}
    for canonical, aliases in {
        **UNITREE_REQUIRED_ASSET_ALIASES,
        **UNITREE_RECOMMENDED_ASSET_ALIASES,
    }.items():
        matched_aliases = [
            alias
            for alias in aliases
            if alias in manifest and manifest.get(alias) not in (None, "", False, 0, [], {})
        ]
        normalized[canonical] = {
            "present": bool(matched_aliases),
            "matched_aliases": matched_aliases,
            "value": (
                None
                if not matched_aliases
                else manifest.get(matched_aliases[0])
            ),
        }
    passthrough = {
        str(key): value
        for key, value in manifest.items()
        if all(key not in aliases for aliases in {
            **UNITREE_REQUIRED_ASSET_ALIASES,
            **UNITREE_RECOMMENDED_ASSET_ALIASES,
        }.values())
        and value not in (None, "", False, 0, [], {})
    }
    if passthrough:
        normalized["additional_assets"] = {
            "present": True,
            "matched_aliases": sorted(passthrough.keys()),
            "value": passthrough,
        }
    return normalized


def required_assets_for_hardware_class(target_hardware_class: str) -> list[str]:
    if str(target_hardware_class) == "unitree_g1_r1_class":
        return list(UNITREE_REQUIRED_ASSET_ALIASES.keys())
    return ["robot_description", "joint_mapping_contract"]


def recommended_assets_for_hardware_class(target_hardware_class: str) -> list[str]:
    if str(target_hardware_class) == "unitree_g1_r1_class":
        return list(UNITREE_RECOMMENDED_ASSET_ALIASES.keys())
    return []


def available_assets_for_hardware_class(
    target_hardware_class: str,
    embodiment_context: Mapping[str, Any] | None,
) -> list[str]:
    normalized = normalize_robot_asset_manifest(embodiment_context)
    relevant_assets = {
        *required_assets_for_hardware_class(target_hardware_class),
        *recommended_assets_for_hardware_class(target_hardware_class),
    }
    return sorted(
        asset_name
        for asset_name in relevant_assets
        if bool(mapping(normalized.get(asset_name)).get("present", False))
    )


def missing_assets_for_hardware_class(
    target_hardware_class: str,
    embodiment_context: Mapping[str, Any] | None,
) -> list[str]:
    available = set(available_assets_for_hardware_class(target_hardware_class, embodiment_context))
    return [
        asset_name
        for asset_name in required_assets_for_hardware_class(target_hardware_class)
        if asset_name not in available
    ]


__all__ = [
    "available_assets_for_hardware_class",
    "extract_robot_asset_manifest",
    "missing_assets_for_hardware_class",
    "normalize_robot_asset_manifest",
    "recommended_assets_for_hardware_class",
    "required_assets_for_hardware_class",
]

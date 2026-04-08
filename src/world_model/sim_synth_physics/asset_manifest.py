"""Canonical humanoid asset-manifest helpers for sim/synth/physics."""

from __future__ import annotations

from pathlib import Path
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


def _asset_verification(value: Any) -> dict[str, Any]:
    if value in (None, "", False, 0, [], {}):
        return {
            "value_is_path": False,
            "local_path_exists": False,
            "verification_status": "missing",
        }
    text = str(value).strip()
    is_path_like = bool(text) and not text.startswith(("http://", "https://", "s3://"))
    local_path_exists = bool(is_path_like and Path(text).exists())
    if not is_path_like:
        status = "declared_non_path"
    elif local_path_exists:
        status = "declared_local_exists"
    else:
        status = "declared_local_missing"
    return {
        "value_is_path": bool(is_path_like),
        "local_path_exists": bool(local_path_exists),
        "verification_status": status,
    }


def _targets_by_id(runtime_target_contract: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for row in list(mapping(runtime_target_contract).get("targets") or []):
        row_mapping = mapping(row)
        target_id = str(row_mapping.get("target_id", "") or "")
        if target_id:
            payload[target_id] = row_mapping
    return payload


def _root_candidates(
    embodiment_context: Mapping[str, Any] | None,
    runtime_target_contract: Mapping[str, Any] | None,
    *,
    context_keys: tuple[str, ...] = (),
    target_ids: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    embodiment = mapping(embodiment_context)
    target_rows = _targets_by_id(runtime_target_contract)
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key in context_keys:
        ref = str(embodiment.get(key, "") or "").strip()
        if ref and ref not in seen:
            candidates.append((f"embodiment_context.{key}", ref))
            seen.add(ref)
    for target_id in target_ids:
        row = mapping(target_rows.get(target_id))
        ref = str(row.get("ref", "") or "").strip()
        if ref and ref not in seen:
            candidates.append((f"runtime_target.{target_id}", ref))
            seen.add(ref)
    return candidates


def _candidate_source(root_source: str, root: Path, candidate: Path) -> str:
    try:
        suffix = candidate.resolve().relative_to(root.resolve())
        return f"{root_source}:{suffix}"
    except Exception:
        return f"{root_source}:{candidate.name}"


def _file_satisfies_markers(path: Path, markers: tuple[str, ...]) -> bool:
    if not markers:
        return True
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return any(marker in text for marker in markers)


def _select_candidate_file(
    *,
    root_candidates: list[tuple[str, str]],
    relative_candidates: tuple[str, ...] = (),
    glob_patterns: tuple[str, ...] = (),
    content_markers: tuple[str, ...] = (),
) -> tuple[str, str]:
    for root_source, root_ref in root_candidates:
        root = Path(root_ref)
        if not root.exists():
            continue
        seen_candidates: set[str] = set()
        candidates: list[Path] = []
        for rel_path in relative_candidates:
            candidate = (root / rel_path).resolve()
            key = str(candidate)
            if key not in seen_candidates:
                candidates.append(candidate)
                seen_candidates.add(key)
        for pattern in glob_patterns:
            for candidate in sorted(root.rglob(pattern)):
                if not candidate.is_file():
                    continue
                key = str(candidate.resolve())
                if key in seen_candidates:
                    continue
                candidates.append(candidate.resolve())
                seen_candidates.add(key)
        for candidate in candidates:
            if not candidate.exists() or not candidate.is_file():
                continue
            if not _file_satisfies_markers(candidate, content_markers):
                continue
            return str(candidate), _candidate_source(root_source, root, candidate)
    return "", ""


def _derived_asset_row(
    *,
    value: str,
    source: str,
    supporting_refs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "present": bool(value),
        "matched_aliases": [],
        "value": value or None,
        **_asset_verification(value),
        "derivation_kind": "local_runtime_repo_evidence",
        "derivation_source": str(source or ""),
        "supporting_refs": list(supporting_refs or ([] if not value else [value])),
    }


def _derive_unitree_assets(
    embodiment_context: Mapping[str, Any] | None,
    runtime_target_contract: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    asset_roots = _root_candidates(
        embodiment_context,
        runtime_target_contract,
        context_keys=("unitree_asset_root", "unitree_model_root", "robot_asset_root"),
        target_ids=("unitree_asset_root", "unitree_model_root"),
    )
    unitree_rl_roots = _root_candidates(
        embodiment_context,
        runtime_target_contract,
        context_keys=("unitree_rl_gym_root", "unitree_runtime_repo_root"),
        target_ids=("unitree_rl_gym_root",),
    )
    humanoidverse_roots = _root_candidates(
        embodiment_context,
        runtime_target_contract,
        context_keys=("humanoidverse_root",),
        target_ids=("humanoidverse_root",),
    )
    unitree_sim_roots = _root_candidates(
        embodiment_context,
        runtime_target_contract,
        context_keys=("unitree_sim_isaaclab_root",),
        target_ids=("unitree_sim_isaaclab_root",),
    )
    xr_roots = _root_candidates(
        embodiment_context,
        runtime_target_contract,
        context_keys=("xr_teleoperate_root",),
        target_ids=("xr_teleoperate_root",),
    )
    lerobot_roots = _root_candidates(
        embodiment_context,
        runtime_target_contract,
        context_keys=("unitree_lerobot_root", "unitree_il_lerobot_root"),
        target_ids=("unitree_il_lerobot_root",),
    )

    robot_description_ref, robot_description_source = _select_candidate_file(
        root_candidates=asset_roots,
        relative_candidates=(
            "G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd",
            "G1/29dof/usd/g1_29dof_rev_1_0/configuration/g1_29dof_rev_1_0_base.usd",
        ),
        glob_patterns=("**/g1_29dof*.usd", "**/g1_29dof*.urdf", "**/g1_body29*.urdf"),
    )
    if not robot_description_ref:
        robot_description_ref, robot_description_source = _select_candidate_file(
            root_candidates=unitree_rl_roots,
            relative_candidates=(
                "resources/robots/g1_description/g1_29dof_rev_1_0.urdf",
                "resources/robots/g1_description/g1_29dof.urdf",
            ),
            glob_patterns=("**/g1_29dof*.urdf",),
        )
    if not robot_description_ref:
        robot_description_ref, robot_description_source = _select_candidate_file(
            root_candidates=humanoidverse_roots,
            relative_candidates=(
                "humanoidverse/data/robots/g1/g1_29dof.usd",
                "humanoidverse/data/robots/g1/g1_29dof.urdf",
            ),
            glob_patterns=("**/g1_29dof.usd", "**/g1_29dof.urdf"),
        )
    if not robot_description_ref:
        robot_description_ref, robot_description_source = _select_candidate_file(
            root_candidates=xr_roots,
            relative_candidates=("assets/g1/g1_body29_hand14.urdf",),
            glob_patterns=("**/g1_body29*.urdf",),
        )

    whole_body_joint_map_ref, whole_body_joint_map_source = _select_candidate_file(
        root_candidates=humanoidverse_roots,
        relative_candidates=("humanoidverse/config/robot/g1/g1_29dof.yaml",),
        glob_patterns=("**/config/robot/g1/g1_29dof.yaml",),
        content_markers=("dof_names",),
    )
    if not whole_body_joint_map_ref:
        whole_body_joint_map_ref, whole_body_joint_map_source = _select_candidate_file(
            root_candidates=unitree_sim_roots,
            relative_candidates=("robots/unitree.py",),
            glob_patterns=("**/robots/unitree.py",),
            content_markers=("joint_names_expr",),
        )
    if not whole_body_joint_map_ref:
        whole_body_joint_map_ref, whole_body_joint_map_source = _select_candidate_file(
            root_candidates=unitree_rl_roots,
            relative_candidates=(
                "resources/robots/g1_description/g1_29dof_rev_1_0.urdf",
                "resources/robots/g1_description/g1_29dof.urdf",
            ),
            glob_patterns=("**/g1_29dof*.urdf",),
            content_markers=("<joint name=",),
        )

    joint_limit_ref, joint_limit_source = _select_candidate_file(
        root_candidates=humanoidverse_roots,
        relative_candidates=("humanoidverse/config/robot/g1/g1_29dof.yaml",),
        glob_patterns=("**/config/robot/g1/g1_29dof.yaml",),
        content_markers=("dof_pos_lower_limit_list", "dof_pos_upper_limit_list"),
    )
    if not joint_limit_ref:
        joint_limit_ref, joint_limit_source = _select_candidate_file(
            root_candidates=unitree_rl_roots,
            relative_candidates=(
                "resources/robots/g1_description/g1_29dof_rev_1_0.urdf",
                "resources/robots/g1_description/g1_29dof.urdf",
            ),
            glob_patterns=("**/g1_29dof*.urdf",),
            content_markers=("<limit",),
        )
    if not joint_limit_ref:
        joint_limit_ref, joint_limit_source = _select_candidate_file(
            root_candidates=xr_roots,
            relative_candidates=("assets/g1/g1_body29_hand14.urdf",),
            glob_patterns=("**/g1_body29*.urdf",),
            content_markers=("<limit",),
        )

    control_frequency_ref, control_frequency_source = _select_candidate_file(
        root_candidates=unitree_sim_roots,
        relative_candidates=("sim_main.py",),
        glob_patterns=("**/sim_main.py",),
        content_markers=("step_hz", "control frequency"),
    )
    if not control_frequency_ref:
        control_frequency_ref, control_frequency_source = _select_candidate_file(
            root_candidates=lerobot_roots,
            relative_candidates=("README.md",),
            glob_patterns=("**/README.md",),
            content_markers=("--frequency", "Hz"),
        )

    teleop_recovery_ref, teleop_recovery_source = _select_candidate_file(
        root_candidates=xr_roots,
        relative_candidates=("teleop/teleop_hand_and_arm.py",),
        glob_patterns=("**/teleop_hand_and_arm.py", "**/README.md"),
        content_markers=("soft emergency stop", "damping mode"),
    )

    derived_assets: dict[str, dict[str, Any]] = {}
    if robot_description_ref:
        derived_assets["unitree_robot_description"] = _derived_asset_row(
            value=robot_description_ref,
            source=robot_description_source,
        )
    if whole_body_joint_map_ref:
        derived_assets["whole_body_joint_map"] = _derived_asset_row(
            value=whole_body_joint_map_ref,
            source=whole_body_joint_map_source,
        )
    if joint_limit_ref:
        derived_assets["joint_limit_profile"] = _derived_asset_row(
            value=joint_limit_ref,
            source=joint_limit_source,
        )
    if control_frequency_ref:
        derived_assets["control_frequency_profile"] = _derived_asset_row(
            value=control_frequency_ref,
            source=control_frequency_source,
        )
    if teleop_recovery_ref:
        derived_assets["teleop_recovery_contract"] = _derived_asset_row(
            value=teleop_recovery_ref,
            source=teleop_recovery_source,
        )
    return derived_assets


def _merge_explicit_and_derived_asset_rows(
    explicit_row: dict[str, Any],
    derived_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if not derived_row:
        return explicit_row
    explicit_present = bool(explicit_row.get("present", False))
    if not explicit_present:
        return derived_row
    if bool(explicit_row.get("local_path_exists", False)):
        explicit_row["derived_candidate_value"] = derived_row.get("value")
        explicit_row["derived_candidate_source"] = derived_row.get("derivation_source")
        return explicit_row
    if str(explicit_row.get("verification_status", "") or "") == "declared_non_path":
        explicit_row["derived_candidate_value"] = derived_row.get("value")
        explicit_row["derived_candidate_source"] = derived_row.get("derivation_source")
        return explicit_row
    merged = dict(derived_row)
    merged["matched_aliases"] = list(explicit_row.get("matched_aliases") or [])
    merged["explicit_declared_value"] = explicit_row.get("value")
    merged["explicit_declared_verification_status"] = explicit_row.get("verification_status")
    merged["selection_reason"] = "derived_local_asset_overrides_missing_declared_path"
    return merged


def normalize_robot_asset_manifest(
    embodiment_context: Mapping[str, Any] | None,
    *,
    runtime_target_contract: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    manifest = extract_robot_asset_manifest(embodiment_context)
    derived_assets = _derive_unitree_assets(
        embodiment_context,
        runtime_target_contract,
    )
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
        explicit_row = {
            "present": bool(matched_aliases),
            "matched_aliases": matched_aliases,
            "value": (
                None
                if not matched_aliases
                else manifest.get(matched_aliases[0])
            ),
            **_asset_verification(
                None if not matched_aliases else manifest.get(matched_aliases[0])
            ),
        }
        normalized[canonical] = _merge_explicit_and_derived_asset_rows(
            explicit_row,
            derived_assets.get(canonical),
        )
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
            "value_is_path": False,
            "local_path_exists": False,
            "verification_status": "additional_mapping",
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
    *,
    runtime_target_contract: Mapping[str, Any] | None = None,
) -> list[str]:
    normalized = normalize_robot_asset_manifest(
        embodiment_context,
        runtime_target_contract=runtime_target_contract,
    )
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
    *,
    runtime_target_contract: Mapping[str, Any] | None = None,
) -> list[str]:
    available = set(
        available_assets_for_hardware_class(
            target_hardware_class,
            embodiment_context,
            runtime_target_contract=runtime_target_contract,
        )
    )
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

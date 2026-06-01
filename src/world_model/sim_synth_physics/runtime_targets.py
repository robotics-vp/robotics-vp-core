"""Concrete runtime-target discovery for sim/synth/physics backend lanes."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Mapping

from .common import mapping
from .holosoma_runtime_gate import holosoma_importable, holosoma_runtime_enabled
from .local_runtime_discovery import discover_named_root


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _context_path(embodiment_context: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(embodiment_context.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _env_path(*keys: str) -> str:
    for key in keys:
        value = str(os.environ.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _resolved_path_with_source(
    *,
    explicit_ref: str,
    discover_names: tuple[str, ...] = (),
) -> tuple[str, str, list[str]]:
    cleaned = str(explicit_ref or "").strip()
    if cleaned:
        return cleaned, "embodiment_or_env", []
    if not discover_names:
        return "", "", []
    discovered = discover_named_root(discover_names)
    checked_paths = discovered.get("checked_paths", [])
    return (
        str(discovered.get("ref", "") or ""),
        str(discovered.get("source", "") or ""),
        [str(path) for path in checked_paths]
        if isinstance(checked_paths, (list, tuple))
        else [],
    )


def _fallback_child_path_with_source(
    *,
    parent_ref: str,
    source_prefix: str,
    relative_candidates: tuple[str, ...],
) -> tuple[str, str, list[str]]:
    parent = Path(str(parent_ref or "").strip())
    checked_paths: list[str] = []
    if not parent:
        return "", "", checked_paths
    for rel_path in relative_candidates:
        candidate = (parent / rel_path).resolve()
        checked_paths.append(str(candidate))
        if candidate.exists():
            return str(candidate), f"{source_prefix}_subpath", checked_paths
    return "", "", checked_paths


def _target_record(
    *,
    target_id: str,
    label: str,
    ref: str,
    source: str,
    checked_paths: list[str] | None = None,
    exact_markers: tuple[str, ...] = (),
    glob_markers: tuple[str, ...] = (),
    marker_policy: str = "any",
) -> dict[str, Any]:
    path = Path(ref) if ref else None
    exists = bool(ref and path is not None and path.exists())
    matched_markers: list[str] = []
    missing_markers: list[str] = []
    marker_refs: list[str] = []
    if exists and path is not None:
        for marker in exact_markers:
            marker_path = (path / marker).resolve()
            if marker_path.exists():
                matched_markers.append(marker)
                marker_refs.append(str(marker_path))
            else:
                missing_markers.append(marker)
        for pattern in glob_markers:
            pattern_matches = sorted(path.rglob(pattern))
            if pattern_matches:
                matched_markers.append(pattern)
                marker_refs.append(str(pattern_matches[0].resolve()))
            else:
                missing_markers.append(pattern)
    marker_expected = list(exact_markers) + list(glob_markers)
    marker_verified = True
    if marker_expected:
        if marker_policy == "all":
            marker_verified = not missing_markers
        else:
            marker_verified = bool(matched_markers)
    verification_status = "missing"
    if exists:
        if not marker_expected:
            verification_status = "local_path_exists"
        elif marker_verified:
            verification_status = "install_shape_ready"
        elif matched_markers:
            verification_status = "install_shape_partial"
        else:
            verification_status = "install_shape_missing"
    return {
        "target_id": target_id,
        "label": label,
        "ref": ref,
        "exists": exists,
        "source": source,
        "checked_paths": list(checked_paths or []),
        "path_kind": (
            "missing"
            if not exists or path is None
            else "directory"
            if path.is_dir()
            else "file"
            if path.is_file()
            else "other"
        ),
        "expected_markers": marker_expected,
        "matched_markers": matched_markers,
        "missing_markers": missing_markers,
        "primary_marker_ref": "" if not marker_refs else marker_refs[0],
        "marker_policy": marker_policy,
        "verified": bool(exists and marker_verified),
        "verification_status": verification_status,
    }


def _runtime_stack_summary(
    *,
    backend: str,
    records: list[dict[str, Any]],
    python_bridge_available: bool,
    required_target_ids: list[str],
    one_of_groups: list[list[str]] | None = None,
) -> dict[str, Any]:
    by_id = {
        str(record["target_id"]): bool(record.get("exists", False))
        for record in records
    }
    verified_by_id = {
        str(record["target_id"]): bool(record.get("verified", False))
        for record in records
    }
    missing_required = [
        target_id
        for target_id in required_target_ids
        if not by_id.get(target_id, False)
    ]
    unverified_required = [
        target_id
        for target_id in required_target_ids
        if by_id.get(target_id, False) and not verified_by_id.get(target_id, False)
    ]
    unresolved_groups: list[list[str]] = []
    for group in list(one_of_groups or []):
        if not any(by_id.get(target_id, False) for target_id in group):
            unresolved_groups.append(list(group))
    target_preflight_status = "preflight_ready"
    if missing_required or unresolved_groups:
        target_preflight_status = "preflight_blocked"
    elif unverified_required:
        target_preflight_status = "preflight_partial"
    return {
        "version": "backend_runtime_target_contract_v1",
        "backend": backend,
        "python_bridge_available": bool(python_bridge_available),
        "targets": records,
        "required_target_ids": list(required_target_ids),
        "missing_required_target_ids": missing_required,
        "unverified_required_target_ids": unverified_required,
        "one_of_target_groups": list(one_of_groups or []),
        "unresolved_one_of_groups": unresolved_groups,
        "runtime_targets_ready": not missing_required and not unresolved_groups,
        "runtime_target_preflight_status": target_preflight_status,
        "ready_target_ids": [
            target_id for target_id, present in by_id.items() if present
        ],
        "verified_target_ids": [
            target_id for target_id, present in verified_by_id.items() if present
        ],
    }


def describe_isaac_runtime_targets(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embodiment = mapping(embodiment_context)
    isaaclab_root, isaaclab_source, isaaclab_checked = _resolved_path_with_source(
        explicit_ref=_context_path(
            embodiment, "isaaclab_root", "isaac_lab_root", "isaac_repo_root"
        )
        or _env_path("ISAACLAB_ROOT", "ISAAC_LAB_ROOT"),
        discover_names=("IsaacLab",),
    )
    isaacsim_root, isaacsim_source, isaacsim_checked = _resolved_path_with_source(
        explicit_ref=_context_path(embodiment, "isaacsim_root", "isaac_sim_root")
        or _env_path("ISAACSIM_ROOT", "ISAAC_SIM_ROOT", "OMNI_ISAAC_ROOT"),
    )
    unitree_sdk2_root, unitree_sdk2_source, unitree_sdk2_checked = (
        _resolved_path_with_source(
            explicit_ref=_context_path(
                embodiment, "unitree_sdk2_root", "unitree_sdk_root"
            )
            or _env_path("UNITREE_SDK2_ROOT", "UNITREE_SDK_ROOT"),
            discover_names=("unitree_sdk2", "unitree_sdk"),
        )
    )
    unitree_asset_root, unitree_asset_source, unitree_asset_checked = (
        _resolved_path_with_source(
            explicit_ref=_context_path(
                embodiment,
                "unitree_asset_root",
                "unitree_assets_root",
                "robot_asset_root",
            )
            or _env_path(
                "UNITREE_ASSET_ROOT", "UNITREE_ASSETS_ROOT", "UNITREE_URDF_ROOT"
            ),
            discover_names=("unitree_assets", "unitree_asset_root", "unitree_models"),
        )
    )
    unitree_rl_gym_root, unitree_rl_source, unitree_rl_checked = (
        _resolved_path_with_source(
            explicit_ref=_context_path(
                embodiment,
                "unitree_rl_gym_root",
                "unitree_runtime_repo_root",
            )
            or _env_path("UNITREE_RL_GYM_ROOT"),
            discover_names=("unitree_rl_gym",),
        )
    )
    unitree_sim_isaaclab_root, unitree_sim_source, unitree_sim_checked = (
        _resolved_path_with_source(
            explicit_ref=_context_path(
                embodiment,
                "unitree_sim_isaaclab_root",
            )
            or _env_path("UNITREE_SIM_ISAACLAB_ROOT"),
            discover_names=("unitree_sim_isaaclab",),
        )
    )
    humanoidverse_root, humanoidverse_source, humanoidverse_checked = (
        _resolved_path_with_source(
            explicit_ref=_context_path(embodiment, "humanoidverse_root")
            or _env_path("HUMANOIDVERSE_ROOT"),
            discover_names=("HumanoidVerse", "humanoidverse"),
        )
    )
    xr_teleoperate_root, xr_source, xr_checked = _resolved_path_with_source(
        explicit_ref=_context_path(embodiment, "xr_teleoperate_root")
        or _env_path("XR_TELEOPERATE_ROOT"),
        discover_names=("xr_teleoperate",),
    )
    unitree_model_root, unitree_model_source, unitree_model_checked = (
        _resolved_path_with_source(
            explicit_ref=_context_path(embodiment, "unitree_model_root")
            or _env_path("UNITREE_MODEL_ROOT"),
            discover_names=("unitree_model", "unitree_models"),
        )
    )
    unitree_policy_root, unitree_policy_source, unitree_policy_checked = (
        _resolved_path_with_source(
            explicit_ref=_context_path(
                embodiment,
                "unitree_policy_root",
                "isaac_policy_root",
            )
            or _env_path("UNITREE_POLICY_ROOT", "ISAAC_POLICY_ROOT"),
        )
    )
    (
        unitree_sdk2_python_root,
        unitree_sdk2_python_source,
        unitree_sdk2_python_checked,
    ) = _resolved_path_with_source(
        explicit_ref=_context_path(
            embodiment,
            "unitree_sdk2_python_root",
        )
        or _env_path("UNITREE_SDK2_PYTHON_ROOT"),
    )
    teleimager_root, teleimager_source, teleimager_checked = _resolved_path_with_source(
        explicit_ref=_context_path(
            embodiment,
            "teleimager_root",
        )
        or _env_path("TELEIMAGER_ROOT"),
    )
    unitree_il_lerobot_root, lerobot_source, lerobot_checked = (
        _resolved_path_with_source(
            explicit_ref=_context_path(
                embodiment,
                "unitree_lerobot_root",
                "unitree_il_lerobot_root",
            )
            or _env_path("UNITREE_IL_LEROBOT_ROOT", "UNITREE_LEROBOT_ROOT"),
            discover_names=(
                "unitree_IL_lerobot",
                "unitree_il_lerobot",
                "unitree_lerobot",
            ),
        )
    )
    records = [
        _target_record(
            target_id="isaaclab_root",
            label="Isaac Lab root",
            ref=isaaclab_root,
            source=isaaclab_source,
            checked_paths=isaaclab_checked,
            exact_markers=("source", "apps"),
        ),
        _target_record(
            target_id="isaacsim_root",
            label="Isaac Sim root",
            ref=isaacsim_root,
            source=isaacsim_source,
            checked_paths=isaacsim_checked,
            exact_markers=("apps",),
        ),
        _target_record(
            target_id="unitree_sdk2_root",
            label="Unitree SDK2 root",
            ref=unitree_sdk2_root,
            source=unitree_sdk2_source,
            checked_paths=unitree_sdk2_checked,
            exact_markers=("include", "lib", "python", "README.md"),
        ),
        _target_record(
            target_id="unitree_asset_root",
            label="Unitree asset root",
            ref=unitree_asset_root,
            source=unitree_asset_source,
            checked_paths=unitree_asset_checked,
            glob_markers=(
                "**/*.usd",
                "**/*.urdf",
                "**/*.xml",
                "**/*.yaml",
                "**/*.json",
            ),
        ),
        _target_record(
            target_id="unitree_sim_isaaclab_root",
            label="Unitree Isaac Lab root",
            ref=unitree_sim_isaaclab_root,
            source=unitree_sim_source,
            checked_paths=unitree_sim_checked,
            exact_markers=("sim_main.py", "dds"),
        ),
        _target_record(
            target_id="unitree_rl_gym_root",
            label="Unitree RL Gym root",
            ref=unitree_rl_gym_root,
            source=unitree_rl_source,
            checked_paths=unitree_rl_checked,
            exact_markers=("deploy", "legged_gym"),
        ),
        _target_record(
            target_id="xr_teleoperate_root",
            label="XR Teleoperate root",
            ref=xr_teleoperate_root,
            source=xr_source,
            checked_paths=xr_checked,
            exact_markers=("teleop",),
        ),
        _target_record(
            target_id="unitree_model_root",
            label="Unitree model root",
            ref=unitree_model_root,
            source=unitree_model_source,
            checked_paths=unitree_model_checked,
            glob_markers=("**/*.usd", "**/*.urdf"),
        ),
        _target_record(
            target_id="unitree_policy_root",
            label="Unitree policy root",
            ref=unitree_policy_root,
            source=unitree_policy_source,
            checked_paths=unitree_policy_checked,
            glob_markers=("**/*.onnx", "**/*.pt", "**/*.pth", "**/*.ckpt"),
        ),
        _target_record(
            target_id="unitree_sdk2_python_root",
            label="Unitree SDK2 Python root",
            ref=unitree_sdk2_python_root,
            source=unitree_sdk2_python_source,
            checked_paths=unitree_sdk2_python_checked,
            exact_markers=("setup.py", "pyproject.toml"),
        ),
        _target_record(
            target_id="humanoidverse_root",
            label="HumanoidVerse root",
            ref=humanoidverse_root,
            source=humanoidverse_source,
            checked_paths=humanoidverse_checked,
            exact_markers=("humanoidverse", "assets"),
        ),
        _target_record(
            target_id="teleimager_root",
            label="Teleimager root",
            ref=teleimager_root,
            source=teleimager_source,
            checked_paths=teleimager_checked,
            exact_markers=("README.md", "scripts"),
        ),
        _target_record(
            target_id="unitree_il_lerobot_root",
            label="Unitree IL Lerobot root",
            ref=unitree_il_lerobot_root,
            source=lerobot_source,
            checked_paths=lerobot_checked,
            exact_markers=("examples", "scripts", "lerobot"),
        ),
    ]
    summary = _runtime_stack_summary(
        backend="isaac",
        records=records,
        python_bridge_available=_has_module(
            "src.motor_backend.workcell_isaaclab_backend"
        ),
        required_target_ids=["unitree_sdk2_root", "unitree_asset_root"],
        one_of_groups=[
            [
                "isaaclab_root",
                "isaacsim_root",
                "unitree_sim_isaaclab_root",
                "unitree_rl_gym_root",
                "unitree_il_lerobot_root",
                "humanoidverse_root",
                "xr_teleoperate_root",
            ]
        ],
    )
    summary["preferred_runtime_roots"] = [
        target_id
        for target_id in (
            "unitree_sim_isaaclab_root",
            "unitree_rl_gym_root",
            "unitree_il_lerobot_root",
            "humanoidverse_root",
            "isaaclab_root",
            "isaacsim_root",
            "xr_teleoperate_root",
            "unitree_model_root",
            "unitree_policy_root",
            "unitree_sdk2_python_root",
            "teleimager_root",
            "unitree_il_lerobot_root",
        )
        if target_id in summary["ready_target_ids"]
    ]
    return summary


def describe_holosoma_runtime_targets(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embodiment = mapping(embodiment_context)
    holosoma_root, holosoma_source, holosoma_checked = _resolved_path_with_source(
        explicit_ref=_context_path(embodiment, "holosoma_root", "holosoma_repo_root")
        or _env_path("HOLOSOMA_ROOT", "HOLOSOMA_REPO_ROOT"),
        discover_names=("holosoma",),
    )
    holosoma_motion_root, holosoma_motion_source, holosoma_motion_checked = (
        _resolved_path_with_source(
            explicit_ref=_context_path(
                embodiment,
                "holosoma_motion_root",
                "motion_data_root",
            )
            or _env_path("HOLOSOMA_MOTION_ROOT"),
            discover_names=("holosoma_motion", "motions"),
        )
    )
    if not holosoma_motion_root and holosoma_root:
        (
            holosoma_motion_root,
            holosoma_motion_source,
            fallback_checked,
        ) = _fallback_child_path_with_source(
            parent_ref=holosoma_root,
            source_prefix=holosoma_source or "holosoma_root",
            relative_candidates=(
                "src/holosoma/holosoma/data/motions",
                "src/holosoma/data/motions",
                "data/motions",
            ),
        )
        holosoma_motion_checked = list(holosoma_motion_checked) + fallback_checked
    holosoma_policy_root, holosoma_policy_source, holosoma_policy_checked = (
        _resolved_path_with_source(
            explicit_ref=_context_path(
                embodiment,
                "holosoma_policy_root",
                "policy_root",
            )
            or _env_path("HOLOSOMA_POLICY_ROOT"),
        )
    )
    if not holosoma_policy_root and holosoma_root:
        (
            holosoma_policy_root,
            holosoma_policy_source,
            fallback_checked,
        ) = _fallback_child_path_with_source(
            parent_ref=holosoma_root,
            source_prefix=holosoma_source or "holosoma_root",
            relative_candidates=(
                "src/holosoma_inference/holosoma_inference/models",
                "src/holosoma_inference/models",
                "models",
                "checkpoints",
            ),
        )
        holosoma_policy_checked = list(holosoma_policy_checked) + fallback_checked
    retargeting_root, retarget_source, retarget_checked = _resolved_path_with_source(
        explicit_ref=_context_path(
            embodiment,
            "retargeting_root",
            "whole_body_retargeting_root",
        )
        or _env_path("RETARGETING_ROOT"),
        discover_names=("retargeting",),
    )
    if not retargeting_root and holosoma_root:
        retargeting_root, retarget_source, fallback_checked = (
            _fallback_child_path_with_source(
                parent_ref=holosoma_root,
                source_prefix=holosoma_source or "holosoma_root",
                relative_candidates=(
                    "src/holosoma_retargeting",
                    "src/holosoma_retargeting/holosoma_retargeting",
                    "retargeting",
                ),
            )
        )
        retarget_checked = list(retarget_checked) + fallback_checked
    records = [
        _target_record(
            target_id="holosoma_root",
            label="Holosoma root",
            ref=holosoma_root,
            source=holosoma_source,
            checked_paths=holosoma_checked,
            exact_markers=("README.md", "src", "scripts"),
        ),
        _target_record(
            target_id="holosoma_motion_root",
            label="Holosoma motion root",
            ref=holosoma_motion_root,
            source=holosoma_motion_source,
            checked_paths=holosoma_motion_checked,
            glob_markers=("**/*.npz", "**/*.npy", "**/*.bvh", "**/*.pkl"),
        ),
        _target_record(
            target_id="holosoma_policy_root",
            label="Holosoma policy root",
            ref=holosoma_policy_root,
            source=holosoma_policy_source,
            checked_paths=holosoma_policy_checked,
            glob_markers=(
                "**/*.onnx",
                "**/*.pt",
                "**/*.pth",
                "**/*.ckpt",
                "**/*.yaml",
                "**/*.yml",
            ),
        ),
        _target_record(
            target_id="retargeting_root",
            label="Whole-body retargeting root",
            ref=retargeting_root,
            source=retarget_source,
            checked_paths=retarget_checked,
            glob_markers=("**/*.yaml", "**/*.json", "**/*.npz"),
        ),
    ]
    summary = _runtime_stack_summary(
        backend="holosoma",
        records=records,
        python_bridge_available=holosoma_runtime_enabled(),
        required_target_ids=["holosoma_motion_root"],
        one_of_groups=[["holosoma_root", "holosoma_policy_root"]],
    )
    summary["python_bridge_importable"] = holosoma_importable()
    return summary


__all__ = ["describe_holosoma_runtime_targets", "describe_isaac_runtime_targets"]

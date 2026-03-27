"""Concrete runtime-target discovery for sim/synth/physics backend lanes."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Mapping

from .common import mapping


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


def _target_record(*, target_id: str, label: str, ref: str, source: str) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "label": label,
        "ref": ref,
        "exists": bool(ref and Path(ref).exists()),
        "source": source,
    }


def _runtime_stack_summary(
    *,
    backend: str,
    records: list[dict[str, Any]],
    python_bridge_available: bool,
    required_target_ids: list[str],
    one_of_groups: list[list[str]] | None = None,
) -> dict[str, Any]:
    by_id = {str(record["target_id"]): bool(record.get("exists", False)) for record in records}
    missing_required = [target_id for target_id in required_target_ids if not by_id.get(target_id, False)]
    unresolved_groups: list[list[str]] = []
    for group in list(one_of_groups or []):
        if not any(by_id.get(target_id, False) for target_id in group):
            unresolved_groups.append(list(group))
    return {
        "version": "backend_runtime_target_contract_v1",
        "backend": backend,
        "python_bridge_available": bool(python_bridge_available),
        "targets": records,
        "required_target_ids": list(required_target_ids),
        "missing_required_target_ids": missing_required,
        "one_of_target_groups": list(one_of_groups or []),
        "unresolved_one_of_groups": unresolved_groups,
        "runtime_targets_ready": not missing_required and not unresolved_groups,
        "ready_target_ids": [target_id for target_id, present in by_id.items() if present],
    }


def describe_isaac_runtime_targets(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embodiment = mapping(embodiment_context)
    isaaclab_root = _context_path(embodiment, "isaaclab_root", "isaac_lab_root", "isaac_repo_root") or _env_path(
        "ISAACLAB_ROOT",
        "ISAAC_LAB_ROOT",
    )
    isaacsim_root = _context_path(embodiment, "isaacsim_root", "isaac_sim_root") or _env_path(
        "ISAACSIM_ROOT",
        "ISAAC_SIM_ROOT",
        "OMNI_ISAAC_ROOT",
    )
    unitree_sdk2_root = _context_path(embodiment, "unitree_sdk2_root", "unitree_sdk_root") or _env_path(
        "UNITREE_SDK2_ROOT",
        "UNITREE_SDK_ROOT",
    )
    unitree_asset_root = _context_path(
        embodiment,
        "unitree_asset_root",
        "unitree_assets_root",
        "robot_asset_root",
    ) or _env_path(
        "UNITREE_ASSET_ROOT",
        "UNITREE_ASSETS_ROOT",
        "UNITREE_URDF_ROOT",
    )
    unitree_rl_gym_root = _context_path(
        embodiment,
        "unitree_rl_gym_root",
        "unitree_runtime_repo_root",
    ) or _env_path("UNITREE_RL_GYM_ROOT")
    humanoidverse_root = _context_path(embodiment, "humanoidverse_root") or _env_path(
        "HUMANOIDVERSE_ROOT"
    )
    records = [
        _target_record(
            target_id="isaaclab_root",
            label="Isaac Lab root",
            ref=isaaclab_root,
            source="embodiment_or_env",
        ),
        _target_record(
            target_id="isaacsim_root",
            label="Isaac Sim root",
            ref=isaacsim_root,
            source="embodiment_or_env",
        ),
        _target_record(
            target_id="unitree_sdk2_root",
            label="Unitree SDK2 root",
            ref=unitree_sdk2_root,
            source="embodiment_or_env",
        ),
        _target_record(
            target_id="unitree_asset_root",
            label="Unitree asset root",
            ref=unitree_asset_root,
            source="embodiment_or_env",
        ),
        _target_record(
            target_id="unitree_rl_gym_root",
            label="Unitree RL Gym root",
            ref=unitree_rl_gym_root,
            source="embodiment_or_env",
        ),
        _target_record(
            target_id="humanoidverse_root",
            label="HumanoidVerse root",
            ref=humanoidverse_root,
            source="embodiment_or_env",
        ),
    ]
    summary = _runtime_stack_summary(
        backend="isaac",
        records=records,
        python_bridge_available=_has_module("src.motor_backend.workcell_isaaclab_backend"),
        required_target_ids=["unitree_sdk2_root", "unitree_asset_root"],
        one_of_groups=[["isaaclab_root", "isaacsim_root", "unitree_rl_gym_root", "humanoidverse_root"]],
    )
    summary["preferred_runtime_roots"] = [
        target_id
        for target_id in ("isaaclab_root", "isaacsim_root", "unitree_rl_gym_root", "humanoidverse_root")
        if target_id in summary["ready_target_ids"]
    ]
    return summary


def describe_holosoma_runtime_targets(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embodiment = mapping(embodiment_context)
    holosoma_root = _context_path(embodiment, "holosoma_root", "holosoma_repo_root") or _env_path(
        "HOLOSOMA_ROOT",
        "HOLOSOMA_REPO_ROOT",
    )
    holosoma_motion_root = _context_path(
        embodiment,
        "holosoma_motion_root",
        "motion_data_root",
    ) or _env_path("HOLOSOMA_MOTION_ROOT")
    holosoma_policy_root = _context_path(
        embodiment,
        "holosoma_policy_root",
        "policy_root",
    ) or _env_path("HOLOSOMA_POLICY_ROOT")
    retargeting_root = _context_path(
        embodiment,
        "retargeting_root",
        "whole_body_retargeting_root",
    ) or _env_path("RETARGETING_ROOT")
    records = [
        _target_record(
            target_id="holosoma_root",
            label="Holosoma root",
            ref=holosoma_root,
            source="embodiment_or_env",
        ),
        _target_record(
            target_id="holosoma_motion_root",
            label="Holosoma motion root",
            ref=holosoma_motion_root,
            source="embodiment_or_env",
        ),
        _target_record(
            target_id="holosoma_policy_root",
            label="Holosoma policy root",
            ref=holosoma_policy_root,
            source="embodiment_or_env",
        ),
        _target_record(
            target_id="retargeting_root",
            label="Whole-body retargeting root",
            ref=retargeting_root,
            source="embodiment_or_env",
        ),
    ]
    summary = _runtime_stack_summary(
        backend="holosoma",
        records=records,
        python_bridge_available=_has_module("holosoma"),
        required_target_ids=["holosoma_motion_root"],
        one_of_groups=[["holosoma_root", "holosoma_policy_root"]],
    )
    return summary


__all__ = ["describe_holosoma_runtime_targets", "describe_isaac_runtime_targets"]

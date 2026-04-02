"""OSS-informed runtime layout and policy contracts for Phase-1 backends."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import mapping
from .local_runtime_discovery import discover_named_root


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


def _resolved_path(explicit_ref: str, *, discover_names: tuple[str, ...] = ()) -> str:
    cleaned = str(explicit_ref or "").strip()
    if cleaned:
        return cleaned
    if not discover_names:
        return ""
    return str(discover_named_root(discover_names).get("ref", "") or "")


def _existing(root: str, rel_paths: Iterable[str]) -> tuple[list[str], list[str]]:
    if not root:
        return [], list(rel_paths)
    base = Path(root)
    matched: list[str] = []
    missing: list[str] = []
    for rel_path in rel_paths:
        if (base / rel_path).exists():
            matched.append(rel_path)
        else:
            missing.append(rel_path)
    return matched, missing


def _git_metadata(root: str) -> dict[str, Any]:
    if not root:
        return {}
    base = Path(root)
    if not base.exists():
        return {}
    try:
        head = subprocess.run(
            ["git", "-C", str(base), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode != 0:
            return {}
        remote = subprocess.run(
            ["git", "-C", str(base), "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(base), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "repo_head": head.stdout.strip(),
            "remote_origin": remote.stdout.strip(),
            "dirty": bool(dirty.stdout.strip()),
        }
    except Exception:
        return {}


def _profile(
    *,
    profile_id: str,
    label: str,
    root: str,
    expected_paths: list[str],
    preferred_entrypoints: list[str],
    deploy_patterns: list[str] | None = None,
    policy_patterns: list[str] | None = None,
    data_patterns: list[str] | None = None,
) -> dict[str, Any]:
    matched_paths, missing_paths = _existing(root, expected_paths)
    deploy_candidate_records = _candidate_records(root, deploy_patterns or [])
    policy_candidate_records = _candidate_records(root, policy_patterns or [])
    data_candidate_records = _candidate_records(root, data_patterns or [])
    return {
        "profile_id": profile_id,
        "label": label,
        "root": root,
        "root_exists": bool(root and Path(root).exists()),
        "root_git_metadata": _git_metadata(root),
        "expected_paths": list(expected_paths),
        "matched_paths": matched_paths,
        "missing_paths": missing_paths,
        "preferred_entrypoints": list(preferred_entrypoints),
        "deploy_candidates": [str(row["ref"]) for row in deploy_candidate_records],
        "policy_candidates": [str(row["ref"]) for row in policy_candidate_records],
        "data_candidates": [str(row["ref"]) for row in data_candidate_records],
        "deploy_candidate_records": deploy_candidate_records,
        "policy_candidate_records": policy_candidate_records,
        "data_candidate_records": data_candidate_records,
        "deploy_candidate_count": len(deploy_candidate_records),
        "policy_candidate_count": len(policy_candidate_records),
        "data_candidate_count": len(data_candidate_records),
        "primary_deploy_candidate": (
            "" if not deploy_candidate_records else str(deploy_candidate_records[0]["ref"])
        ),
        "primary_policy_candidate": (
            "" if not policy_candidate_records else str(policy_candidate_records[0]["ref"])
        ),
        "primary_data_candidate": (
            "" if not data_candidate_records else str(data_candidate_records[0]["ref"])
        ),
        "profile_ready": bool(root and Path(root).exists() and not missing_paths),
    }


def _candidate_files(root: str, patterns: Iterable[str], limit: int = 6) -> list[str]:
    if not root:
        return []
    base = Path(root)
    if not base.exists():
        return []
    matches: list[str] = []
    for pattern in patterns:
        for path in sorted(base.rglob(pattern)):
            matches.append(str(path.resolve()))
            if len(matches) >= limit:
                return matches
    return matches


def _candidate_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "ref": str(path.resolve()),
        "name": path.name,
        "suffix": path.suffix.lower(),
        "bytes": int(stat.st_size),
        "mtime_s": float(stat.st_mtime),
    }


def _candidate_records(root: str, patterns: Iterable[str], limit: int = 6) -> list[dict[str, Any]]:
    if not root:
        return []
    base = Path(root)
    if not base.exists():
        return []
    matches: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in sorted(base.rglob(pattern)):
            matches.append(_candidate_record(path))
            if len(matches) >= limit:
                return matches
    return matches


def describe_isaac_runtime_layouts(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embodiment = mapping(embodiment_context)
    isaaclab_root = _resolved_path(
        _context_path(
        embodiment, "isaaclab_root", "isaac_lab_root", "isaac_repo_root"
    ) or _env_path("ISAACLAB_ROOT", "ISAAC_LAB_ROOT"),
        discover_names=("IsaacLab",),
    )
    unitree_sim_isaaclab_root = _resolved_path(
        _context_path(
        embodiment, "unitree_sim_isaaclab_root"
    ) or _env_path("UNITREE_SIM_ISAACLAB_ROOT"),
        discover_names=("unitree_sim_isaaclab",),
    )
    unitree_rl_gym_root = _resolved_path(
        _context_path(
        embodiment, "unitree_rl_gym_root", "unitree_runtime_repo_root"
    ) or _env_path("UNITREE_RL_GYM_ROOT"),
        discover_names=("unitree_rl_gym",),
    )
    humanoidverse_root = _resolved_path(
        _context_path(
        embodiment, "humanoidverse_root"
    ) or _env_path("HUMANOIDVERSE_ROOT"),
        discover_names=("HumanoidVerse", "humanoidverse"),
    )
    xr_teleoperate_root = _resolved_path(
        _context_path(
        embodiment, "xr_teleoperate_root"
    ) or _env_path("XR_TELEOPERATE_ROOT"),
        discover_names=("xr_teleoperate",),
    )
    unitree_lerobot_root = _resolved_path(
        _context_path(
        embodiment, "unitree_lerobot_root", "unitree_il_lerobot_root"
    ) or _env_path("UNITREE_LEROBOT_ROOT", "UNITREE_IL_LEROBOT_ROOT"),
        discover_names=("unitree_IL_lerobot", "unitree_il_lerobot", "unitree_lerobot"),
    )
    unitree_model_root = _resolved_path(
        _context_path(
        embodiment, "unitree_model_root"
    ) or _env_path("UNITREE_MODEL_ROOT"),
        discover_names=("unitree_model", "unitree_models"),
    )
    profiles = [
        _profile(
            profile_id="isaaclab_core",
            label="Isaac Lab core repo",
            root=isaaclab_root,
            expected_paths=["apps", "source", "pyproject.toml"],
            preferred_entrypoints=["source", "apps"],
            deploy_patterns=["source/standalone/workflows/rl/play.py", "apps/**/*"],
            policy_patterns=["logs/**/*.pt", "logs/**/*.onnx"],
            data_patterns=["logs/**/*.json", "logs/**/*.yaml"],
        ),
        _profile(
            profile_id="unitree_sim_isaaclab",
            label="Unitree Isaac Lab bridge repo",
            root=unitree_sim_isaaclab_root,
            expected_paths=["sim_main.py", "dds", "action_provider"],
            preferred_entrypoints=["sim_main.py", "dds"],
            deploy_patterns=["sim_main.py", "dds/**/*", "action_provider/**/*"],
            policy_patterns=["logs/**/*.pt", "logs/**/*.onnx", "policies/**/*.onnx"],
            data_patterns=["logs/**/*.json", "generated/**/*.json", "recordings/**/*"],
        ),
        _profile(
            profile_id="unitree_rl_gym",
            label="Unitree RL Gym repo",
            root=unitree_rl_gym_root,
            expected_paths=["legged_gym", "resources", "deploy"],
            preferred_entrypoints=["deploy", "legged_gym"],
            deploy_patterns=["deploy/**/*.py", "deploy_real/**/*"],
            policy_patterns=["logs/**/*.pt", "logs/**/*.onnx", "exported/**/*"],
            data_patterns=["logs/**/*.json", "logs/**/*.yaml", "logs/**/*.csv"],
        ),
        _profile(
            profile_id="humanoidverse",
            label="HumanoidVerse repo",
            root=humanoidverse_root,
            expected_paths=["humanoidverse", "assets"],
            preferred_entrypoints=["humanoidverse"],
            deploy_patterns=["humanoidverse/run.py", "humanoidverse/**/*.py"],
            policy_patterns=["logs/**/*.pt", "logs/**/*.onnx"],
            data_patterns=["logs/**/*.json", "outputs/**/*"],
        ),
        _profile(
            profile_id="xr_teleoperate",
            label="XR Teleoperate repo",
            root=xr_teleoperate_root,
            expected_paths=["teleop"],
            preferred_entrypoints=["teleop"],
            deploy_patterns=["teleop/teleop_hand_and_arm.py", "teleop/televuer/**/*"],
            policy_patterns=["policies/**/*.onnx", "policies/**/*.pt"],
            data_patterns=["teleop/utils/data/**/*", "teleop/**/*.json"],
        ),
        _profile(
            profile_id="unitree_lerobot",
            label="Unitree LeRobot repo",
            root=unitree_lerobot_root,
            expected_paths=[],
            preferred_entrypoints=["examples", "scripts", "lerobot"],
            deploy_patterns=["examples/**/*.py", "scripts/**/*.py", "lerobot/**/*.py"],
            policy_patterns=[
                "outputs/**/*.onnx",
                "outputs/**/*.pt",
                "checkpoints/**/*",
                "logs/**/*.onnx",
                "logs/**/*.pt",
            ],
            data_patterns=[
                "data/**/*",
                "episodes/**/*",
                "replay/**/*",
                "metrics/**/*.json",
            ],
        ),
        _profile(
            profile_id="unitree_model_assets",
            label="Unitree model/assets repo",
            root=unitree_model_root,
            expected_paths=["README.md"],
            preferred_entrypoints=["README.md"],
            deploy_patterns=["**/*.usd", "**/*.urdf"],
            data_patterns=["**/*.yaml", "**/*.json"],
        ),
    ]
    ready_profiles = [
        profile["profile_id"] for profile in profiles if bool(profile.get("profile_ready", False))
    ]
    return {
        "version": "backend_runtime_layout_contract_v1",
        "backend": "isaac",
        "profiles": profiles,
        "ready_profiles": ready_profiles,
        "preferred_profile_order": [
            "unitree_sim_isaaclab",
            "unitree_rl_gym",
            "unitree_lerobot",
            "humanoidverse",
            "isaaclab_core",
            "xr_teleoperate",
            "unitree_model_assets",
        ],
    }


def describe_isaac_policy_contract(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embodiment = mapping(embodiment_context)
    explicit_policy_root = _context_path(
        embodiment, "isaac_policy_root", "unitree_policy_root", "policy_root"
    ) or _env_path("ISAAC_POLICY_ROOT", "UNITREE_POLICY_ROOT")
    candidate_policy_roots = [
        explicit_policy_root,
        _resolved_path(
            _context_path(embodiment, "unitree_rl_gym_root", "unitree_runtime_repo_root")
            or _env_path("UNITREE_RL_GYM_ROOT"),
            discover_names=("unitree_rl_gym",),
        ),
        _resolved_path(
            _context_path(embodiment, "unitree_sim_isaaclab_root")
            or _env_path("UNITREE_SIM_ISAACLAB_ROOT"),
            discover_names=("unitree_sim_isaaclab",),
        ),
        _resolved_path(
            _context_path(embodiment, "unitree_lerobot_root", "unitree_il_lerobot_root")
            or _env_path("UNITREE_LEROBOT_ROOT", "UNITREE_IL_LEROBOT_ROOT"),
            discover_names=("unitree_IL_lerobot", "unitree_il_lerobot", "unitree_lerobot"),
        ),
        _resolved_path(
            _context_path(embodiment, "humanoidverse_root")
            or _env_path("HUMANOIDVERSE_ROOT"),
            discover_names=("HumanoidVerse", "humanoidverse"),
        ),
        _resolved_path(
            _context_path(embodiment, "xr_teleoperate_root")
            or _env_path("XR_TELEOPERATE_ROOT"),
            discover_names=("xr_teleoperate",),
        ),
    ]
    policy_ref = _context_path(
        embodiment, "isaac_policy_id", "runtime_policy_id", "evaluation_policy_id", "policy_id"
    ) or str(os.environ.get("ISAAC_POLICY_PATH", "") or "").strip()
    policy_root = ""
    checkpoint_candidates: list[str] = []
    deploy_config_candidates: list[str] = []
    runtime_report_candidates: list[str] = []
    checkpoint_candidate_records: list[dict[str, Any]] = []
    deploy_config_candidate_records: list[dict[str, Any]] = []
    runtime_report_candidate_records: list[dict[str, Any]] = []
    for root in candidate_policy_roots:
        if not root:
            continue
        checkpoint_candidate_records = _candidate_records(root, ("*.pt", "*.pth", "*.onnx", "*.ckpt"))
        deploy_config_candidate_records = _candidate_records(root, ("*.yaml", "*.yml", "*.json"))
        runtime_report_candidate_records = _candidate_records(
            root,
            ("logs/**/*.json", "logs/**/*.yaml", "deploy/**/*.yaml", "deploy_real/**/*"),
        )
        checkpoint_candidates = [str(row["ref"]) for row in checkpoint_candidate_records]
        deploy_config_candidates = [str(row["ref"]) for row in deploy_config_candidate_records]
        runtime_report_candidates = [str(row["ref"]) for row in runtime_report_candidate_records]
        if checkpoint_candidate_records or root == explicit_policy_root:
            policy_root = root
            break
    if not policy_root:
        policy_root = explicit_policy_root
    policy_ref_exists = bool(policy_ref and Path(policy_ref).exists())
    return {
        "version": "backend_policy_contract_v1",
        "backend": "isaac",
        "policy_root": policy_root,
        "policy_root_exists": bool(policy_root and Path(policy_root).exists()),
        "policy_ref": policy_ref,
        "policy_ref_exists": policy_ref_exists,
        "checkpoint_candidates": checkpoint_candidates,
        "checkpoint_candidate_records": checkpoint_candidate_records,
        "checkpoint_candidate_count": len(checkpoint_candidate_records),
        "primary_checkpoint_ref": (
            policy_ref if policy_ref_exists else (checkpoint_candidates[0] if checkpoint_candidates else "")
        ),
        "deploy_config_candidates": deploy_config_candidates,
        "deploy_config_candidate_records": deploy_config_candidate_records,
        "deploy_config_candidate_count": len(deploy_config_candidate_records),
        "primary_deploy_config_ref": (
            "" if not deploy_config_candidates else deploy_config_candidates[0]
        ),
        "runtime_report_candidates": runtime_report_candidates,
        "runtime_report_candidate_records": runtime_report_candidate_records,
        "runtime_report_candidate_count": len(runtime_report_candidate_records),
        "primary_runtime_report_ref": (
            "" if not runtime_report_candidates else runtime_report_candidates[0]
        ),
        "policy_ready": bool(policy_ref_exists or checkpoint_candidates),
    }


def describe_holosoma_runtime_layouts(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embodiment = mapping(embodiment_context)
    holosoma_root = _resolved_path(
        _context_path(
        embodiment, "holosoma_root", "holosoma_repo_root"
    ) or _env_path("HOLOSOMA_ROOT", "HOLOSOMA_REPO_ROOT"),
        discover_names=("holosoma",),
    )
    motion_root = _resolved_path(
        _context_path(
        embodiment, "holosoma_motion_root", "motion_data_root"
    ) or _env_path("HOLOSOMA_MOTION_ROOT"),
        discover_names=("holosoma_motion", "motions"),
    )
    policy_root = _resolved_path(
        _context_path(
        embodiment, "holosoma_policy_root", "policy_root"
    ) or _env_path("HOLOSOMA_POLICY_ROOT"),
    )
    retargeting_root = _resolved_path(
        _context_path(
        embodiment, "retargeting_root", "whole_body_retargeting_root"
    ) or _env_path("RETARGETING_ROOT"),
        discover_names=("retargeting",),
    )
    profiles = [
        _profile(
            profile_id="holosoma_repo",
            label="Holosoma runtime repo",
            root=holosoma_root,
            expected_paths=["README.md"],
            preferred_entrypoints=["README.md"],
            deploy_patterns=["**/*.py", "scripts/**/*"],
            policy_patterns=["checkpoints/**/*", "**/*.onnx", "**/*.pt"],
            data_patterns=["logs/**/*.json", "outputs/**/*", "runs/**/*"],
        ),
        _profile(
            profile_id="holosoma_motion_bank",
            label="Holosoma motion corpus",
            root=motion_root,
            expected_paths=[],
            preferred_entrypoints=[],
            data_patterns=["**/*.npz", "**/*.npy", "**/*.bvh", "**/*.pkl"],
        ),
        _profile(
            profile_id="holosoma_policy_bank",
            label="Holosoma policy bank",
            root=policy_root,
            expected_paths=[],
            preferred_entrypoints=[],
            policy_patterns=["**/*.onnx", "**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.yaml"],
        ),
        _profile(
            profile_id="retargeting_bundle",
            label="Whole-body retargeting bundle",
            root=retargeting_root,
            expected_paths=[],
            preferred_entrypoints=[],
            data_patterns=["**/*.yaml", "**/*.json", "**/*.npz"],
        ),
    ]
    ready_profiles = [
        profile["profile_id"] for profile in profiles if bool(profile.get("root_exists", False))
    ]
    return {
        "version": "backend_runtime_layout_contract_v1",
        "backend": "holosoma",
        "profiles": profiles,
        "ready_profiles": ready_profiles,
        "preferred_profile_order": [
            "holosoma_repo",
            "holosoma_motion_bank",
            "holosoma_policy_bank",
            "retargeting_bundle",
        ],
    }


def describe_holosoma_policy_contract(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embodiment = mapping(embodiment_context)
    explicit_policy_root = _context_path(
        embodiment, "holosoma_policy_root", "policy_root"
    ) or _env_path("HOLOSOMA_POLICY_ROOT")
    candidate_policy_roots = [
        explicit_policy_root,
        _resolved_path(
            _context_path(embodiment, "holosoma_root", "holosoma_repo_root")
            or _env_path("HOLOSOMA_ROOT", "HOLOSOMA_REPO_ROOT"),
            discover_names=("holosoma",),
        ),
    ]
    policy_ref = _context_path(
        embodiment, "holosoma_policy_id", "runtime_policy_id", "evaluation_policy_id", "policy_id"
    ) or str(os.environ.get("HOLOSOMA_POLICY_PATH", "") or "").strip()
    policy_root = ""
    checkpoint_candidates: list[str] = []
    deploy_config_candidates: list[str] = []
    runtime_report_candidates: list[str] = []
    checkpoint_candidate_records: list[dict[str, Any]] = []
    deploy_config_candidate_records: list[dict[str, Any]] = []
    runtime_report_candidate_records: list[dict[str, Any]] = []
    for root in candidate_policy_roots:
        if not root:
            continue
        checkpoint_candidate_records = _candidate_records(root, ("*.pt", "*.pth", "*.onnx", "*.ckpt"))
        deploy_config_candidate_records = _candidate_records(root, ("*.yaml", "*.yml", "*.json"))
        runtime_report_candidate_records = _candidate_records(
            root, ("logs/**/*.json", "metrics/**/*.json", "outputs/**/*.json")
        )
        checkpoint_candidates = [str(row["ref"]) for row in checkpoint_candidate_records]
        deploy_config_candidates = [str(row["ref"]) for row in deploy_config_candidate_records]
        runtime_report_candidates = [str(row["ref"]) for row in runtime_report_candidate_records]
        if checkpoint_candidate_records or root == explicit_policy_root:
            policy_root = root
            break
    if not policy_root:
        policy_root = explicit_policy_root
    policy_ref_exists = bool(policy_ref and Path(policy_ref).exists())
    return {
        "version": "backend_policy_contract_v1",
        "backend": "holosoma",
        "policy_root": policy_root,
        "policy_root_exists": bool(policy_root and Path(policy_root).exists()),
        "policy_ref": policy_ref,
        "policy_ref_exists": policy_ref_exists,
        "checkpoint_candidates": checkpoint_candidates,
        "checkpoint_candidate_records": checkpoint_candidate_records,
        "checkpoint_candidate_count": len(checkpoint_candidate_records),
        "primary_checkpoint_ref": (
            policy_ref if policy_ref_exists else (checkpoint_candidates[0] if checkpoint_candidates else "")
        ),
        "deploy_config_candidates": deploy_config_candidates,
        "deploy_config_candidate_records": deploy_config_candidate_records,
        "deploy_config_candidate_count": len(deploy_config_candidate_records),
        "primary_deploy_config_ref": (
            "" if not deploy_config_candidates else deploy_config_candidates[0]
        ),
        "runtime_report_candidates": runtime_report_candidates,
        "runtime_report_candidate_records": runtime_report_candidate_records,
        "runtime_report_candidate_count": len(runtime_report_candidate_records),
        "primary_runtime_report_ref": (
            "" if not runtime_report_candidates else runtime_report_candidates[0]
        ),
        "policy_ready": bool(policy_ref_exists or checkpoint_candidates),
    }


__all__ = [
    "describe_holosoma_policy_contract",
    "describe_holosoma_runtime_layouts",
    "describe_isaac_policy_contract",
    "describe_isaac_runtime_layouts",
]

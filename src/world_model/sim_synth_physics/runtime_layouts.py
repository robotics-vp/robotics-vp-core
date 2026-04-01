"""OSS-informed runtime layout and policy contracts for Phase-1 backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import mapping


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
    return {
        "profile_id": profile_id,
        "label": label,
        "root": root,
        "root_exists": bool(root and Path(root).exists()),
        "expected_paths": list(expected_paths),
        "matched_paths": matched_paths,
        "missing_paths": missing_paths,
        "preferred_entrypoints": list(preferred_entrypoints),
        "deploy_candidates": _candidate_files(root, deploy_patterns or []),
        "policy_candidates": _candidate_files(root, policy_patterns or []),
        "data_candidates": _candidate_files(root, data_patterns or []),
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


def describe_isaac_runtime_layouts(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embodiment = mapping(embodiment_context)
    isaaclab_root = _context_path(
        embodiment, "isaaclab_root", "isaac_lab_root", "isaac_repo_root"
    ) or _env_path("ISAACLAB_ROOT", "ISAAC_LAB_ROOT")
    unitree_sim_isaaclab_root = _context_path(
        embodiment, "unitree_sim_isaaclab_root"
    ) or _env_path("UNITREE_SIM_ISAACLAB_ROOT")
    unitree_rl_gym_root = _context_path(
        embodiment, "unitree_rl_gym_root", "unitree_runtime_repo_root"
    ) or _env_path("UNITREE_RL_GYM_ROOT")
    humanoidverse_root = _context_path(
        embodiment, "humanoidverse_root"
    ) or _env_path("HUMANOIDVERSE_ROOT")
    xr_teleoperate_root = _context_path(
        embodiment, "xr_teleoperate_root"
    ) or _env_path("XR_TELEOPERATE_ROOT")
    unitree_lerobot_root = _context_path(
        embodiment, "unitree_lerobot_root", "unitree_il_lerobot_root"
    ) or _env_path("UNITREE_LEROBOT_ROOT", "UNITREE_IL_LEROBOT_ROOT")
    unitree_model_root = _context_path(
        embodiment, "unitree_model_root"
    ) or _env_path("UNITREE_MODEL_ROOT")
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
    policy_root = _context_path(
        embodiment, "isaac_policy_root", "unitree_policy_root", "policy_root"
    ) or _env_path("ISAAC_POLICY_ROOT", "UNITREE_POLICY_ROOT")
    policy_ref = _context_path(
        embodiment, "isaac_policy_id", "runtime_policy_id", "evaluation_policy_id", "policy_id"
    ) or str(os.environ.get("ISAAC_POLICY_PATH", "") or "").strip()
    checkpoint_candidates = _candidate_files(policy_root, ("*.pt", "*.pth", "*.onnx", "*.ckpt"))
    deploy_config_candidates = _candidate_files(policy_root, ("*.yaml", "*.yml", "*.json"))
    runtime_report_candidates = _candidate_files(
        policy_root,
        ("logs/**/*.json", "logs/**/*.yaml", "deploy/**/*.yaml", "deploy_real/**/*"),
    )
    policy_ref_exists = bool(policy_ref and Path(policy_ref).exists())
    return {
        "version": "backend_policy_contract_v1",
        "backend": "isaac",
        "policy_root": policy_root,
        "policy_root_exists": bool(policy_root and Path(policy_root).exists()),
        "policy_ref": policy_ref,
        "policy_ref_exists": policy_ref_exists,
        "checkpoint_candidates": checkpoint_candidates,
        "deploy_config_candidates": deploy_config_candidates,
        "runtime_report_candidates": runtime_report_candidates,
        "policy_ready": bool(policy_ref_exists or checkpoint_candidates),
    }


def describe_holosoma_runtime_layouts(
    embodiment_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    embodiment = mapping(embodiment_context)
    holosoma_root = _context_path(
        embodiment, "holosoma_root", "holosoma_repo_root"
    ) or _env_path("HOLOSOMA_ROOT", "HOLOSOMA_REPO_ROOT")
    motion_root = _context_path(
        embodiment, "holosoma_motion_root", "motion_data_root"
    ) or _env_path("HOLOSOMA_MOTION_ROOT")
    policy_root = _context_path(
        embodiment, "holosoma_policy_root", "policy_root"
    ) or _env_path("HOLOSOMA_POLICY_ROOT")
    retargeting_root = _context_path(
        embodiment, "retargeting_root", "whole_body_retargeting_root"
    ) or _env_path("RETARGETING_ROOT")
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
    policy_root = _context_path(
        embodiment, "holosoma_policy_root", "policy_root"
    ) or _env_path("HOLOSOMA_POLICY_ROOT")
    policy_ref = _context_path(
        embodiment, "holosoma_policy_id", "runtime_policy_id", "evaluation_policy_id", "policy_id"
    ) or str(os.environ.get("HOLOSOMA_POLICY_PATH", "") or "").strip()
    checkpoint_candidates = _candidate_files(policy_root, ("*.pt", "*.pth", "*.onnx", "*.ckpt"))
    deploy_config_candidates = _candidate_files(policy_root, ("*.yaml", "*.yml", "*.json"))
    policy_ref_exists = bool(policy_ref and Path(policy_ref).exists())
    return {
        "version": "backend_policy_contract_v1",
        "backend": "holosoma",
        "policy_root": policy_root,
        "policy_root_exists": bool(policy_root and Path(policy_root).exists()),
        "policy_ref": policy_ref,
        "policy_ref_exists": policy_ref_exists,
        "checkpoint_candidates": checkpoint_candidates,
        "deploy_config_candidates": deploy_config_candidates,
        "policy_ready": bool(policy_ref_exists or checkpoint_candidates),
    }


__all__ = [
    "describe_holosoma_policy_contract",
    "describe_holosoma_runtime_layouts",
    "describe_isaac_policy_contract",
    "describe_isaac_runtime_layouts",
]

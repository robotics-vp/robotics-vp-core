"""WM-owned runtime bundles and launch specs for Phase-1 backend bring-up."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from .adapters.holosoma_executable_adapter import (
    build_holosoma_executable_adapter_request,
)
from .adapters.holosoma_executable_consumer import (
    build_holosoma_executable_adapter_consumer,
)
from .adapters.isaac_unitree_executable_adapter import (
    build_isaac_unitree_executable_adapter_request,
)
from .adapters.isaac_unitree_executable_consumer import (
    build_isaac_unitree_executable_adapter_consumer,
)
from .common import mapping, strings
from .runtime_outcomes import build_backend_runtime_output_contract


ISAAC_PROFILE_TO_TARGET_IDS = {
    "isaaclab_core": ["isaaclab_root", "isaacsim_root"],
    "unitree_sim_isaaclab": ["unitree_sim_isaaclab_root"],
    "unitree_rl_gym": ["unitree_rl_gym_root"],
    "unitree_lerobot": ["unitree_il_lerobot_root"],
    "humanoidverse": ["humanoidverse_root"],
    "xr_teleoperate": ["xr_teleoperate_root"],
    "unitree_model_assets": ["unitree_model_root", "unitree_asset_root"],
}
HOLOSOMA_PROFILE_TO_TARGET_IDS = {
    "holosoma_repo": ["holosoma_root"],
    "holosoma_motion_bank": ["holosoma_motion_root"],
    "holosoma_policy_bank": ["holosoma_policy_root"],
    "retargeting_bundle": ["retargeting_root"],
}
ISAAC_PROFILE_COMMANDS = {
    "unitree_sim_isaaclab": (
        "python ${UNITREE_SIM_ISAACLAB_ROOT}/sim_main.py "
        "--task {task_id} --policy {policy_ref} --headless"
    ),
    "unitree_rl_gym": (
        "python ${UNITREE_RL_GYM_ROOT}/deploy/deploy.py "
        "--task {task_id} --checkpoint {policy_ref}"
    ),
    "unitree_lerobot": (
        "python ${UNITREE_IL_LEROBOT_ROOT}/examples/eval_policy.py "
        "--task {task_id} --policy {policy_ref}"
    ),
    "humanoidverse": (
        "python ${HUMANOIDVERSE_ROOT}/humanoidverse/run.py "
        "--task {task_id} --checkpoint {policy_ref}"
    ),
    "isaaclab_core": (
        "python ${ISAACLAB_ROOT}/source/standalone/workflows/rl/play.py "
        "--task {task_id} --checkpoint {policy_ref}"
    ),
    "xr_teleoperate": (
        "python ${XR_TELEOPERATE_ROOT}/teleop/run_teleop.py "
        "--task {task_id} --policy {policy_ref}"
    ),
}
HOLOSOMA_PROFILE_COMMANDS = {
    "holosoma_repo": (
        "python -m holosoma.eval --task-id {task_id} --policy {policy_ref}"
    ),
    "holosoma_motion_bank": (
        "python scripts/local_holosoma_smoke.py "
        "--task-id {task_id} --policy-id {policy_ref} --episodes 1"
    ),
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _target_ref(runtime_target_contract: Mapping[str, Any], target_id: str) -> str:
    for row in list(runtime_target_contract.get("targets", []) or []):
        row_mapping = mapping(row)
        if str(row_mapping.get("target_id", "")) == target_id:
            return str(row_mapping.get("ref", "") or "")
    return ""


def _profile_root(
    runtime_target_contract: Mapping[str, Any],
    target_profile_map: Mapping[str, list[str]],
    profile_id: str,
) -> str:
    for target_id in target_profile_map.get(profile_id, []):
        ref = _target_ref(runtime_target_contract, target_id)
        if ref:
            return ref
    return ""


def _ready_profiles(
    *,
    runtime_target_contract: Mapping[str, Any],
    runtime_layout_contract: Mapping[str, Any],
    target_profile_map: Mapping[str, list[str]],
) -> list[str]:
    profiles = strings(runtime_layout_contract.get("ready_profiles"))
    ready_targets = set(strings(runtime_target_contract.get("ready_target_ids")))
    for profile_id, target_ids in target_profile_map.items():
        if any(target_id in ready_targets for target_id in target_ids):
            if profile_id not in profiles:
                profiles.append(profile_id)
    return profiles


def _preferred_profile(
    *,
    runtime_target_contract: Mapping[str, Any],
    runtime_layout_contract: Mapping[str, Any],
    target_profile_map: Mapping[str, list[str]],
    deployment_contract: Mapping[str, Any] | None = None,
) -> str:
    ready_profiles = _ready_profiles(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        target_profile_map=target_profile_map,
    )
    preferred_order = strings(
        mapping(deployment_contract).get("preferred_profile_order")
    ) or strings(runtime_layout_contract.get("preferred_profile_order"))
    for profile_id in preferred_order:
        if profile_id in ready_profiles:
            return profile_id
    return ready_profiles[0] if ready_profiles else ""


def _format_command(template: str, *, task_id: str, policy_ref: str) -> str:
    command = template.replace("{task_id}", task_id or "unknown_task")
    return command.replace("{policy_ref}", policy_ref or "${POLICY_REF}")


def _launch_specs_for_backend(
    *,
    backend: str,
    task_id: str,
    policy_ref: str,
    runtime_target_contract: Mapping[str, Any],
    runtime_layout_contract: Mapping[str, Any],
    deployment_contract: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if backend == "isaac":
        target_profile_map = ISAAC_PROFILE_TO_TARGET_IDS
        command_templates = ISAAC_PROFILE_COMMANDS
        upstream_profiles = {
            "unitree_sim_isaaclab": {
                "repo": "unitree_sim_isaaclab",
                "url": "https://github.com/unitreerobotics/unitree_sim_isaaclab",
            },
            "unitree_rl_gym": {
                "repo": "unitree_rl_gym",
                "url": "https://github.com/unitreerobotics/unitree_rl_gym",
            },
            "unitree_lerobot": {
                "repo": "unitree_IL_lerobot",
                "url": "https://github.com/unitreerobotics/unitree_IL_lerobot",
            },
            "humanoidverse": {
                "repo": "HumanoidVerse",
                "url": "https://github.com/LeCAR-Lab/HumanoidVerse",
            },
            "isaaclab_core": {
                "repo": "IsaacLab",
                "url": "https://isaac-sim.github.io/IsaacLab",
            },
            "xr_teleoperate": {
                "repo": "xr_teleoperate",
                "url": "https://github.com/unitreerobotics/xr_teleoperate",
            },
        }
    else:
        target_profile_map = HOLOSOMA_PROFILE_TO_TARGET_IDS
        command_templates = HOLOSOMA_PROFILE_COMMANDS
        upstream_profiles = {
            "holosoma_repo": {"repo": "holosoma", "url": "https://pypi.org/project/holosoma/"},
        }
    ready_profiles = _ready_profiles(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        target_profile_map=target_profile_map,
    )
    preferred_order = strings(
        mapping(deployment_contract).get("preferred_profile_order")
    ) or strings(runtime_layout_contract.get("preferred_profile_order"))
    ordered_profiles = [
        profile_id for profile_id in preferred_order if profile_id in ready_profiles
    ] + [profile_id for profile_id in ready_profiles if profile_id not in preferred_order]
    specs: list[dict[str, Any]] = []
    for profile_id in ordered_profiles:
        command_template = command_templates.get(profile_id, "")
        if not command_template:
            continue
        specs.append(
            {
                "profile_id": profile_id,
                "root": _profile_root(
                    runtime_target_contract,
                    target_profile_map,
                    profile_id,
                ),
                "command": _format_command(
                    command_template,
                    task_id=task_id,
                    policy_ref=policy_ref,
                ),
                "upstream_profile": mapping(upstream_profiles.get(profile_id)),
            }
        )
    return specs


def build_backend_runtime_bundle(
    *,
    backend: str,
    task_id: str,
    policy_ref: str,
    runtime_target_contract: Mapping[str, Any],
    runtime_layout_contract: Mapping[str, Any],
    policy_contract: Mapping[str, Any],
    robot_asset_manifest: Mapping[str, Any],
    normalized_robot_asset_manifest: Mapping[str, Any],
    robot_contract_context: Mapping[str, Any] | None = None,
    deployment_contract: Mapping[str, Any] | None = None,
    output_root: Optional[Path],
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    target_profile_map = (
        ISAAC_PROFILE_TO_TARGET_IDS if backend == "isaac" else HOLOSOMA_PROFILE_TO_TARGET_IDS
    )
    preferred_profile = _preferred_profile(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        target_profile_map=target_profile_map,
        deployment_contract=deployment_contract,
    )
    launch_specs = _launch_specs_for_backend(
        backend=backend,
        task_id=task_id,
        policy_ref=policy_ref,
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        deployment_contract=deployment_contract,
    )
    preferred_launch_spec = next(
        (spec for spec in launch_specs if spec.get("profile_id") == preferred_profile),
        (launch_specs[0] if launch_specs else {}),
    )
    runtime_bundle = {
        "version": "backend_runtime_bundle_v1",
        "backend": backend,
        "task_id": task_id,
        "preferred_profile": preferred_profile,
        "ready_profiles": _ready_profiles(
            runtime_target_contract=runtime_target_contract,
            runtime_layout_contract=runtime_layout_contract,
            target_profile_map=target_profile_map,
        ),
        "runtime_target_contract": mapping(runtime_target_contract),
        "runtime_layout_contract": mapping(runtime_layout_contract),
        "policy_contract": mapping(policy_contract),
        "deployment_contract": mapping(deployment_contract),
        "robot_asset_manifest": mapping(robot_asset_manifest),
        "normalized_robot_asset_manifest": mapping(normalized_robot_asset_manifest),
        "robot_contract_context": mapping(robot_contract_context),
        "launch_specs": list(launch_specs),
    }
    output_contract = build_backend_runtime_output_contract(runtime_bundle, preferred_launch_spec)
    runtime_bundle["output_contract"] = output_contract
    executable_adapter_request: dict[str, Any] = {}
    executable_adapter_consumer: dict[str, Any] = {}
    if backend == "isaac":
        executable_adapter_request = build_isaac_unitree_executable_adapter_request(
            task_id=task_id,
            policy_ref=policy_ref,
            preferred_profile=preferred_profile,
            launch_spec=preferred_launch_spec,
            runtime_target_contract=runtime_target_contract,
            deployment_contract=mapping(deployment_contract),
            normalized_robot_asset_manifest=normalized_robot_asset_manifest,
            robot_contract_context=mapping(robot_contract_context),
            output_contract=output_contract,
        )
        executable_adapter_consumer = build_isaac_unitree_executable_adapter_consumer(
            executable_adapter_request
        )
        runtime_bundle["executable_adapter_request"] = executable_adapter_request
        runtime_bundle["executable_adapter_consumer"] = executable_adapter_consumer
    elif backend == "holosoma":
        executable_adapter_request = build_holosoma_executable_adapter_request(
            task_id=task_id,
            policy_ref=policy_ref,
            preferred_profile=preferred_profile,
            launch_spec=preferred_launch_spec,
            runtime_target_contract=runtime_target_contract,
            policy_contract=policy_contract,
            normalized_robot_asset_manifest=normalized_robot_asset_manifest,
            robot_contract_context=mapping(robot_contract_context),
            output_contract=output_contract,
        )
        executable_adapter_consumer = build_holosoma_executable_adapter_consumer(
            executable_adapter_request
        )
        runtime_bundle["executable_adapter_request"] = executable_adapter_request
        runtime_bundle["executable_adapter_consumer"] = executable_adapter_consumer
    launch_spec = {
        "version": "backend_launch_spec_v1",
        "backend": backend,
        "task_id": task_id,
        "preferred_profile": preferred_profile,
        "policy_ref": policy_ref,
        "policy_ready": bool(policy_contract.get("policy_ready", False)),
        "runtime_targets_ready": bool(runtime_target_contract.get("runtime_targets_ready", False)),
        "deployment_contract": mapping(deployment_contract),
        "command": str(mapping(preferred_launch_spec).get("command", "") or ""),
        "root": str(mapping(preferred_launch_spec).get("root", "") or ""),
        "upstream_profile": mapping(
            mapping(preferred_launch_spec).get("upstream_profile")
        ),
        "executable_adapter_request": executable_adapter_request,
        "executable_adapter_consumer": executable_adapter_consumer,
        "alternative_launch_specs": list(launch_specs),
        "output_contract": output_contract,
    }
    refs: list[str] = []
    if output_root is not None:
        bundle_path = output_root / "backend_runtime_bundle.json"
        launch_spec_path = output_root / "backend_launch_spec.json"
        _write_json(bundle_path, runtime_bundle)
        _write_json(launch_spec_path, launch_spec)
        refs.extend([str(bundle_path.resolve()), str(launch_spec_path.resolve())])
    return refs, runtime_bundle, launch_spec


__all__ = ["build_backend_runtime_bundle"]

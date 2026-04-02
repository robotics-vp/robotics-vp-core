"""Holosoma-aware deployment contracts for the sim/synth/physics WM."""

from __future__ import annotations

from typing import Any, Mapping

from ..common import mapping, strings


def _verified_targets(runtime_target_contract: Mapping[str, Any]) -> set[str]:
    verified = strings(runtime_target_contract.get("verified_target_ids"))
    if verified:
        return set(verified)
    return set(strings(runtime_target_contract.get("ready_target_ids")))


def _profile_by_id(
    runtime_layout_contract: Mapping[str, Any],
    profile_id: str,
) -> dict[str, Any]:
    for row in list(runtime_layout_contract.get("profiles", []) or []):
        row_mapping = mapping(row)
        if str(row_mapping.get("profile_id", "") or "") == profile_id:
            return row_mapping
    return {}


def _has_motion_sources(
    embodiment_context: Mapping[str, Any],
    *,
    runtime_target_contract: Mapping[str, Any],
    runtime_layout_contract: Mapping[str, Any],
) -> bool:
    embodiment = mapping(embodiment_context)
    if bool(
        embodiment.get("motion_clip_datapacks")
        or embodiment.get("motion_clips")
        or embodiment.get("motion_clip_paths")
    ):
        return True
    if "holosoma_motion_root" in _verified_targets(runtime_target_contract):
        return True
    motion_profile = _profile_by_id(runtime_layout_contract, "holosoma_motion_bank")
    return bool(strings(motion_profile.get("data_candidates")))


def _has_retargeting_contract(
    embodiment_context: Mapping[str, Any],
    *,
    runtime_target_contract: Mapping[str, Any],
    runtime_layout_contract: Mapping[str, Any],
) -> bool:
    embodiment = mapping(embodiment_context)
    if bool(
        embodiment.get("retargeting_contract")
        or embodiment.get("whole_body_retargeting")
        or embodiment.get("retargeting_root")
    ):
        return True
    if "retargeting_root" in _verified_targets(runtime_target_contract):
        return True
    retarget_profile = _profile_by_id(runtime_layout_contract, "retargeting_bundle")
    return bool(strings(retarget_profile.get("data_candidates")))


def _has_reward_overlay(embodiment_context: Mapping[str, Any]) -> bool:
    embodiment = mapping(embodiment_context)
    return bool(embodiment.get("whole_body_reward_overlay"))


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


def _mode_contract(
    *,
    mode_id: str,
    label: str,
    profile_candidates: list[str],
    required_target_ids: list[str],
    policy_required: bool,
    motion_required: bool,
    retargeting_required: bool,
    reward_overlay_required: bool,
    ready_profiles: list[str],
    ready_targets: set[str],
    policy_ready: bool,
    embodiment_context: Mapping[str, Any],
    runtime_layout_contract: Mapping[str, Any],
) -> dict[str, Any]:
    missing_preconditions: list[str] = []
    if not any(profile in ready_profiles for profile in profile_candidates):
        missing_preconditions.append("runtime_profile")
    missing_preconditions.extend(
        target_id for target_id in required_target_ids if target_id not in ready_targets
    )
    if policy_required and not policy_ready:
        missing_preconditions.append("policy_checkpoint")
    if motion_required and not _has_motion_sources(
        embodiment_context,
        runtime_target_contract={"verified_target_ids": sorted(ready_targets)},
        runtime_layout_contract=runtime_layout_contract,
    ):
        missing_preconditions.append("motion_source_bundle")
    if retargeting_required and not _has_retargeting_contract(
        embodiment_context,
        runtime_target_contract={"verified_target_ids": sorted(ready_targets)},
        runtime_layout_contract=runtime_layout_contract,
    ):
        missing_preconditions.append("whole_body_retargeting_contract")
    if reward_overlay_required and not _has_reward_overlay(embodiment_context):
        missing_preconditions.append("whole_body_reward_overlay")
    return {
        "mode_id": mode_id,
        "label": label,
        "profile_candidates": list(profile_candidates),
        "required_target_ids": list(required_target_ids),
        "ready": not missing_preconditions,
        "missing_preconditions": missing_preconditions,
    }


def build_holosoma_deployment_contract(
    *,
    embodiment_context: Mapping[str, Any],
    runtime_target_contract: Mapping[str, Any],
    runtime_layout_contract: Mapping[str, Any],
    policy_contract: Mapping[str, Any],
) -> dict[str, Any]:
    ready_profiles = _usable_profiles(runtime_layout_contract)
    ready_targets = _verified_targets(runtime_target_contract)
    policy_ready = bool(policy_contract.get("policy_ready", False))
    deployment_modes = [
        _mode_contract(
            mode_id="sim_eval",
            label="Holosoma sim evaluation",
            profile_candidates=["holosoma_repo", "holosoma_policy_bank"],
            required_target_ids=["holosoma_motion_root"],
            policy_required=True,
            motion_required=False,
            retargeting_required=False,
            reward_overlay_required=False,
            ready_profiles=ready_profiles,
            ready_targets=ready_targets,
            policy_ready=policy_ready,
            embodiment_context=embodiment_context,
            runtime_layout_contract=runtime_layout_contract,
        ),
        _mode_contract(
            mode_id="motion_train",
            label="Holosoma motion-bank training",
            profile_candidates=["holosoma_motion_bank", "holosoma_repo"],
            required_target_ids=["holosoma_motion_root"],
            policy_required=False,
            motion_required=True,
            retargeting_required=False,
            reward_overlay_required=False,
            ready_profiles=ready_profiles,
            ready_targets=ready_targets,
            policy_ready=policy_ready,
            embodiment_context=embodiment_context,
            runtime_layout_contract=runtime_layout_contract,
        ),
        _mode_contract(
            mode_id="retarget_eval",
            label="Holosoma retargeted whole-body evaluation",
            profile_candidates=["retargeting_bundle", "holosoma_repo"],
            required_target_ids=["holosoma_motion_root", "retargeting_root"],
            policy_required=True,
            motion_required=True,
            retargeting_required=True,
            reward_overlay_required=False,
            ready_profiles=ready_profiles,
            ready_targets=ready_targets,
            policy_ready=policy_ready,
            embodiment_context=embodiment_context,
            runtime_layout_contract=runtime_layout_contract,
        ),
    ]
    ready_modes = [row["mode_id"] for row in deployment_modes if bool(row.get("ready", False))]
    preferred_profile_order = [
        "holosoma_repo",
        "holosoma_motion_bank",
        "holosoma_policy_bank",
        "retargeting_bundle",
    ]
    preferred_profile = next(
        (profile_id for profile_id in preferred_profile_order if profile_id in ready_profiles),
        (ready_profiles[0] if ready_profiles else ""),
    )
    return {
        "version": "holosoma_deployment_contract_v1",
        "ready_profiles": ready_profiles,
        "verified_target_ids": sorted(ready_targets),
        "runtime_target_preflight_status": str(
            runtime_target_contract.get("runtime_target_preflight_status", "") or ""
        ),
        "preferred_profile_order": preferred_profile_order,
        "preferred_profile": preferred_profile,
        "policy_ready": policy_ready,
        "ready_modes": ready_modes,
        "sim_launch_ready": "sim_eval" in ready_modes,
        "motion_train_ready": "motion_train" in ready_modes,
        "retarget_eval_ready": "retarget_eval" in ready_modes,
        "deployment_modes": deployment_modes,
    }


__all__ = ["build_holosoma_deployment_contract"]

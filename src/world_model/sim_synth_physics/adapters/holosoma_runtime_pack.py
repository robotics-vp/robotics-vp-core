"""Upstream runtime pack contract for Holosoma Phase-1 execution."""

from __future__ import annotations

from typing import Any, Mapping

from ..common import mapping, stable_id, strings


def _profile_row(runtime_layout_contract: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    for row in list(runtime_layout_contract.get("profiles", []) or []):
        row_mapping = mapping(row)
        if str(row_mapping.get("profile_id", "") or "") == profile_id:
            return row_mapping
    return {}


def build_holosoma_runtime_pack(
    *,
    runtime_target_contract: Mapping[str, Any],
    runtime_layout_contract: Mapping[str, Any],
    policy_contract: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    embodiment_context: Mapping[str, Any],
) -> dict[str, Any]:
    deployment = mapping(deployment_contract)
    preferred_profile = str(
        deployment.get("preferred_profile")
        or runtime_layout_contract.get("preferred_profile")
        or ""
    )
    profile = _profile_row(runtime_layout_contract, preferred_profile)
    ready_modes = strings(deployment.get("ready_modes"))
    ready_targets = strings(runtime_target_contract.get("ready_target_ids"))
    checkpoint_candidates = strings(policy_contract.get("checkpoint_candidates"))
    deploy_config_candidates = strings(policy_contract.get("deploy_config_candidates"))
    runtime_report_candidates = strings(policy_contract.get("runtime_report_candidates"))
    embodiment = mapping(embodiment_context)
    motion_sources = strings(embodiment.get("motion_clip_datapacks")) + strings(
        embodiment.get("motion_clip_paths")
    )
    if embodiment.get("motion_clips"):
        motion_sources.append("inline_motion_clips")
    retargeting_present = bool(
        embodiment.get("retargeting_contract")
        or embodiment.get("whole_body_retargeting")
        or embodiment.get("retargeting_root")
    )
    reward_overlay_present = bool(embodiment.get("whole_body_reward_overlay"))

    ready_surfaces: list[str] = []
    if preferred_profile:
        ready_surfaces.append("runtime_profile_surface")
    if ready_targets:
        ready_surfaces.append("runtime_target_surface")
    if checkpoint_candidates:
        ready_surfaces.append("policy_surface")
    if motion_sources or "motion_train" in ready_modes:
        ready_surfaces.append("motion_surface")
    if retargeting_present:
        ready_surfaces.append("retargeting_surface")
    if reward_overlay_present:
        ready_surfaces.append("reward_overlay_surface")
    if runtime_report_candidates or strings(profile.get("data_candidates")):
        ready_surfaces.append("telemetry_surface")

    missing_components: list[str] = []
    if not preferred_profile:
        missing_components.append("preferred_runtime_profile")
    if not ready_modes:
        missing_components.append("deployment_mode")
    if not ready_targets:
        missing_components.append("runtime_targets")
    if not checkpoint_candidates and "motion_train" not in ready_modes:
        missing_components.append("policy_checkpoint")
    if not motion_sources and "motion_train" not in ready_modes:
        missing_components.append("motion_sources")

    pack_status = "pack_ready"
    if ready_surfaces and missing_components:
        pack_status = "pack_partial"
    elif not ready_surfaces:
        pack_status = "pack_blocked"

    payload = {
        "backend": "holosoma",
        "preferred_profile": preferred_profile,
        "pack_status": pack_status,
        "ready_modes": ready_modes,
        "ready_surfaces": ready_surfaces,
    }
    return {
        "version": "backend_upstream_runtime_pack_v1",
        "pack_id": stable_id("backend_upstream_runtime_pack", payload),
        **payload,
        "runtime_target_ids": ready_targets,
        "profile_root": str(profile.get("root", "") or ""),
        "profile_matched_paths": strings(profile.get("matched_paths")),
        "deploy_candidates": strings(profile.get("deploy_candidates")) or deploy_config_candidates,
        "policy_candidates": strings(profile.get("policy_candidates")) or checkpoint_candidates,
        "data_candidates": strings(profile.get("data_candidates")) or runtime_report_candidates,
        "checkpoint_candidates": checkpoint_candidates,
        "runtime_report_candidates": runtime_report_candidates,
        "motion_sources": motion_sources,
        "retargeting_present": retargeting_present,
        "reward_overlay_present": reward_overlay_present,
        "missing_components": missing_components,
        "notes": [
            "Upstream runtime pack makes Holosoma host/runtime/motion/policy/retargeting surfaces explicit.",
            "This pack keeps train-from-motion and eval-with-policy as distinct bounded modes.",
        ],
    }


__all__ = ["build_holosoma_runtime_pack"]

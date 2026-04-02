"""Upstream runtime pack contract for Holosoma Phase-1 execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..common import mapping, stable_id, strings
from ..ref_evidence import select_best_named_ref


def _profile_row(runtime_layout_contract: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    for row in list(runtime_layout_contract.get("profiles", []) or []):
        row_mapping = mapping(row)
        if str(row_mapping.get("profile_id", "") or "") == profile_id:
            return row_mapping
    return {}


def _fallback_preferred_profile(runtime_layout_contract: Mapping[str, Any]) -> str:
    profiles = [
        mapping(row) for row in list(runtime_layout_contract.get("profiles", []) or [])
    ]
    by_id = {
        str(row.get("profile_id", "") or ""): row
        for row in profiles
        if str(row.get("profile_id", "") or "")
    }
    for profile_id in strings(runtime_layout_contract.get("preferred_profile_order")):
        row = by_id.get(profile_id, {})
        if bool(row.get("root_exists", False)) and str(
            row.get("install_preflight_status", "") or ""
        ) != "install_blocked":
            return str(profile_id)
    return ""


def _profile_install_by_id(runtime_layout_contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for row in list(runtime_layout_contract.get("profiles", []) or []):
        row_mapping = mapping(row)
        profile_id = str(row_mapping.get("profile_id", "") or "")
        if not profile_id:
            continue
        payload[profile_id] = {
            "install_preflight_status": str(
                row_mapping.get("install_preflight_status", "") or ""
            ),
            "install_missing_components": strings(
                row_mapping.get("install_missing_components")
            ),
            "install_verified_components": strings(
                row_mapping.get("install_verified_components")
            ),
            "primary_entrypoint_ref": str(
                row_mapping.get("primary_entrypoint_ref", "") or ""
            ),
        }
    return payload


def _motion_source_exists(ref: str) -> bool:
    if ref == "inline_motion_clips":
        return True
    path = Path(ref)
    if path.exists():
        return True
    return "/" not in ref and "\\" not in ref and not path.suffix


def _named_candidates(prefix: str, refs: list[str]) -> list[tuple[str, str]]:
    return [(f"{prefix}[{index}]", ref) for index, ref in enumerate(refs) if str(ref)]


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
        or _fallback_preferred_profile(runtime_layout_contract)
        or ""
    )
    profile = _profile_row(runtime_layout_contract, preferred_profile)
    profile_install_by_id = _profile_install_by_id(runtime_layout_contract)
    ready_modes = strings(deployment.get("ready_modes"))
    ready_targets = strings(runtime_target_contract.get("verified_target_ids")) or strings(
        runtime_target_contract.get("ready_target_ids")
    )
    profile_policy_candidates = strings(profile.get("policy_candidates"))
    profile_deploy_candidates = strings(profile.get("deploy_candidates"))
    profile_data_candidates = strings(profile.get("data_candidates"))
    checkpoint_candidates = list(
        dict.fromkeys(profile_policy_candidates + strings(policy_contract.get("checkpoint_candidates")))
    )
    deploy_config_candidates = list(
        dict.fromkeys(
            profile_deploy_candidates + strings(policy_contract.get("deploy_config_candidates"))
        )
    )
    runtime_report_candidates = list(
        dict.fromkeys(
            profile_data_candidates + strings(policy_contract.get("runtime_report_candidates"))
        )
    )
    checkpoint_selection = select_best_named_ref(
        [
            ("policy_contract.primary_checkpoint_ref", policy_contract.get("primary_checkpoint_ref")),
            *_named_candidates("profile.policy_candidates", profile_policy_candidates),
            *_named_candidates(
                "policy_contract.checkpoint_candidates",
                strings(policy_contract.get("checkpoint_candidates")),
            ),
        ]
    )
    deploy_selection = select_best_named_ref(
        [
            ("policy_contract.primary_deploy_config_ref", policy_contract.get("primary_deploy_config_ref")),
            *_named_candidates("profile.deploy_candidates", profile_deploy_candidates),
            *_named_candidates(
                "policy_contract.deploy_config_candidates",
                strings(policy_contract.get("deploy_config_candidates")),
            ),
        ]
    )
    runtime_report_selection = select_best_named_ref(
        [
            ("policy_contract.primary_runtime_report_ref", policy_contract.get("primary_runtime_report_ref")),
            *_named_candidates("profile.data_candidates", profile_data_candidates),
            *_named_candidates(
                "policy_contract.runtime_report_candidates",
                strings(policy_contract.get("runtime_report_candidates")),
            ),
        ]
    )
    primary_checkpoint_ref = str(checkpoint_selection.get("ref", "") or "")
    primary_deploy_config_ref = str(deploy_selection.get("ref", "") or "")
    primary_runtime_report_ref = str(runtime_report_selection.get("ref", "") or "")
    embodiment = mapping(embodiment_context)
    motion_sources = strings(embodiment.get("motion_clip_datapacks")) + strings(
        embodiment.get("motion_clip_paths")
    )
    if embodiment.get("motion_clips"):
        motion_sources.append("inline_motion_clips")
    existing_motion_sources = [ref for ref in motion_sources if _motion_source_exists(ref)]
    retargeting_present = bool(
        embodiment.get("retargeting_contract")
        or embodiment.get("whole_body_retargeting")
        or embodiment.get("retargeting_root")
    )
    reward_overlay_present = bool(embodiment.get("whole_body_reward_overlay"))
    profile_install_preflight_status = str(profile.get("install_preflight_status", "") or "")
    profile_install_missing_components = strings(profile.get("install_missing_components"))
    profile_primary_entrypoint_ref = str(profile.get("primary_entrypoint_ref", "") or "")

    ready_surfaces: list[str] = []
    if preferred_profile and profile_install_preflight_status != "install_blocked":
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
    if profile_install_preflight_status == "install_blocked":
        missing_components.extend(profile_install_missing_components)
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
        "profile_install_by_id": profile_install_by_id,
    }
    return {
        "version": "backend_upstream_runtime_pack_v1",
        "pack_id": stable_id("backend_upstream_runtime_pack", payload),
        **payload,
        "runtime_target_ids": ready_targets,
        "runtime_target_preflight_status": str(
            runtime_target_contract.get("runtime_target_preflight_status", "") or ""
        ),
        "unverified_runtime_target_ids": strings(
            runtime_target_contract.get("unverified_required_target_ids")
        ),
        "profile_root": str(profile.get("root", "") or ""),
        "profile_git_metadata": mapping(profile.get("root_git_metadata")),
        "profile_candidate_counts": {
            "deploy": int(profile.get("deploy_candidate_count", 0) or 0),
            "policy": int(profile.get("policy_candidate_count", 0) or 0),
            "data": int(profile.get("data_candidate_count", 0) or 0),
        },
        "profile_install_preflight_status": profile_install_preflight_status,
        "profile_install_missing_components": profile_install_missing_components,
        "profile_install_verified_components": strings(
            profile.get("install_verified_components")
        ),
        "profile_primary_entrypoint_ref": profile_primary_entrypoint_ref,
        "primary_profile_deploy_ref": str(profile.get("primary_deploy_candidate", "") or ""),
        "primary_profile_policy_ref": str(profile.get("primary_policy_candidate", "") or ""),
        "primary_profile_data_ref": str(profile.get("primary_data_candidate", "") or ""),
        "profile_matched_paths": strings(profile.get("matched_paths")),
        "deploy_candidates": deploy_config_candidates,
        "policy_candidates": checkpoint_candidates,
        "data_candidates": runtime_report_candidates,
        "checkpoint_candidates": checkpoint_candidates,
        "runtime_report_candidates": runtime_report_candidates,
        "primary_policy_ref": primary_checkpoint_ref,
        "primary_policy_ref_source": str(checkpoint_selection.get("source", "") or ""),
        "policy_candidate_evidence_summary": mapping(checkpoint_selection.get("summary")),
        "primary_deploy_config_ref": primary_deploy_config_ref,
        "primary_deploy_config_ref_source": str(deploy_selection.get("source", "") or ""),
        "deploy_candidate_evidence_summary": mapping(deploy_selection.get("summary")),
        "primary_runtime_report_ref": primary_runtime_report_ref,
        "primary_runtime_report_ref_source": str(
            runtime_report_selection.get("source", "") or ""
        ),
        "runtime_report_candidate_evidence_summary": mapping(
            runtime_report_selection.get("summary")
        ),
        "motion_sources": motion_sources,
        "existing_motion_sources": existing_motion_sources,
        "missing_motion_sources": [
            ref for ref in motion_sources if ref not in existing_motion_sources
        ],
        "retargeting_present": retargeting_present,
        "reward_overlay_present": reward_overlay_present,
        "missing_components": missing_components,
        "notes": [
            "Upstream runtime pack makes Holosoma host/runtime/motion/policy/retargeting surfaces explicit.",
            "This pack keeps train-from-motion and eval-with-policy as distinct bounded modes.",
        ],
    }


__all__ = ["build_holosoma_runtime_pack"]

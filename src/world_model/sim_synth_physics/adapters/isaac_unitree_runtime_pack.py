"""Upstream runtime pack contract for Isaac/Unitree Phase-1 execution."""

from __future__ import annotations

from typing import Any, Mapping

from ..common import mapping, stable_id, strings


def _profile_row(runtime_layout_contract: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    for row in list(runtime_layout_contract.get("profiles", []) or []):
        row_mapping = mapping(row)
        if str(row_mapping.get("profile_id", "") or "") == profile_id:
            return row_mapping
    return {}


def _asset_rows(
    normalized_robot_asset_manifest: Mapping[str, Any],
) -> tuple[dict[str, str], list[str], list[str], list[str]]:
    refs: dict[str, str] = {}
    ready: list[str] = []
    verified: list[str] = []
    declared_only: list[str] = []
    for asset_id, row in mapping(normalized_robot_asset_manifest).items():
        row_mapping = mapping(row)
        if bool(row_mapping.get("present", False)):
            ready.append(str(asset_id))
            ref = str(row_mapping.get("value", "") or "")
            if ref:
                refs[str(asset_id)] = ref
            if bool(row_mapping.get("local_path_exists", False)):
                verified.append(str(asset_id))
            else:
                declared_only.append(str(asset_id))
    return refs, sorted(ready), sorted(verified), sorted(declared_only)


def build_isaac_unitree_runtime_pack(
    *,
    runtime_target_contract: Mapping[str, Any],
    runtime_layout_contract: Mapping[str, Any],
    policy_contract: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    normalized_robot_asset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    deployment = mapping(deployment_contract)
    preferred_profile = str(
        deployment.get("preferred_profile")
        or runtime_layout_contract.get("preferred_profile")
        or ""
    )
    profile = _profile_row(runtime_layout_contract, preferred_profile)
    asset_refs, ready_assets, verified_assets, declared_only_assets = _asset_rows(
        normalized_robot_asset_manifest
    )
    ready_modes = strings(deployment.get("ready_modes"))
    ready_targets = strings(runtime_target_contract.get("ready_target_ids"))
    policy_candidates = strings(policy_contract.get("checkpoint_candidates"))
    deploy_config_candidates = strings(policy_contract.get("deploy_config_candidates"))
    runtime_report_candidates = strings(policy_contract.get("runtime_report_candidates"))
    primary_checkpoint_ref = str(policy_contract.get("primary_checkpoint_ref", "") or "")
    primary_deploy_config_ref = str(policy_contract.get("primary_deploy_config_ref", "") or "")
    primary_runtime_report_ref = str(policy_contract.get("primary_runtime_report_ref", "") or "")

    ready_surfaces: list[str] = []
    if preferred_profile:
        ready_surfaces.append("runtime_profile_surface")
    if ready_targets:
        ready_surfaces.append("runtime_target_surface")
    if policy_candidates:
        ready_surfaces.append("policy_surface")
    if deploy_config_candidates:
        ready_surfaces.append("deploy_surface")
    if ready_assets:
        ready_surfaces.append("asset_surface")
    if runtime_report_candidates or strings(profile.get("data_candidates")):
        ready_surfaces.append("telemetry_surface")

    missing_components: list[str] = []
    if not preferred_profile:
        missing_components.append("preferred_runtime_profile")
    if not ready_modes:
        missing_components.append("deployment_mode")
    if not ready_targets:
        missing_components.append("runtime_targets")
    if not policy_candidates:
        missing_components.append("policy_checkpoint")
    if not ready_assets:
        missing_components.append("robot_assets")

    pack_status = "pack_ready"
    if ready_surfaces and missing_components:
        pack_status = "pack_partial"
    elif not ready_surfaces:
        pack_status = "pack_blocked"

    payload = {
        "backend": "isaac",
        "preferred_profile": preferred_profile,
        "pack_status": pack_status,
        "ready_modes": ready_modes,
        "ready_surfaces": ready_surfaces,
    }
    return {
        "version": "backend_upstream_runtime_pack_v1",
        "pack_id": stable_id("backend_upstream_runtime_pack", payload),
        **payload,
        "robot_variant": str(deployment.get("robot_variant", "") or ""),
        "placement_class": str(deployment.get("placement_class", "") or ""),
        "runtime_target_ids": ready_targets,
        "profile_root": str(profile.get("root", "") or ""),
        "profile_git_metadata": mapping(profile.get("root_git_metadata")),
        "profile_candidate_counts": {
            "deploy": int(profile.get("deploy_candidate_count", 0) or 0),
            "policy": int(profile.get("policy_candidate_count", 0) or 0),
            "data": int(profile.get("data_candidate_count", 0) or 0),
        },
        "primary_profile_deploy_ref": str(profile.get("primary_deploy_candidate", "") or ""),
        "primary_profile_policy_ref": str(profile.get("primary_policy_candidate", "") or ""),
        "primary_profile_data_ref": str(profile.get("primary_data_candidate", "") or ""),
        "profile_matched_paths": strings(profile.get("matched_paths")),
        "deploy_candidates": strings(profile.get("deploy_candidates")) or deploy_config_candidates,
        "policy_candidates": strings(profile.get("policy_candidates")) or policy_candidates,
        "data_candidates": strings(profile.get("data_candidates")) or runtime_report_candidates,
        "checkpoint_candidates": policy_candidates,
        "runtime_report_candidates": runtime_report_candidates,
        "primary_policy_ref": primary_checkpoint_ref,
        "primary_deploy_config_ref": primary_deploy_config_ref,
        "primary_runtime_report_ref": primary_runtime_report_ref,
        "asset_refs": asset_refs,
        "ready_asset_ids": ready_assets,
        "verified_asset_ids": verified_assets,
        "declared_only_asset_ids": declared_only_assets,
        "asset_evidence_summary": {
            "declared_asset_count": len(ready_assets),
            "verified_asset_count": len(verified_assets),
            "declared_only_asset_count": len(declared_only_assets),
        },
        "missing_components": missing_components,
        "notes": [
            "Upstream runtime pack makes Isaac/Unitree external runtime, policy, and asset surfaces explicit.",
            "This pack is still provider-owned reality; the WM remains the truth owner over how it is consumed.",
        ],
    }


__all__ = ["build_isaac_unitree_runtime_pack"]

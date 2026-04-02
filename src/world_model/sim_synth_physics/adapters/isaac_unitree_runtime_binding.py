"""Concrete binding of Isaac/Unitree runtime packs into executable launch surfaces."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..common import mapping, stable_id, strings


def _mode_contract(deployment_contract: Mapping[str, Any], deployment_mode: str) -> dict[str, Any]:
    for row in list(deployment_contract.get("deployment_modes", []) or []):
        row_mapping = mapping(row)
        if str(row_mapping.get("mode_id", "") or "") == deployment_mode:
            return row_mapping
    return {}


def _target_refs(
    runtime_target_contract: Mapping[str, Any],
    required_target_ids: list[str],
) -> tuple[dict[str, str], list[str]]:
    refs: dict[str, str] = {}
    missing: list[str] = []
    rows = list(runtime_target_contract.get("targets", []) or [])
    for target_id in required_target_ids:
        ref = ""
        for row in rows:
            row_mapping = mapping(row)
            if str(row_mapping.get("target_id", "") or "") == target_id:
                ref = str(row_mapping.get("ref", "") or "")
                if ref:
                    refs[target_id] = ref
                break
        if not ref:
            missing.append(target_id)
    return refs, missing


def _required_surfaces(deployment_mode: str) -> list[str]:
    by_mode = {
        "sim_eval": [
            "runtime_profile_surface",
            "runtime_target_surface",
            "policy_surface",
            "asset_surface",
        ],
        "teleop_bridge": [
            "runtime_profile_surface",
            "runtime_target_surface",
            "asset_surface",
        ],
        "lerobot_eval": [
            "runtime_profile_surface",
            "runtime_target_surface",
            "policy_surface",
            "asset_surface",
        ],
        "physical_deploy": [
            "runtime_profile_surface",
            "runtime_target_surface",
            "policy_surface",
            "asset_surface",
        ],
    }
    return list(by_mode.get(deployment_mode, by_mode["sim_eval"]))


def build_isaac_unitree_runtime_binding(
    *,
    task_id: str,
    explicit_policy_ref: str,
    preferred_profile: str,
    launch_specs: Sequence[Mapping[str, Any]],
    runtime_target_contract: Mapping[str, Any],
    policy_contract: Mapping[str, Any],
    deployment_contract: Mapping[str, Any],
    upstream_runtime_pack: Mapping[str, Any],
) -> dict[str, Any]:
    pack = mapping(upstream_runtime_pack)
    selected_profile = preferred_profile
    selected_launch_spec = next(
        (
            mapping(spec)
            for spec in list(launch_specs)
            if str(mapping(spec).get("profile_id", "") or "") == selected_profile
        ),
        mapping(launch_specs[0]) if launch_specs else {},
    )
    deployment_mode = str(
        selected_launch_spec.get("deployment_mode")
        or next(
            (
                row.get("mode_id")
                for row in list(deployment_contract.get("deployment_modes", []) or [])
                if preferred_profile in strings(mapping(row).get("profile_candidates"))
            ),
            "sim_eval",
        )
        or "sim_eval"
    )
    mode_contract = _mode_contract(deployment_contract, deployment_mode)
    pack_ready_surfaces = strings(pack.get("ready_surfaces"))
    required_surfaces = _required_surfaces(deployment_mode)
    surface_gaps = [surface for surface in required_surfaces if surface not in pack_ready_surfaces]

    selected_policy_ref = str(
        explicit_policy_ref
        or policy_contract.get("policy_ref")
        or (strings(pack.get("policy_candidates")) or strings(pack.get("checkpoint_candidates")) or [""])[0]
        or ""
    )
    selected_deploy_config = str(
        (strings(pack.get("deploy_candidates")) or strings(policy_contract.get("deploy_config_candidates")) or [""])[0]
        or ""
    )
    selected_runtime_report = str(
        (strings(pack.get("runtime_report_candidates")) or strings(pack.get("data_candidates")) or [""])[0]
        or ""
    )
    selected_launch_root = str(
        selected_launch_spec.get("root")
        or pack.get("profile_root")
        or ""
    )
    selected_command = str(selected_launch_spec.get("command", "") or "")
    required_target_ids = strings(mode_contract.get("required_target_ids"))
    selected_target_refs, missing_targets = _target_refs(runtime_target_contract, required_target_ids)

    missing_components = strings(pack.get("missing_components"))
    for item in strings(mode_contract.get("missing_preconditions")):
        if item not in missing_components:
            missing_components.append(item)
    for item in surface_gaps:
        if item not in missing_components:
            missing_components.append(item)
    for item in missing_targets:
        if item not in missing_components:
            missing_components.append(item)
    if "policy_surface" in required_surfaces and not selected_policy_ref and "policy_checkpoint" not in missing_components:
        missing_components.append("policy_checkpoint")
    if not selected_launch_root and "launch_root" not in missing_components:
        missing_components.append("launch_root")
    if not selected_command and "launch_command" not in missing_components:
        missing_components.append("launch_command")

    binding_status = "binding_ready"
    if missing_components and pack_ready_surfaces:
        binding_status = "binding_partial"
    elif missing_components:
        binding_status = "binding_blocked"

    payload = {
        "backend": "isaac",
        "preferred_profile": preferred_profile,
        "selected_profile": selected_profile,
        "deployment_mode": deployment_mode,
        "binding_status": binding_status,
        "task_id": task_id,
        "selected_policy_ref": selected_policy_ref,
        "selected_launch_root": selected_launch_root,
    }
    return {
        "version": "backend_runtime_binding_v1",
        "binding_id": stable_id("backend_runtime_binding", payload),
        **payload,
        "required_surfaces": required_surfaces,
        "ready_surfaces": pack_ready_surfaces,
        "selected_launch_spec": selected_launch_spec,
        "selected_command": selected_command,
        "selected_deploy_config": selected_deploy_config,
        "selected_runtime_report": selected_runtime_report,
        "selected_target_refs": selected_target_refs,
        "selected_asset_refs": mapping(pack.get("asset_refs")),
        "selected_asset_ids": strings(pack.get("ready_asset_ids")),
        "missing_components": list(dict.fromkeys(missing_components)),
        "pack_status": str(pack.get("pack_status", "") or ""),
        "pack_id": str(pack.get("pack_id", "") or ""),
        "notes": [
            "This binding turns the Isaac/Unitree runtime pack into selected executable surfaces.",
            "It remains real-or-unavailable: missing policy, target, asset, or launch surfaces stay explicit.",
        ],
    }


__all__ = ["build_isaac_unitree_runtime_binding"]

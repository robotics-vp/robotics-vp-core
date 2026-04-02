"""Concrete binding of Holosoma runtime packs into executable launch surfaces."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..common import mapping, stable_id, strings
from ..ref_evidence import (
    describe_ref_evidence,
    select_best_named_ref,
    summarize_preflight_evidence,
)


def _mode_contract(deployment_contract: Mapping[str, Any], deployment_mode: str) -> dict[str, Any]:
    for row in list(deployment_contract.get("deployment_modes", []) or []):
        row_mapping = mapping(row)
        if str(row_mapping.get("mode_id", "") or "") == deployment_mode:
            return row_mapping
    return {}


def _target_rows_by_id(runtime_target_contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for row in list(runtime_target_contract.get("targets", []) or []):
        row_mapping = mapping(row)
        target_id = str(row_mapping.get("target_id", "") or "")
        if target_id:
            payload[target_id] = row_mapping
    return payload


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


def _target_ref_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    ref = str(row.get("ref", "") or "")
    fallback = describe_ref_evidence(ref)
    return {
        **fallback,
        "verification_status": str(
            row.get("verification_status", fallback.get("verification_status", "")) or ""
        ),
        "ready": bool(row.get("verified", fallback.get("ready", False))),
        "verified": bool(row.get("verified", fallback.get("verified", False))),
        "path_kind": str(row.get("path_kind", "") or ""),
        "matched_markers": strings(row.get("matched_markers")),
        "missing_markers": strings(row.get("missing_markers")),
        "primary_marker_ref": str(row.get("primary_marker_ref", "") or ""),
    }


def _required_surfaces(deployment_mode: str) -> list[str]:
    by_mode = {
        "sim_eval": [
            "runtime_profile_surface",
            "runtime_target_surface",
            "policy_surface",
        ],
        "motion_train": [
            "runtime_profile_surface",
            "runtime_target_surface",
            "motion_surface",
        ],
        "retarget_eval": [
            "runtime_profile_surface",
            "runtime_target_surface",
            "policy_surface",
            "motion_surface",
            "retargeting_surface",
        ],
    }
    return list(by_mode.get(deployment_mode, by_mode["sim_eval"]))


def _local_required_surfaces(deployment_mode: str) -> list[str]:
    by_mode = {
        "sim_eval": [
            "policy_surface",
        ],
        "motion_train": [
            "motion_surface",
        ],
        "retarget_eval": [
            "policy_surface",
            "motion_surface",
            "retargeting_surface",
        ],
    }
    return list(by_mode.get(deployment_mode, by_mode["sim_eval"]))


def _relevant_pack_missing_components(
    *,
    pack_missing_components: Sequence[str],
    required_surfaces: Sequence[str],
    local_runtime_binding: bool,
    explicit_policy_available: bool,
    motion_sources_available: bool,
    retargeting_available: bool,
) -> list[str]:
    relevant: list[str] = []
    for item in list(pack_missing_components):
        if item == "deployment_mode":
            continue
        if item in {"profile_root", "profile_entrypoint"} or str(item).startswith(
            "expected_path::"
        ):
            continue
        if local_runtime_binding and item in {"preferred_runtime_profile", "runtime_targets"}:
            continue
        if item == "preferred_runtime_profile" and local_runtime_binding:
            continue
        if item == "runtime_targets" and local_runtime_binding:
            continue
        if item == "policy_checkpoint" and explicit_policy_available:
            continue
        if item == "policy_checkpoint" and "policy_surface" not in required_surfaces:
            continue
        if item == "motion_sources" and motion_sources_available:
            continue
        if item == "motion_sources" and "motion_surface" not in required_surfaces:
            continue
        if item == "retargeting_root" and retargeting_available:
            continue
        relevant.append(str(item))
    return relevant


def _required_target_ids(
    *,
    deployment_mode: str,
    mode_contract: Mapping[str, Any],
    local_runtime_binding: bool,
) -> list[str]:
    if not local_runtime_binding:
        return strings(mode_contract.get("required_target_ids"))
    local_by_mode = {
        "sim_eval": [],
        "motion_train": [],
        "retarget_eval": ["retargeting_root"],
    }
    return list(local_by_mode.get(deployment_mode, []))


def _named_candidates(prefix: str, refs: Sequence[str]) -> list[tuple[str, str]]:
    return [(f"{prefix}[{index}]", ref) for index, ref in enumerate(refs) if str(ref)]


def build_holosoma_runtime_binding(
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
    ready_modes = strings(deployment_contract.get("ready_modes"))
    selected_profile = preferred_profile
    if not explicit_policy_ref and not bool(policy_contract.get("policy_ready", False)):
        if "motion_train" in ready_modes:
            selected_profile = "holosoma_motion_bank"
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
        or ("motion_train" if selected_profile == "holosoma_motion_bank" else "sim_eval")
        or "sim_eval"
    )
    if explicit_policy_ref:
        policy_selection = {
            "ref": str(explicit_policy_ref),
            "source": "explicit_policy_ref",
            "evidence": describe_ref_evidence(explicit_policy_ref),
        }
    else:
        policy_selection = select_best_named_ref(
            [
                ("policy_contract.policy_ref", policy_contract.get("policy_ref")),
                ("pack.primary_policy_ref", pack.get("primary_policy_ref")),
                *_named_candidates("pack.policy_candidates", strings(pack.get("policy_candidates"))),
                *_named_candidates(
                    "pack.checkpoint_candidates",
                    strings(pack.get("checkpoint_candidates")),
                ),
            ]
        )
    selected_policy_ref = str(policy_selection.get("ref", "") or "")
    selected_policy_ref_source = str(policy_selection.get("source", "") or "")
    selected_motion_sources = strings(pack.get("existing_motion_sources")) or strings(
        pack.get("motion_sources")
    )
    runtime_report_selection = select_best_named_ref(
        [
            ("pack.primary_runtime_report_ref", pack.get("primary_runtime_report_ref")),
            *_named_candidates(
                "pack.runtime_report_candidates",
                strings(pack.get("runtime_report_candidates")),
            ),
            *_named_candidates("pack.data_candidates", strings(pack.get("data_candidates"))),
        ]
    )
    selected_runtime_report = str(runtime_report_selection.get("ref", "") or "")
    selected_runtime_report_source = str(runtime_report_selection.get("source", "") or "")
    selected_launch_root = str(
        selected_launch_spec.get("root")
        or pack.get("profile_root")
        or ""
    )
    selected_command = str(selected_launch_spec.get("command", "") or "")
    selected_profile_install = mapping(
        mapping(pack.get("profile_install_by_id")).get(selected_profile, {})
    )
    selected_profile_install_preflight_status = str(
        selected_profile_install.get(
            "install_preflight_status",
            pack.get("profile_install_preflight_status", ""),
        )
        or ""
    )
    selected_profile_install_missing_components = strings(
        selected_profile_install.get(
            "install_missing_components",
            pack.get("profile_install_missing_components"),
        )
    )
    selected_profile_primary_entrypoint_ref = str(
        selected_profile_install.get(
            "primary_entrypoint_ref",
            pack.get("profile_primary_entrypoint_ref", ""),
        )
        or ""
    )
    mode_contract = _mode_contract(deployment_contract, deployment_mode)
    pack_ready_surfaces = strings(pack.get("ready_surfaces"))
    local_runtime_binding = bool(runtime_target_contract.get("python_bridge_available", False))
    required_surfaces = (
        _local_required_surfaces(deployment_mode)
        if local_runtime_binding
        else _required_surfaces(deployment_mode)
    )
    required_target_ids = _required_target_ids(
        deployment_mode=deployment_mode,
        mode_contract=mode_contract,
        local_runtime_binding=local_runtime_binding,
    )
    selected_target_refs, missing_targets = _target_refs(runtime_target_contract, required_target_ids)
    target_rows = _target_rows_by_id(runtime_target_contract)
    selected_retargeting_root = str(selected_target_refs.get("retargeting_root", "") or "")
    selected_ref_evidence = {
        "policy_ref": mapping(policy_selection.get("evidence")),
        "launch_root": describe_ref_evidence(selected_launch_root),
        "profile_entrypoint": describe_ref_evidence(selected_profile_primary_entrypoint_ref),
        "runtime_report_ref": mapping(runtime_report_selection.get("evidence")),
        "retargeting_root": describe_ref_evidence(selected_retargeting_root),
    }
    selected_target_ref_evidence = {
        target_id: _target_ref_evidence(target_rows.get(target_id, {"ref": ref}))
        for target_id, ref in selected_target_refs.items()
    }
    selected_motion_source_evidence = {
        source: describe_ref_evidence(source) for source in selected_motion_sources
    }
    selected_verified_target_ids = [
        target_id
        for target_id, evidence in selected_target_ref_evidence.items()
        if bool(evidence.get("verified", False))
    ]
    selected_partial_target_ids = [
        target_id
        for target_id, evidence in selected_target_ref_evidence.items()
        if not bool(evidence.get("ready", False))
    ]
    surface_gaps: list[str] = []
    for surface in required_surfaces:
        if surface in pack_ready_surfaces:
            continue
        if (
            surface == "runtime_profile_surface"
            and selected_profile_install_preflight_status != "install_blocked"
        ):
            continue
        if surface == "policy_surface" and selected_policy_ref:
            continue
        if surface == "motion_surface" and selected_motion_sources:
            continue
        if surface == "retargeting_surface" and selected_retargeting_root:
            continue
        surface_gaps.append(surface)

    missing_components = _relevant_pack_missing_components(
        pack_missing_components=strings(pack.get("missing_components")),
        required_surfaces=required_surfaces,
        local_runtime_binding=local_runtime_binding,
        explicit_policy_available=bool(selected_policy_ref),
        motion_sources_available=bool(selected_motion_sources),
        retargeting_available=bool(selected_retargeting_root),
    )
    for item in strings(mode_contract.get("missing_preconditions")):
        if (
            item in {"runtime_profile", "holosoma_motion_root"}
            and local_runtime_binding
            and item != "retargeting_root"
        ):
            continue
        if item == "policy_checkpoint" and selected_policy_ref:
            continue
        if item == "motion_source_bundle" and selected_motion_sources:
            continue
        if item == "motion_source_bundle" and "motion_surface" not in required_surfaces:
            continue
        if item == "whole_body_retargeting_contract" and selected_retargeting_root:
            continue
        if item == "whole_body_retargeting_contract" and "retargeting_surface" not in required_surfaces:
            continue
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
    if "motion_surface" in required_surfaces and not selected_motion_sources and "motion_sources" not in missing_components:
        missing_components.append("motion_sources")
    if "retargeting_surface" in required_surfaces and not selected_retargeting_root and "retargeting_root" not in missing_components:
        missing_components.append("retargeting_root")
    if not local_runtime_binding and not selected_launch_root and "launch_root" not in missing_components:
        missing_components.append("launch_root")
    if not local_runtime_binding and not selected_command and "launch_command" not in missing_components:
        missing_components.append("launch_command")
    if not local_runtime_binding:
        for item in selected_profile_install_missing_components:
            if item not in missing_components:
                missing_components.append(item)

    preflight_required_components: list[str] = []
    if "policy_surface" in required_surfaces:
        preflight_required_components.append("policy_ref")
    if not local_runtime_binding:
        preflight_required_components.append("launch_root")
        if selected_profile_primary_entrypoint_ref or "profile_entrypoint" in selected_profile_install_missing_components:
            preflight_required_components.append("profile_entrypoint")
    for target_id in required_target_ids:
        preflight_required_components.append(f"target::{target_id}")
    if "retargeting_surface" in required_surfaces:
        preflight_required_components.append("retargeting_root")
    preflight_evidence = {
        **selected_ref_evidence,
        **{
            f"target::{target_id}": evidence
            for target_id, evidence in selected_target_ref_evidence.items()
        },
    }
    host_preflight = summarize_preflight_evidence(
        preflight_required_components,
        preflight_evidence,
    )
    if "motion_surface" in required_surfaces and not selected_motion_sources:
        host_preflight["status"] = "preflight_blocked"
        if "motion_sources" not in host_preflight["missing_components"]:
            host_preflight["missing_components"].append("motion_sources")

    binding_status = "binding_ready"
    if missing_components and pack_ready_surfaces:
        binding_status = "binding_partial"
    elif missing_components:
        binding_status = "binding_blocked"

    payload = {
        "backend": "holosoma",
        "preferred_profile": preferred_profile,
        "selected_profile": selected_profile,
        "deployment_mode": deployment_mode,
        "binding_status": binding_status,
        "task_id": task_id,
        "selected_policy_ref": selected_policy_ref,
        "selected_launch_root": selected_launch_root,
        "local_runtime_binding": local_runtime_binding,
    }
    return {
        "version": "backend_runtime_binding_v1",
        "binding_id": stable_id("backend_runtime_binding", payload),
        **payload,
        "required_surfaces": required_surfaces,
        "ready_surfaces": pack_ready_surfaces,
        "selected_launch_spec": selected_launch_spec,
        "selected_command": selected_command,
        "selected_runtime_report": selected_runtime_report,
        "selected_runtime_report_source": selected_runtime_report_source,
        "selected_target_refs": selected_target_refs,
        "selected_verified_target_ids": selected_verified_target_ids,
        "selected_partial_target_ids": selected_partial_target_ids,
        "selected_motion_sources": selected_motion_sources,
        "missing_motion_sources": strings(pack.get("missing_motion_sources")),
        "selected_retargeting_root": selected_retargeting_root,
        "selected_profile_install_preflight_status": selected_profile_install_preflight_status,
        "selected_profile_install_missing_components": selected_profile_install_missing_components,
        "selected_profile_primary_entrypoint_ref": selected_profile_primary_entrypoint_ref,
        "selected_policy_ref_source": selected_policy_ref_source,
        "selected_ref_evidence": selected_ref_evidence,
        "selected_target_ref_evidence": selected_target_ref_evidence,
        "selected_motion_source_evidence": selected_motion_source_evidence,
        "host_preflight_status": host_preflight["status"],
        "host_preflight_missing_components": host_preflight["missing_components"],
        "host_preflight_symbolic_components": host_preflight["symbolic_components"],
        "host_preflight_verified_components": host_preflight["verified_components"],
        "host_preflight_ready_components": host_preflight["ready_components"],
        "missing_components": list(dict.fromkeys(missing_components)),
        "pack_status": str(pack.get("pack_status", "") or ""),
        "pack_id": str(pack.get("pack_id", "") or ""),
        "notes": [
            "This binding turns the Holosoma runtime pack into selected executable surfaces.",
            "It keeps motion-train and sim-eval distinct and does not invent policy or retargeting surfaces.",
        ],
    }


__all__ = ["build_holosoma_runtime_binding"]

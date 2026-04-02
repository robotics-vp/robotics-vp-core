"""Backend runtime work-order compilation for Phase-1 sim/synth/physics WM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .common import mapping, stable_id, strings
from .receipts import (
    BackendRuntimeBridgeReceipt,
    BackendRuntimeExecutionReceipt,
    BackendRuntimeOutcomeReceipt,
    BackendRuntimeWorkOrderReceipt,
    RobotAssetContractReceipt,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
NON_TRAINING_GPU_RUN_BACKLOG_PATH = REPO_ROOT / "scripts" / "NON_TRAINING_GPU_RUN_BACKLOG.json"
BACKEND_BACKLOG_IDS = {
    "isaac": ["isaac_unitree_runtime_smoke"],
    "holosoma": ["holosoma_runtime_eval_smoke"],
}
BACKEND_WORK_ORDER_KINDS = {
    "isaac": "isaac_unitree_runtime_bringup",
    "holosoma": "holosoma_runtime_bringup",
}
VALIDATED_RUNTIME_OUTCOME_STATUSES = {
    "selected_refs_matched",
    "no_expected_selected_refs",
    "legacy_unchecked",
}


def _load_command_hints(loop_run_ids: list[str]) -> list[str]:
    if not NON_TRAINING_GPU_RUN_BACKLOG_PATH.exists():
        return []
    try:
        payload = json.loads(
            NON_TRAINING_GPU_RUN_BACKLOG_PATH.read_text(encoding="utf-8")
        )
    except Exception:
        return []
    hints: list[str] = []
    by_id = {
        str(row.get("loop_run_id", "")): str(row.get("command", ""))
        for row in list(payload.get("backlog", []) or [])
        if str(row.get("loop_run_id", ""))
    }
    for loop_run_id in loop_run_ids:
        command = by_id.get(loop_run_id, "")
        if command:
            hints.append(command)
    return hints


def _artifact_refs(
    bridge_receipt: BackendRuntimeBridgeReceipt,
    runtime_receipt: Optional[BackendRuntimeExecutionReceipt],
    runtime_outcome_receipt: Optional[BackendRuntimeOutcomeReceipt],
) -> list[str]:
    refs: list[str] = []
    for source in (
        list(bridge_receipt.artifact_refs),
        [] if runtime_receipt is None else list(runtime_receipt.artifact_refs),
        [] if runtime_outcome_receipt is None else list(runtime_outcome_receipt.artifact_refs),
    ):
        for ref in source:
            if ref and ref not in refs:
                refs.append(str(ref))
    return refs


def _work_order_status(
    *,
    bridge_receipt: BackendRuntimeBridgeReceipt,
    runtime_receipt: Optional[BackendRuntimeExecutionReceipt],
    runtime_outcome_receipt: Optional[BackendRuntimeOutcomeReceipt],
    selected_ref_validation_status: str,
    missing_runtime_targets: list[str],
    missing_assets: list[str],
    missing_preconditions: list[str],
    upstream_runtime_pack: dict[str, object],
    runtime_binding: dict[str, object],
) -> str:
    if bridge_receipt.execution_authority == "concrete_runtime":
        return "satisfied_by_concrete_runtime"
    structured_outputs = mapping(
        {}
        if runtime_outcome_receipt is None
        else runtime_outcome_receipt.metadata.get("structured_outputs")
    )
    ready_surfaces = strings(structured_outputs.get("ready_surfaces"))
    if (
        runtime_outcome_receipt is not None
        and str(runtime_outcome_receipt.outcome_status) == "runtime_outputs_harvested"
        and ready_surfaces
        and selected_ref_validation_status in VALIDATED_RUNTIME_OUTCOME_STATUSES
    ):
        return "satisfied_by_external_runtime_outcomes"
    if missing_runtime_targets:
        return "blocked_by_runtime_targets"
    if missing_assets:
        return "blocked_by_assets"
    if str(runtime_binding.get("binding_status", "") or "") == "binding_blocked":
        return "blocked_by_runtime_binding"
    if str(upstream_runtime_pack.get("pack_status", "") or "") == "pack_blocked":
        return "blocked_by_runtime_pack"
    if missing_preconditions:
        return "blocked_by_runtime_preconditions"
    if bridge_receipt.execution_authority == "shadow_runtime":
        return "ready_for_gpu_runtime"
    if runtime_receipt is not None:
        return "runtime_request_materialized"
    return "planning_only"


def build_backend_runtime_work_orders(
    *,
    bridge_receipt: BackendRuntimeBridgeReceipt,
    runtime_receipt: Optional[BackendRuntimeExecutionReceipt],
    runtime_outcome_receipt: Optional[BackendRuntimeOutcomeReceipt],
    robot_asset_contract_receipt: Optional[RobotAssetContractReceipt],
    world_state_id: str,
    physics_execution_contract_id: str,
) -> list[BackendRuntimeWorkOrderReceipt]:
    backend = str(bridge_receipt.backend or "")
    if backend not in BACKEND_BACKLOG_IDS:
        return []
    bridge_metadata = mapping(bridge_receipt.metadata)
    runtime_target_contract = mapping(bridge_metadata.get("runtime_target_contract"))
    runtime_layout_contract = mapping(bridge_metadata.get("runtime_layout_contract"))
    policy_contract = mapping(bridge_metadata.get("policy_contract"))
    runtime_layout_ready_profiles = strings(
        bridge_metadata.get("runtime_layout_ready_profiles")
    ) or strings(runtime_layout_contract.get("ready_profiles"))
    runtime_layout_usable_profiles = strings(
        bridge_metadata.get("runtime_layout_usable_profiles")
    ) or strings(runtime_layout_contract.get("usable_profiles"))
    runtime_metadata = {} if runtime_receipt is None else mapping(runtime_receipt.metadata)
    upstream_runtime_pack = mapping(
        runtime_metadata.get("upstream_runtime_pack")
    ) or mapping(bridge_metadata.get("upstream_runtime_pack"))
    runtime_binding = mapping(
        runtime_metadata.get("runtime_binding")
    ) or mapping(
        mapping(runtime_metadata.get("runtime_bundle")).get("runtime_binding")
    ) or mapping(
        mapping(runtime_metadata.get("launch_spec")).get("runtime_binding")
    )
    missing_runtime_targets = strings(
        runtime_target_contract.get("missing_required_target_ids")
    )
    missing_assets = list(
        bridge_metadata.get("missing_assets")
        or ([] if robot_asset_contract_receipt is None else robot_asset_contract_receipt.missing_assets)
    )
    missing_preconditions = strings(
        {}
        if runtime_receipt is None
        else mapping(runtime_receipt.metadata).get("missing_preconditions")
    )
    pack_missing_components = strings(upstream_runtime_pack.get("missing_components"))
    for item in pack_missing_components:
        if item not in missing_preconditions:
            missing_preconditions.append(item)
    for item in strings(runtime_binding.get("missing_components")):
        if item not in missing_preconditions:
            missing_preconditions.append(item)
    for item in strings(runtime_binding.get("host_preflight_missing_components")):
        if item not in missing_preconditions:
            missing_preconditions.append(item)
    outcome_metadata = (
        {} if runtime_outcome_receipt is None else mapping(runtime_outcome_receipt.metadata)
    )
    structured_outputs = mapping(outcome_metadata.get("structured_outputs"))
    selected_ref_validation = mapping(outcome_metadata.get("selected_ref_validation"))
    selected_ref_validation_status = str(
        selected_ref_validation.get("status", "legacy_unchecked")
        if runtime_outcome_receipt is not None
        else ""
    )
    if selected_ref_validation_status not in VALIDATED_RUNTIME_OUTCOME_STATUSES:
        for component in strings(selected_ref_validation.get("mismatched_components")):
            precondition = f"selected_runtime_output::{component}"
            if precondition not in missing_preconditions:
                missing_preconditions.append(precondition)
        for component in strings(selected_ref_validation.get("missing_components")):
            precondition = f"selected_runtime_output::{component}"
            if precondition not in missing_preconditions:
                missing_preconditions.append(precondition)
    linked_backlog_ids = list(BACKEND_BACKLOG_IDS.get(backend, []))
    command_hints = _load_command_hints(linked_backlog_ids)
    launch_spec = mapping(runtime_metadata.get("launch_spec"))
    launch_command = str(launch_spec.get("command", "") or "")
    if launch_command and launch_command not in command_hints:
        command_hints.append(launch_command)
    work_order_kind = BACKEND_WORK_ORDER_KINDS.get(backend, f"{backend}_runtime_bringup")
    status = _work_order_status(
        bridge_receipt=bridge_receipt,
        runtime_receipt=runtime_receipt,
        runtime_outcome_receipt=runtime_outcome_receipt,
        selected_ref_validation_status=selected_ref_validation_status,
        missing_runtime_targets=missing_runtime_targets,
        missing_assets=missing_assets,
        missing_preconditions=missing_preconditions,
        upstream_runtime_pack=upstream_runtime_pack,
        runtime_binding=runtime_binding,
    )
    payload = {
        "backend": backend,
        "bridge_id": bridge_receipt.bridge_id,
        "work_order_kind": work_order_kind,
        "status": status,
        "linked_backlog_ids": linked_backlog_ids,
    }
    return [
        BackendRuntimeWorkOrderReceipt(
            receipt_id=stable_id("backend_runtime_work_order", payload),
            backend=backend,
            bridge_id=bridge_receipt.bridge_id,
            work_order_kind=work_order_kind,
            status=status,
            linked_backlog_ids=linked_backlog_ids,
            command_hints=command_hints,
            missing_runtime_targets=missing_runtime_targets,
            missing_assets=missing_assets,
            missing_preconditions=missing_preconditions,
            artifact_refs=_artifact_refs(
                bridge_receipt,
                runtime_receipt,
                runtime_outcome_receipt,
            ),
            metadata={
                "world_state_id": world_state_id,
                "physics_execution_contract_id": physics_execution_contract_id,
                "backend_runtime_bridge_receipt_id": bridge_receipt.receipt_id,
                "backend_runtime_execution_receipt_id": (
                    "" if runtime_receipt is None else runtime_receipt.receipt_id
                ),
                "backend_runtime_outcome_receipt_id": (
                    ""
                    if runtime_outcome_receipt is None
                    else runtime_outcome_receipt.receipt_id
                ),
                "backend_runtime_outcome_status": (
                    ""
                    if runtime_outcome_receipt is None
                    else runtime_outcome_receipt.outcome_status
                ),
                "backend_runtime_output_count": (
                    0
                    if runtime_outcome_receipt is None
                    else runtime_outcome_receipt.harvested_output_count
                ),
                "backend_runtime_ready_surfaces": strings(structured_outputs.get("ready_surfaces")),
                "backend_runtime_primary_policy_ref": str(
                    structured_outputs.get("primary_policy_ref", "") or ""
                ),
                "backend_runtime_selected_ref_validation_status": str(
                    selected_ref_validation.get("status", "") or ""
                ),
                "backend_runtime_selected_ref_validation_mismatched_components": strings(
                    selected_ref_validation.get("mismatched_components")
                ),
                "backend_runtime_selected_ref_validation_missing_components": strings(
                    selected_ref_validation.get("missing_components")
                ),
                "backend_runtime_metric_keys": strings(structured_outputs.get("metric_keys")),
                "execution_authority": bridge_receipt.execution_authority,
                "transport_profile": bridge_receipt.transport_profile,
                "bridge_readiness_score": bridge_receipt.bridge_readiness_score,
                "target_hardware_class": bridge_metadata.get("target_hardware_class", ""),
                "runtime_targets_ready": runtime_target_contract.get(
                    "runtime_targets_ready", False
                ),
                "runtime_layout_ready_profiles": runtime_layout_ready_profiles,
                "runtime_layout_usable_profiles": runtime_layout_usable_profiles,
                "runtime_layout_contract": runtime_layout_contract,
                "policy_contract": policy_contract,
                "policy_ready": bool(bridge_metadata.get("policy_ready", False)),
                "upstream_runtime_pack": upstream_runtime_pack,
                "upstream_runtime_pack_status": str(
                    upstream_runtime_pack.get("pack_status", "") or ""
                ),
                "upstream_runtime_profile_root": str(
                    upstream_runtime_pack.get("profile_root", "") or ""
                ),
                "upstream_runtime_profile_git_metadata": mapping(
                    upstream_runtime_pack.get("profile_git_metadata")
                ),
                "upstream_runtime_profile_candidate_counts": mapping(
                    upstream_runtime_pack.get("profile_candidate_counts")
                ),
                "upstream_runtime_profile_install_preflight_status": str(
                    upstream_runtime_pack.get("profile_install_preflight_status", "") or ""
                ),
                "upstream_runtime_profile_install_missing_components": strings(
                    upstream_runtime_pack.get("profile_install_missing_components")
                ),
                "upstream_runtime_profile_primary_entrypoint_ref": str(
                    upstream_runtime_pack.get("profile_primary_entrypoint_ref", "") or ""
                ),
                "upstream_runtime_pack_ready_surfaces": strings(
                    upstream_runtime_pack.get("ready_surfaces")
                ),
                "upstream_runtime_primary_policy_ref": str(
                    upstream_runtime_pack.get("primary_policy_ref", "") or ""
                ),
                "upstream_runtime_primary_policy_ref_source": str(
                    upstream_runtime_pack.get("primary_policy_ref_source", "") or ""
                ),
                "upstream_runtime_policy_candidate_evidence_summary": mapping(
                    upstream_runtime_pack.get("policy_candidate_evidence_summary")
                ),
                "upstream_runtime_primary_deploy_config_ref": str(
                    upstream_runtime_pack.get("primary_deploy_config_ref", "") or ""
                ),
                "upstream_runtime_primary_deploy_config_ref_source": str(
                    upstream_runtime_pack.get("primary_deploy_config_ref_source", "") or ""
                ),
                "upstream_runtime_deploy_candidate_evidence_summary": mapping(
                    upstream_runtime_pack.get("deploy_candidate_evidence_summary")
                ),
                "upstream_runtime_primary_runtime_report_ref": str(
                    upstream_runtime_pack.get("primary_runtime_report_ref", "") or ""
                ),
                "upstream_runtime_primary_runtime_report_ref_source": str(
                    upstream_runtime_pack.get("primary_runtime_report_ref_source", "") or ""
                ),
                "upstream_runtime_runtime_report_candidate_evidence_summary": mapping(
                    upstream_runtime_pack.get("runtime_report_candidate_evidence_summary")
                ),
                "upstream_runtime_verified_asset_ids": strings(
                    upstream_runtime_pack.get("verified_asset_ids")
                ),
                "upstream_runtime_declared_only_asset_ids": strings(
                    upstream_runtime_pack.get("declared_only_asset_ids")
                ),
                "upstream_runtime_existing_motion_sources": strings(
                    upstream_runtime_pack.get("existing_motion_sources")
                ),
                "runtime_binding": runtime_binding,
                "runtime_binding_status": str(runtime_binding.get("binding_status", "") or ""),
                "runtime_binding_selected_profile": str(
                    runtime_binding.get("selected_profile", "") or ""
                ),
                "runtime_binding_selected_policy_ref": str(
                    runtime_binding.get("selected_policy_ref", "") or ""
                ),
                "runtime_binding_selected_policy_ref_source": str(
                    runtime_binding.get("selected_policy_ref_source", "") or ""
                ),
                "runtime_binding_selected_deploy_config": str(
                    runtime_binding.get("selected_deploy_config", "") or ""
                ),
                "runtime_binding_selected_deploy_config_source": str(
                    runtime_binding.get("selected_deploy_config_source", "") or ""
                ),
                "runtime_binding_selected_runtime_report": str(
                    runtime_binding.get("selected_runtime_report", "") or ""
                ),
                "runtime_binding_selected_runtime_report_source": str(
                    runtime_binding.get("selected_runtime_report_source", "") or ""
                ),
                "runtime_binding_selected_launch_root": str(
                    runtime_binding.get("selected_launch_root", "") or ""
                ),
                "runtime_binding_selected_profile_install_preflight_status": str(
                    runtime_binding.get("selected_profile_install_preflight_status", "") or ""
                ),
                "runtime_binding_selected_profile_install_missing_components": strings(
                    runtime_binding.get("selected_profile_install_missing_components")
                ),
                "runtime_binding_selected_profile_primary_entrypoint_ref": str(
                    runtime_binding.get("selected_profile_primary_entrypoint_ref", "") or ""
                ),
                "runtime_binding_selected_verified_target_ids": strings(
                    runtime_binding.get("selected_verified_target_ids")
                ),
                "runtime_binding_selected_partial_target_ids": strings(
                    runtime_binding.get("selected_partial_target_ids")
                ),
                "runtime_binding_host_preflight_status": str(
                    runtime_binding.get("host_preflight_status", "") or ""
                ),
                "runtime_binding_host_preflight_missing_components": strings(
                    runtime_binding.get("host_preflight_missing_components")
                ),
                "runtime_binding_host_preflight_symbolic_components": strings(
                    runtime_binding.get("host_preflight_symbolic_components")
                ),
                "runtime_binding_selected_ref_evidence": mapping(
                    runtime_binding.get("selected_ref_evidence")
                ),
                "runtime_bundle": runtime_metadata.get("runtime_bundle", {}),
                "launch_spec": launch_spec,
            },
        )
    ]


__all__ = ["build_backend_runtime_work_orders"]

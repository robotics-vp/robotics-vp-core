"""Backend runtime work-order compilation for Phase-1 sim/synth/physics WM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .common import mapping, stable_id, strings
from .receipts import (
    BackendRuntimeBridgeReceipt,
    BackendRuntimeExecutionReceipt,
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
) -> list[str]:
    refs: list[str] = []
    for source in (
        list(bridge_receipt.artifact_refs),
        [] if runtime_receipt is None else list(runtime_receipt.artifact_refs),
    ):
        for ref in source:
            if ref and ref not in refs:
                refs.append(str(ref))
    return refs


def _work_order_status(
    *,
    bridge_receipt: BackendRuntimeBridgeReceipt,
    runtime_receipt: Optional[BackendRuntimeExecutionReceipt],
    missing_runtime_targets: list[str],
    missing_assets: list[str],
    missing_preconditions: list[str],
) -> str:
    if bridge_receipt.execution_authority == "concrete_runtime":
        return "satisfied_by_concrete_runtime"
    if missing_runtime_targets:
        return "blocked_by_runtime_targets"
    if missing_assets:
        return "blocked_by_assets"
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
    runtime_metadata = {} if runtime_receipt is None else mapping(runtime_receipt.metadata)
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
        missing_runtime_targets=missing_runtime_targets,
        missing_assets=missing_assets,
        missing_preconditions=missing_preconditions,
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
            artifact_refs=_artifact_refs(bridge_receipt, runtime_receipt),
            metadata={
                "world_state_id": world_state_id,
                "physics_execution_contract_id": physics_execution_contract_id,
                "backend_runtime_bridge_receipt_id": bridge_receipt.receipt_id,
                "backend_runtime_execution_receipt_id": (
                    "" if runtime_receipt is None else runtime_receipt.receipt_id
                ),
                "execution_authority": bridge_receipt.execution_authority,
                "transport_profile": bridge_receipt.transport_profile,
                "bridge_readiness_score": bridge_receipt.bridge_readiness_score,
                "target_hardware_class": bridge_metadata.get("target_hardware_class", ""),
                "runtime_targets_ready": runtime_target_contract.get(
                    "runtime_targets_ready", False
                ),
                "runtime_layout_ready_profiles": runtime_layout_ready_profiles,
                "runtime_layout_contract": runtime_layout_contract,
                "policy_contract": policy_contract,
                "policy_ready": bool(bridge_metadata.get("policy_ready", False)),
                "runtime_bundle": runtime_metadata.get("runtime_bundle", {}),
                "launch_spec": launch_spec,
            },
        )
    ]


__all__ = ["build_backend_runtime_work_orders"]

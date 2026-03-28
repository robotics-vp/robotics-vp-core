"""Runtime-evidence summarization for sim/synth/physics receipts."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .common import mapping
from .receipts import (
    BackendRuntimeExecutionReceipt,
    BackendRuntimeLaunchReceipt,
    BackendRuntimeOutcomeReceipt,
    BackendShadowExecutionReceipt,
    RenderProviderReceipt,
    SimulationOutcomeReceipt,
)


def _is_materialized_status(status: str) -> bool:
    normalized = str(status or "")
    return normalized in {
        "scene_materialized",
        "counterfactuals_materialized",
        "ggds_scene_materialized",
        "work_order_materialized",
        "work_order_materialized_with_preconditions",
    }


def _is_concrete_runtime_status(status: str) -> bool:
    return str(status or "") in {"runtime_execution_completed", "runtime_training_completed"}


def summarize_runtime_evidence(
    *,
    backend_runtime_execution_receipt: Optional[BackendRuntimeExecutionReceipt],
    backend_runtime_launch_receipt: Optional[BackendRuntimeLaunchReceipt],
    backend_runtime_outcome_receipt: Optional[BackendRuntimeOutcomeReceipt],
    backend_shadow_execution_receipt: Optional[BackendShadowExecutionReceipt],
    render_provider_receipts: Sequence[RenderProviderReceipt],
    outcome_receipts: Sequence[SimulationOutcomeReceipt],
) -> dict[str, Any]:
    render_receipts = list(render_provider_receipts)
    outcome_list = list(outcome_receipts)
    runtime_status = (
        ""
        if backend_runtime_execution_receipt is None
        else backend_runtime_execution_receipt.execution_status
    )
    launch_status = (
        ""
        if backend_runtime_launch_receipt is None
        else backend_runtime_launch_receipt.launch_status
    )
    runtime_outcome_status = (
        ""
        if backend_runtime_outcome_receipt is None
        else backend_runtime_outcome_receipt.outcome_status
    )
    shadow_status = (
        "" if backend_shadow_execution_receipt is None else backend_shadow_execution_receipt.execution_status
    )
    return {
        "runtime_execution_status": runtime_status,
        "runtime_launch_status": launch_status,
        "runtime_launch_executed": (
            False
            if backend_runtime_launch_receipt is None
            else bool(backend_runtime_launch_receipt.executed)
        ),
        "runtime_output_status": runtime_outcome_status,
        "runtime_output_harvested": runtime_outcome_status == "runtime_outputs_harvested",
        "runtime_output_executed": (
            False
            if backend_runtime_outcome_receipt is None
            else bool(backend_runtime_outcome_receipt.executed)
        ),
        "runtime_concrete_completed": _is_concrete_runtime_status(runtime_status),
        "runtime_artifact_count": (
            0
            if backend_runtime_execution_receipt is None
            else len(backend_runtime_execution_receipt.artifact_refs)
        ),
        "runtime_launch_artifact_count": (
            0
            if backend_runtime_launch_receipt is None
            else len(backend_runtime_launch_receipt.artifact_refs)
        ),
        "runtime_output_artifact_count": (
            0
            if backend_runtime_outcome_receipt is None
            else int(backend_runtime_outcome_receipt.harvested_output_count)
        ),
        "runtime_output_artifact_kinds": sorted(
            mapping(
                {}
                if backend_runtime_outcome_receipt is None
                else backend_runtime_outcome_receipt.metadata.get("artifact_kind_counts")
            ).keys()
        ),
        "runtime_episode_count": (
            0
            if backend_runtime_execution_receipt is None
            else int(backend_runtime_execution_receipt.metadata.get("rollout_episode_count", 0) or 0)
        ),
        "shadow_execution_status": shadow_status,
        "shadow_artifact_count": (
            0
            if backend_shadow_execution_receipt is None
            else len(backend_shadow_execution_receipt.artifact_refs)
        ),
        "shadow_missing_asset_count": (
            0
            if backend_shadow_execution_receipt is None
            else len(
                list(
                    backend_shadow_execution_receipt.metadata.get("missing_assets", [])
                    or []
                )
            )
        ),
        "materialized_render_provider_count": sum(
            1
            for receipt in render_receipts
            if _is_materialized_status(str(receipt.materialization_status))
        ),
        "blocked_render_provider_count": sum(
            1
            for receipt in render_receipts
            if str(receipt.materialization_status) == "materialization_blocked"
        ),
        "render_precondition_gap_count": sum(
            len(list(receipt.metadata.get("unsatisfied_preconditions", []) or []))
            for receipt in render_receipts
        ),
        "render_artifact_count": sum(len(list(receipt.artifact_refs)) for receipt in render_receipts),
        "planned_branch_count": sum(
            1 for receipt in outcome_list if str(receipt.status).startswith("planned_")
        ),
        "blocked_branch_count": sum(
            1 for receipt in outcome_list if str(receipt.status).startswith("blocked_")
        ),
        "provider_truth_classes": sorted(
            {
                str(receipt.metadata.get("provider_truth_class", "") or "")
                for receipt in render_receipts
                if str(receipt.metadata.get("provider_truth_class", "") or "")
            }
        ),
        "render_receipt_refs": [receipt.receipt_id for receipt in render_receipts],
        "outcome_receipt_refs": [receipt.receipt_id for receipt in outcome_list],
        "backend_shadow_metadata": (
            {}
            if backend_shadow_execution_receipt is None
            else mapping(backend_shadow_execution_receipt.metadata)
        ),
    }


__all__ = ["summarize_runtime_evidence"]

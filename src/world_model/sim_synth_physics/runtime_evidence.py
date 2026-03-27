"""Runtime-evidence summarization for sim/synth/physics receipts."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .common import mapping
from .receipts import BackendShadowExecutionReceipt, RenderProviderReceipt, SimulationOutcomeReceipt


def summarize_runtime_evidence(
    *,
    backend_shadow_execution_receipt: Optional[BackendShadowExecutionReceipt],
    render_provider_receipts: Sequence[RenderProviderReceipt],
    outcome_receipts: Sequence[SimulationOutcomeReceipt],
) -> dict[str, Any]:
    render_receipts = list(render_provider_receipts)
    outcome_list = list(outcome_receipts)
    shadow_status = (
        "" if backend_shadow_execution_receipt is None else backend_shadow_execution_receipt.execution_status
    )
    return {
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
            if str(receipt.materialization_status)
            not in {"", "planned_only", "materialization_blocked"}
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

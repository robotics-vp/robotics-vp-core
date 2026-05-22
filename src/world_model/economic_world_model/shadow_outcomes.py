"""Phase-5.1 shadow outcome loop for Economic WM work orders.

This closes the local advisory loop by joining shadow work orders to typed
supervision records and emitting structural outcome receipts. These are local
comparison receipts only; they are not hardware outcomes, provider receipts,
promotion-grade benchmarks, or training evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.economic_world_model.shadow_execution import (
    EconomicWMShadowExecutionReport,
    EconomicWMShadowOutcomeComparison,
    EconomicWMShadowWorkOrder,
    load_economic_wm_shadow_execution_report,
    load_economic_wm_shadow_outcome_comparisons,
    load_economic_wm_shadow_work_orders,
)
from src.world_model.economic_world_model.supervision_substrate import (
    EconomicWMSupervisionManifest,
    EconomicWMSupervisionRecord,
    load_economic_wm_supervision_manifest,
    load_economic_wm_supervision_records,
)

ECONOMIC_WM_SHADOW_OUTCOME_RECEIPT_VERSION = "economic_wm_shadow_outcome_receipt_v1"
ECONOMIC_WM_SHADOW_OUTCOME_LOOP_REPORT_VERSION = (
    "economic_wm_shadow_outcome_loop_report_v1"
)

SHADOW_OUTCOME_LOOP_BLOCKERS = (
    "hardware_outcome_receipts_not_collected",
    "provider_runtime_outcomes_not_collected",
    "promotion_grade_shadow_benchmarks_missing",
    "gpu_training_not_run",
)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _float_dict(payload: Mapping[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, value in dict(payload or {}).items():
        try:
            values[str(key)] = float(value)
        except Exception:
            continue
    return values


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class EconomicWMShadowOutcomeReceipt:
    """Structural local receipt for one advisory shadow work order."""

    receipt_id: str
    work_order_id: str
    allocation_label: str
    recommended_action: str
    receipt_class: str = "local_structural_shadow_outcome"
    observed_effects: Dict[str, float] = field(default_factory=dict)
    expected_effects: Dict[str, float] = field(default_factory=dict)
    comparison_metrics: Dict[str, float] = field(default_factory=dict)
    evidence_refs: Dict[str, Any] = field(default_factory=dict)
    authority_class: str = "shadow_outcome_receipt_only"
    hardware_executed: bool = False
    provider_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_SHADOW_OUTCOME_RECEIPT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "version": self.version,
            "work_order_id": self.work_order_id,
            "allocation_label": self.allocation_label,
            "recommended_action": self.recommended_action,
            "receipt_class": self.receipt_class,
            "observed_effects": _float_dict(self.observed_effects),
            "expected_effects": _float_dict(self.expected_effects),
            "comparison_metrics": _float_dict(self.comparison_metrics),
            "evidence_refs": _mapping(self.evidence_refs),
            "authority_class": self.authority_class,
            "hardware_executed": bool(self.hardware_executed),
            "provider_executed": bool(self.provider_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EconomicWMShadowOutcomeReceipt":
        return cls(
            receipt_id=str(payload.get("receipt_id", "")),
            work_order_id=str(payload.get("work_order_id", "")),
            allocation_label=str(payload.get("allocation_label", "")),
            recommended_action=str(payload.get("recommended_action", "")),
            receipt_class=str(
                payload.get("receipt_class", "local_structural_shadow_outcome")
            ),
            observed_effects=_float_dict(payload.get("observed_effects", {})),
            expected_effects=_float_dict(payload.get("expected_effects", {})),
            comparison_metrics=_float_dict(payload.get("comparison_metrics", {})),
            evidence_refs=_mapping(payload.get("evidence_refs")),
            authority_class=str(
                payload.get("authority_class", "shadow_outcome_receipt_only")
            ),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_SHADOW_OUTCOME_RECEIPT_VERSION)
            ),
        )


@dataclass(frozen=True)
class EconomicWMShadowOutcomeLoopReport:
    """Manifest for the local shadow outcome loop."""

    report_id: str
    shadow_execution_report_id: str
    supervision_manifest_id: str
    outcome_receipt_count: int
    completed_comparison_count: int
    outcome_receipts_path: str
    updated_comparisons_path: str
    status: str
    authority_class: str = "shadow_outcome_loop_only"
    local_structural_loop_closed: bool = False
    hardware_executed: bool = False
    provider_executed: bool = False
    live_policy_control: bool = False
    reward_math_mutation: bool = False
    promotion_eligible: bool = False
    blockers: list[str] = field(default_factory=list)
    aggregate_counts: Dict[str, float] = field(default_factory=dict)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = ECONOMIC_WM_SHADOW_OUTCOME_LOOP_REPORT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "version": self.version,
            "shadow_execution_report_id": self.shadow_execution_report_id,
            "supervision_manifest_id": self.supervision_manifest_id,
            "outcome_receipt_count": int(self.outcome_receipt_count),
            "completed_comparison_count": int(self.completed_comparison_count),
            "outcome_receipts_path": self.outcome_receipts_path,
            "updated_comparisons_path": self.updated_comparisons_path,
            "status": self.status,
            "authority_class": self.authority_class,
            "local_structural_loop_closed": bool(self.local_structural_loop_closed),
            "hardware_executed": bool(self.hardware_executed),
            "provider_executed": bool(self.provider_executed),
            "live_policy_control": bool(self.live_policy_control),
            "reward_math_mutation": bool(self.reward_math_mutation),
            "promotion_eligible": bool(self.promotion_eligible),
            "blockers": list(self.blockers),
            "aggregate_counts": _float_dict(self.aggregate_counts),
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EconomicWMShadowOutcomeLoopReport":
        return cls(
            report_id=str(payload.get("report_id", "")),
            shadow_execution_report_id=str(
                payload.get("shadow_execution_report_id", "")
            ),
            supervision_manifest_id=str(payload.get("supervision_manifest_id", "")),
            outcome_receipt_count=int(payload.get("outcome_receipt_count", 0) or 0),
            completed_comparison_count=int(
                payload.get("completed_comparison_count", 0) or 0
            ),
            outcome_receipts_path=str(payload.get("outcome_receipts_path", "")),
            updated_comparisons_path=str(payload.get("updated_comparisons_path", "")),
            status=str(payload.get("status", "blocked")),
            authority_class=str(
                payload.get("authority_class", "shadow_outcome_loop_only")
            ),
            local_structural_loop_closed=bool(
                payload.get("local_structural_loop_closed", False)
            ),
            hardware_executed=bool(payload.get("hardware_executed", False)),
            provider_executed=bool(payload.get("provider_executed", False)),
            live_policy_control=bool(payload.get("live_policy_control", False)),
            reward_math_mutation=bool(payload.get("reward_math_mutation", False)),
            promotion_eligible=bool(payload.get("promotion_eligible", False)),
            blockers=[str(item) for item in list(payload.get("blockers", []) or [])],
            aggregate_counts=_float_dict(payload.get("aggregate_counts", {})),
            artifact_refs=_mapping(payload.get("artifact_refs")),
            metadata=_mapping(payload.get("metadata")),
            version=str(
                payload.get("version", ECONOMIC_WM_SHADOW_OUTCOME_LOOP_REPORT_VERSION)
            ),
        )


def _build_receipt(
    *,
    order: EconomicWMShadowWorkOrder,
    supervision_manifest: EconomicWMSupervisionManifest,
    supervision_records: list[EconomicWMSupervisionRecord],
) -> EconomicWMShadowOutcomeReceipt:
    record_count = max(1, supervision_manifest.record_count)
    ready_fraction = supervision_manifest.ready_record_count / record_count
    candidate_density = (
        sum(record.candidate_count for record in supervision_records) / record_count
    )
    target_density = (
        sum(record.value_target_count for record in supervision_records) / record_count
    )
    observed_effects = {
        "outcome_receipts_observed": 1.0,
        "local_structural_loop_closed": 1.0,
        "supervision_record_coverage": ready_fraction,
        "counterfactual_eval_coverage": supervision_manifest.counterfactual_eval_count
        / record_count,
        "value_target_pack_coverage": supervision_manifest.value_target_pack_count
        / record_count,
        "value_ledger_receipt_coverage": supervision_manifest.value_ledger_receipt_count
        / record_count,
        "candidate_density": candidate_density,
        "value_target_density": target_density,
        "hardware_outcome_coverage": 0.0,
        "provider_outcome_coverage": 0.0,
    }
    expected_value = float(order.expected_effects.get("expected_value", 0.0))
    comparison_metrics = {
        "expected_value": expected_value,
        "structural_coverage_delta": ready_fraction - min(1.0, expected_value),
        "counterfactual_accuracy_observed": 0.0,
        "pareto_quality_observed": 0.0,
        "promotion_grade_evidence_observed": 0.0,
    }
    payload = {
        "work_order_id": order.work_order_id,
        "supervision_manifest_id": supervision_manifest.manifest_id,
        "observed_effects": observed_effects,
        "comparison_metrics": comparison_metrics,
    }
    return EconomicWMShadowOutcomeReceipt(
        receipt_id=f"ewm_shadow_outcome_receipt_{sha256_json(payload)[:16]}",
        work_order_id=order.work_order_id,
        allocation_label=order.allocation_label,
        recommended_action=order.recommended_action,
        observed_effects=observed_effects,
        expected_effects=order.expected_effects,
        comparison_metrics=comparison_metrics,
        evidence_refs={
            "supervision_manifest_id": supervision_manifest.manifest_id,
            "supervision_records_path": supervision_manifest.records_path,
            "work_order_id": order.work_order_id,
        },
        blockers=list(SHADOW_OUTCOME_LOOP_BLOCKERS),
        metadata={
            "boundary": "local structural outcome receipt only",
            "no_hardware_or_provider_execution": True,
        },
    )


def _updated_comparison(
    *,
    comparison: EconomicWMShadowOutcomeComparison,
    receipt: EconomicWMShadowOutcomeReceipt,
) -> EconomicWMShadowOutcomeComparison:
    return EconomicWMShadowOutcomeComparison(
        comparison_id=comparison.comparison_id,
        work_order_id=comparison.work_order_id,
        comparison_status="local_structural_receipt_joined",
        expected_effect_keys=sorted(receipt.expected_effects.keys()),
        observed_outcome_refs=[receipt.receipt_id],
        comparison_metrics={
            **comparison.comparison_metrics,
            **receipt.comparison_metrics,
            "outcome_receipts_observed": 1.0,
            "local_structural_loop_closed": 1.0,
        },
        authority_class=comparison.authority_class,
        live_policy_control=False,
        reward_math_mutation=False,
        promotion_eligible=False,
        blockers=list(SHADOW_OUTCOME_LOOP_BLOCKERS),
        metadata={
            **comparison.metadata,
            "receipt_class": receipt.receipt_class,
            "boundary": "structural comparison only",
        },
    )


def build_economic_wm_shadow_outcome_loop(
    *,
    shadow_execution_report: EconomicWMShadowExecutionReport,
    work_orders: Iterable[EconomicWMShadowWorkOrder],
    outcome_comparisons: Iterable[EconomicWMShadowOutcomeComparison],
    supervision_manifest: EconomicWMSupervisionManifest,
    supervision_records: Iterable[EconomicWMSupervisionRecord],
    outcome_receipts_path: str | Path,
    updated_comparisons_path: str | Path,
    artifact_refs: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[
    EconomicWMShadowOutcomeLoopReport,
    list[EconomicWMShadowOutcomeReceipt],
    list[EconomicWMShadowOutcomeComparison],
]:
    order_items = list(work_orders)
    comparison_by_order = {item.work_order_id: item for item in outcome_comparisons}
    record_items = list(supervision_records)
    receipts = [
        _build_receipt(
            order=order,
            supervision_manifest=supervision_manifest,
            supervision_records=record_items,
        )
        for order in order_items
    ]
    updated = [
        _updated_comparison(
            comparison=comparison_by_order.get(
                receipt.work_order_id,
                EconomicWMShadowOutcomeComparison(
                    comparison_id=f"ewm_shadow_outcome_{sha256_json({'work_order_id': receipt.work_order_id})[:16]}",
                    work_order_id=receipt.work_order_id,
                ),
            ),
            receipt=receipt,
        )
        for receipt in receipts
    ]
    loop_closed = bool(
        receipts
        and len(receipts) == len(order_items)
        and supervision_manifest.ready_for_shadow_outcome_loop
    )
    status = "ok" if loop_closed else "blocked"
    payload = {
        "shadow_execution_report_id": shadow_execution_report.report_id,
        "supervision_manifest_id": supervision_manifest.manifest_id,
        "receipt_ids": [receipt.receipt_id for receipt in receipts],
        "comparison_ids": [comparison.comparison_id for comparison in updated],
    }
    report = EconomicWMShadowOutcomeLoopReport(
        report_id=f"ewm_shadow_outcome_loop_{sha256_json(payload)[:16]}",
        shadow_execution_report_id=shadow_execution_report.report_id,
        supervision_manifest_id=supervision_manifest.manifest_id,
        outcome_receipt_count=len(receipts),
        completed_comparison_count=len(updated),
        outcome_receipts_path=str(outcome_receipts_path),
        updated_comparisons_path=str(updated_comparisons_path),
        status=status,
        local_structural_loop_closed=loop_closed,
        blockers=list(SHADOW_OUTCOME_LOOP_BLOCKERS),
        aggregate_counts={
            "work_order_count": float(len(order_items)),
            "outcome_receipt_count": float(len(receipts)),
            "completed_comparison_count": float(len(updated)),
            "supervision_record_count": float(supervision_manifest.record_count),
            "ready_supervision_record_count": float(
                supervision_manifest.ready_record_count
            ),
            "hardware_executed_count": 0.0,
            "provider_executed_count": 0.0,
            "promotion_grade_evidence_count": 0.0,
        },
        artifact_refs={
            **_mapping(artifact_refs),
            "outcome_receipts_path": str(outcome_receipts_path),
            "updated_comparisons_path": str(updated_comparisons_path),
        },
        metadata={
            **_mapping(metadata),
            "boundary": "local structural shadow outcome loop only",
        },
    )
    return report, receipts, updated


def save_economic_wm_shadow_outcome_loop(
    *,
    report_path: str | Path,
    report: EconomicWMShadowOutcomeLoopReport,
    outcome_receipts: Iterable[EconomicWMShadowOutcomeReceipt],
    updated_comparisons: Iterable[EconomicWMShadowOutcomeComparison],
) -> None:
    _write_json(report_path, report.to_dict())
    _write_jsonl(
        report.outcome_receipts_path, [item.to_dict() for item in outcome_receipts]
    )
    _write_jsonl(
        report.updated_comparisons_path,
        [item.to_dict() for item in updated_comparisons],
    )


def load_economic_wm_shadow_outcome_loop_report(
    path: str | Path,
) -> EconomicWMShadowOutcomeLoopReport:
    return EconomicWMShadowOutcomeLoopReport.from_dict(_load_json(path))


def load_economic_wm_shadow_outcome_receipts(
    path: str | Path,
) -> list[EconomicWMShadowOutcomeReceipt]:
    return [EconomicWMShadowOutcomeReceipt.from_dict(row) for row in _load_jsonl(path)]


def build_economic_wm_shadow_outcome_loop_from_paths(
    *,
    shadow_execution_report_path: str | Path,
    supervision_manifest_path: str | Path,
    report_path: str | Path,
    outcome_receipts_path: str | Path,
    updated_comparisons_path: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
) -> EconomicWMShadowOutcomeLoopReport:
    shadow_report = load_economic_wm_shadow_execution_report(
        shadow_execution_report_path
    )
    work_orders = load_economic_wm_shadow_work_orders(shadow_report.work_orders_path)
    comparisons = load_economic_wm_shadow_outcome_comparisons(
        shadow_report.outcome_comparisons_path
    )
    supervision_manifest = load_economic_wm_supervision_manifest(
        supervision_manifest_path
    )
    supervision_records = load_economic_wm_supervision_records(
        supervision_manifest.records_path
    )
    report, receipts, updated = build_economic_wm_shadow_outcome_loop(
        shadow_execution_report=shadow_report,
        work_orders=work_orders,
        outcome_comparisons=comparisons,
        supervision_manifest=supervision_manifest,
        supervision_records=supervision_records,
        outcome_receipts_path=outcome_receipts_path,
        updated_comparisons_path=updated_comparisons_path,
        artifact_refs={
            "shadow_execution_report_path": str(shadow_execution_report_path),
            "supervision_manifest_path": str(supervision_manifest_path),
            "report_path": str(report_path),
        },
        metadata=metadata,
    )
    save_economic_wm_shadow_outcome_loop(
        report_path=report_path,
        report=report,
        outcome_receipts=receipts,
        updated_comparisons=updated,
    )
    return report

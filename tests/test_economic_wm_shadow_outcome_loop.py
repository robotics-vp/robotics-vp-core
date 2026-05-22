from __future__ import annotations

from pathlib import Path

from scripts.economic_world_model.run_economic_wm_shadow_outcome_loop import (
    run_economic_wm_shadow_outcome_loop,
)
from src.world_model.economic_world_model import (
    EconomicWMShadowExecutionReport,
    EconomicWMShadowOutcomeComparison,
    EconomicWMSupervisionManifest,
    EconomicWMSupervisionRecord,
    build_economic_wm_shadow_outcome_loop,
    load_economic_wm_shadow_outcome_loop_report,
    load_economic_wm_shadow_outcome_receipts,
    load_economic_wm_shadow_work_orders,
    save_economic_wm_shadow_execution_report,
    save_economic_wm_supervision_substrate,
)
from src.world_model.economic_world_model.shadow_execution import (
    EconomicWMShadowWorkOrder,
)


def _order() -> EconomicWMShadowWorkOrder:
    return EconomicWMShadowWorkOrder(
        work_order_id="work_order_test",
        allocation_candidate_id="candidate_test",
        allocation_label="prepare_teacher_provider_evidence_contracts",
        recommended_action="prepare_non_stub_provider_receipt_runbook",
        priority=0.7,
        source_row_ids=["row_a", "row_b"],
        temporal_window_ids=["window_a"],
        resource_request={"local_dev_budget": 0.3, "gpu_budget": 0.0},
        expected_effects={
            "expected_value": 0.7,
            "phase5_composition_rows_covered": 2.0,
        },
    )


def _supervision(
    tmp_path,
) -> tuple[EconomicWMSupervisionManifest, list[EconomicWMSupervisionRecord]]:
    records = [
        EconomicWMSupervisionRecord(
            supervision_record_id="sup_a",
            source_row_id="row_a",
            source_episode_id="episode_a",
            join_row_id="join_a",
            counterfactual_eval_id="cf_a",
            value_target_pack_id="vt_a",
            value_ledger_receipt_id="ledger_a",
            recommended_action="collect_more_data",
            candidate_count=3,
            value_target_count=3,
            ready_for_shadow_outcome_loop=True,
        ),
        EconomicWMSupervisionRecord(
            supervision_record_id="sup_b",
            source_row_id="row_b",
            source_episode_id="episode_b",
            join_row_id="join_b",
            counterfactual_eval_id="cf_b",
            value_target_pack_id="vt_b",
            value_ledger_receipt_id="ledger_b",
            recommended_action="noop",
            candidate_count=2,
            value_target_count=3,
            ready_for_shadow_outcome_loop=True,
        ),
    ]
    manifest = EconomicWMSupervisionManifest(
        manifest_id="sup_manifest",
        phase5_manifest_id="phase5_test",
        record_count=2,
        ready_record_count=2,
        counterfactual_eval_count=2,
        value_target_pack_count=2,
        value_ledger_receipt_count=2,
        records_path=str(tmp_path / "supervision_records.jsonl"),
        status="ok",
        ready_for_shadow_outcome_loop=True,
    )
    return manifest, records


def _shadow_report(
    tmp_path, order: EconomicWMShadowWorkOrder
) -> EconomicWMShadowExecutionReport:
    return EconomicWMShadowExecutionReport(
        report_id="shadow_execution_test",
        phase5_manifest_id="phase5_test",
        allocation_eval_id="alloc_test",
        trainer_scaffold_id="trainer_test",
        recommended_candidate="prepare_teacher_provider_evidence_contracts",
        work_order_count=1,
        outcome_comparison_count=1,
        work_orders_path=str(tmp_path / "work_orders.jsonl"),
        outcome_comparisons_path=str(tmp_path / "comparisons.jsonl"),
        status="ok",
        ready_for_shadow_comparison=True,
    )


def test_shadow_outcome_loop_closes_local_structural_loop(tmp_path) -> None:
    order = _order()
    comparison = EconomicWMShadowOutcomeComparison(
        comparison_id="comparison_test",
        work_order_id=order.work_order_id,
        expected_effect_keys=sorted(order.expected_effects),
    )
    supervision_manifest, supervision_records = _supervision(tmp_path)
    shadow_report = _shadow_report(tmp_path, order)

    report, receipts, comparisons = build_economic_wm_shadow_outcome_loop(
        shadow_execution_report=shadow_report,
        work_orders=[order],
        outcome_comparisons=[comparison],
        supervision_manifest=supervision_manifest,
        supervision_records=supervision_records,
        outcome_receipts_path="receipts.jsonl",
        updated_comparisons_path="updated.jsonl",
    )

    assert report.version == "economic_wm_shadow_outcome_loop_report_v1"
    assert report.status == "ok"
    assert report.local_structural_loop_closed is True
    assert report.hardware_executed is False
    assert report.provider_executed is False
    assert report.promotion_eligible is False
    assert receipts[0].receipt_class == "local_structural_shadow_outcome"
    assert receipts[0].observed_effects["supervision_record_coverage"] == 1.0
    assert receipts[0].comparison_metrics["promotion_grade_evidence_observed"] == 0.0
    assert comparisons[0].comparison_status == "local_structural_receipt_joined"
    assert comparisons[0].observed_outcome_refs == [receipts[0].receipt_id]


def test_shadow_outcome_loop_script_roundtrip(tmp_path) -> None:
    order = _order()
    comparison = EconomicWMShadowOutcomeComparison(
        comparison_id="comparison_test",
        work_order_id=order.work_order_id,
        expected_effect_keys=sorted(order.expected_effects),
    )
    shadow_report = _shadow_report(tmp_path, order)
    shadow_path = tmp_path / "shadow_report.json"
    save_economic_wm_shadow_execution_report(
        report_path=shadow_path,
        report=shadow_report,
        work_orders=[order],
        outcome_comparisons=[comparison],
    )
    supervision_manifest, supervision_records = _supervision(tmp_path)
    supervision_path = tmp_path / "supervision_manifest.json"
    save_economic_wm_supervision_substrate(
        manifest_path=supervision_path,
        manifest=supervision_manifest,
        records=supervision_records,
    )

    payload = run_economic_wm_shadow_outcome_loop(
        output_dir=tmp_path / "outcome_loop",
        shadow_execution_report_path=shadow_path,
        supervision_manifest_path=supervision_path,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["local_structural_loop_closed"] is True
    assert payload["hardware_executed"] is False
    assert payload["promotion_eligible"] is False
    assert Path(payload["artifact_refs"]["markdown_path"]).exists()
    loaded = load_economic_wm_shadow_outcome_loop_report(
        payload["artifact_refs"]["report_path"]
    )
    receipts = load_economic_wm_shadow_outcome_receipts(
        payload["artifact_refs"]["outcome_receipts_path"]
    )
    orders = load_economic_wm_shadow_work_orders(shadow_report.work_orders_path)
    assert loaded.report_id == payload["report_id"]
    assert len(receipts) == len(orders) == 1

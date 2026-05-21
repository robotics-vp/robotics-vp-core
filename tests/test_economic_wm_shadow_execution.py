from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.run_economic_wm_shadow_execution import (
    run_economic_wm_shadow_execution,
)
from src.world_model.economic_world_model import (
    EconomicWMAllocationCandidate,
    EconomicWMReplayFeatureRow,
    EconomicWMShadowAllocationEval,
    EconomicWMTrainingCorpusManifest,
    build_economic_wm_lower_wm_consumption_preflight,
    build_economic_wm_phase5_local_prep,
    build_economic_wm_resource_surfaces,
    build_economic_wm_shadow_execution_report,
    load_economic_wm_shadow_execution_report,
    load_economic_wm_shadow_outcome_comparisons,
    load_economic_wm_shadow_work_orders,
    save_economic_wm_phase5_local_prep,
    save_economic_wm_shadow_allocation_eval,
)


def _row(idx: int, *, benchmark_ready: bool) -> EconomicWMReplayFeatureRow:
    return EconomicWMReplayFeatureRow(
        row_id=f"ewm_row_shadow_exec_{idx}",
        source_episode_id=f"video_shadow_exec_{idx}:proposal_{idx}",
        video_id=f"video_shadow_exec_{idx}",
        proposal_id=f"proposal_{idx}",
        readiness_regime="scaffold_ready_training_blocked",
        benchmark_ready=benchmark_ready,
        shadow_only=not benchmark_ready,
        local_materialization_eligible=True,
        gpu_training_eligible=False,
        feature_vector={"benchmark_gate_ready": 1.0 if benchmark_ready else 0.0},
        target_vector={
            "benchmark_training_weight": 1.0 if benchmark_ready else 0.0,
            "shadow_gap_weight": 0.0 if benchmark_ready else 1.0,
            "provider_bringup_gap_weight": 1.0,
            "gpu_training_deferred_weight": 1.0,
        },
        denied_promotion_reasons=["gpu_training_not_run"],
        source_refs={
            "counterfactual_eval_path": f"artifacts/video_shadow_exec_{idx}/counterfactual_eval.json",
            "value_target_pack_path": f"artifacts/video_shadow_exec_{idx}/value_targets.json",
        },
    )


def _phase5(tmp_path) -> tuple[object, Path]:
    rows = [_row(0, benchmark_ready=True), _row(1, benchmark_ready=False)]
    corpus = EconomicWMTrainingCorpusManifest(
        corpus_id="ewm_corpus_shadow_exec",
        scaffold_id="ewm_scaffold_shadow_exec",
        row_count=2,
        benchmark_ready_count=1,
        shadow_only_count=1,
        rows_path="rows.jsonl",
        readiness_class="scaffold_ready_training_blocked",
        ready_for_training=False,
        promotion_eligible=False,
        training_blockers=["gpu_training_not_run"],
        row_ids=[row.row_id for row in rows],
    )
    lower, consumption = build_economic_wm_lower_wm_consumption_preflight(
        corpus_manifest=corpus,
        rows=rows,
        output_dir=tmp_path / "lower",
        consumption_rows_path=tmp_path / "lower" / "consumption.jsonl",
        compile_missing_refs=True,
    )
    resource, receipts, _contracts, _runbooks, telemetry = (
        build_economic_wm_resource_surfaces(
            corpus_manifest=corpus,
            rows=rows,
            receipts_path=tmp_path / "resource" / "receipts.jsonl",
            contracts_path=tmp_path / "resource" / "contracts.jsonl",
            degraded_runbooks_path=tmp_path / "resource" / "runbooks.jsonl",
            telemetry_surfaces_path=tmp_path / "resource" / "telemetry.jsonl",
        )
    )
    phase5, compositions, joins, windows = build_economic_wm_phase5_local_prep(
        corpus_manifest=corpus,
        rows=rows,
        lower_wm_preflight=lower,
        canonical_consumption_rows=consumption,
        resource_manifest=resource,
        resource_receipts=receipts,
        queue_telemetry_surfaces=telemetry,
        composition_rows_path=tmp_path / "phase5" / "compositions.jsonl",
        counterfactual_value_joins_path=tmp_path / "phase5" / "joins.jsonl",
        temporal_windows_path=tmp_path / "phase5" / "windows.jsonl",
        window_size=2,
    )
    phase5_path = tmp_path / "phase5" / "manifest.json"
    save_economic_wm_phase5_local_prep(
        manifest_path=phase5_path,
        manifest=phase5,
        composition_rows=compositions,
        counterfactual_value_joins=joins,
        temporal_windows=windows,
    )
    return phase5, phase5_path


def _allocation_eval() -> EconomicWMShadowAllocationEval:
    candidates = [
        EconomicWMAllocationCandidate(
            candidate_id="candidate_provider_contracts",
            label="prepare_teacher_provider_evidence_contracts",
            allowed=True,
            expected_value=0.9,
            resource_request={"local_dev_budget": 0.3, "gpu_budget": 0.0},
            rationale="Prepare provider evidence contracts.",
        ),
        EconomicWMAllocationCandidate(
            candidate_id="candidate_shadow_gap",
            label="close_shadow_gap_replay",
            allowed=True,
            expected_value=0.4,
            resource_request={"local_dev_budget": 0.35, "gpu_budget": 0.0},
            rationale="Close shadow row gaps.",
        ),
        EconomicWMAllocationCandidate(
            candidate_id="candidate_gpu_training",
            label="run_gpu_training",
            allowed=False,
            expected_value=0.0,
            resource_request={"gpu_budget": 1.0},
            rationale="Denied until GPU/provider evidence exists.",
            denial_reasons=["gpu_training_not_run"],
        ),
    ]
    return EconomicWMShadowAllocationEval(
        eval_id="ewm_shadow_alloc_eval_test",
        scaffold_id="ewm_scaffold_shadow_exec",
        corpus_id="ewm_corpus_shadow_exec",
        allocation_envelope_id="allocation_envelope_test",
        recommended_candidate="prepare_teacher_provider_evidence_contracts",
        candidates=candidates,
        row_count=2,
        benchmark_ready_count=1,
        shadow_only_count=1,
        ready_for_training=False,
        promotion_eligible=False,
        training_blockers=["gpu_training_not_run"],
    )


def _trainer_manifest(tmp_path) -> tuple[dict, Path]:
    payload = {
        "trainer_scaffold_id": "ewm_trainer_scaffold_test",
        "authority_class": "trainer_scaffold_only",
        "dataset_contract_ready": True,
        "cpu_smoke_forward_passed": True,
        "training_executed": False,
        "weights_written": False,
        "promotion_eligible": False,
        "reward_math_mutation": False,
    }
    path = tmp_path / "trainer" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload, path


def test_shadow_execution_builds_advisory_work_orders(tmp_path) -> None:
    phase5, _phase5_path = _phase5(tmp_path)
    allocation = _allocation_eval()
    trainer, _trainer_path = _trainer_manifest(tmp_path)

    report, orders, comparisons = build_economic_wm_shadow_execution_report(
        phase5_manifest=phase5,
        allocation_eval=allocation,
        trainer_scaffold_manifest=trainer,
        work_orders_path="orders.jsonl",
        outcome_comparisons_path="comparisons.jsonl",
    )

    assert report.version == "economic_wm_shadow_execution_report_v1"
    assert report.status == "ok"
    assert report.ready_for_shadow_comparison is True
    assert report.live_policy_control is False
    assert report.reward_math_mutation is False
    assert report.promotion_eligible is False
    assert report.work_order_count == 2
    assert report.outcome_comparison_count == 2
    assert orders[0].recommended_action == "prepare_non_stub_provider_receipt_runbook"
    assert orders[0].authority_class == "shadow_work_order_only"
    assert "live_policy_control" in orders[0].denied_authority
    assert orders[0].resource_request["gpu_budget"] == 0.0
    assert comparisons[0].comparison_status == "awaiting_outcome_receipts"


def test_shadow_execution_script_roundtrip(tmp_path) -> None:
    _phase5_manifest, phase5_path = _phase5(tmp_path)
    allocation = _allocation_eval()
    allocation_path = tmp_path / "allocation" / "eval.json"
    save_economic_wm_shadow_allocation_eval(allocation_path, allocation)
    _trainer, trainer_path = _trainer_manifest(tmp_path)

    payload = run_economic_wm_shadow_execution(
        output_dir=tmp_path / "shadow_execution",
        phase5_prep_path=phase5_path,
        allocation_eval_path=allocation_path,
        trainer_scaffold_path=trainer_path,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["ready_for_shadow_comparison"] is True
    assert payload["live_policy_control"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["promotion_eligible"] is False
    assert Path(payload["artifact_refs"]["markdown_path"]).exists()

    loaded = load_economic_wm_shadow_execution_report(
        payload["artifact_refs"]["report_path"]
    )
    orders = load_economic_wm_shadow_work_orders(
        payload["artifact_refs"]["work_orders_path"]
    )
    comparisons = load_economic_wm_shadow_outcome_comparisons(
        payload["artifact_refs"]["outcome_comparisons_path"]
    )
    assert loaded.report_id == payload["report_id"]
    assert len(orders) == len(comparisons) == 2

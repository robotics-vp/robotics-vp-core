from __future__ import annotations

from pathlib import Path

from scripts.economic_world_model.prepare_economic_wm_resource_surfaces import (
    run_prepare_economic_wm_resource_surfaces,
)
from src.world_model.economic_world_model import (
    EconomicWMReplayFeatureRow,
    EconomicWMTrainingCorpusManifest,
    build_economic_wm_resource_surfaces,
    load_economic_wm_companion_compute_contracts,
    load_economic_wm_degraded_mode_runbooks,
    load_economic_wm_queue_telemetry_surfaces,
    load_economic_wm_resource_ingestion_manifest,
    load_economic_wm_resource_receipts,
    save_economic_wm_training_corpus,
)


def _row(row_id: str = "ewm_row_resource") -> EconomicWMReplayFeatureRow:
    return EconomicWMReplayFeatureRow(
        row_id=row_id,
        source_episode_id="video_resource:proposal_0",
        video_id="video_resource",
        proposal_id="proposal_0",
        readiness_regime="scaffold_ready_training_blocked",
        benchmark_ready=True,
        shadow_only=False,
        local_materialization_eligible=True,
        gpu_training_eligible=False,
        feature_vector={"benchmark_gate_ready": 1.0, "provider_friction": 1.0},
        target_vector={
            "benchmark_training_weight": 1.0,
            "shadow_gap_weight": 0.0,
            "provider_bringup_gap_weight": 1.0,
            "gpu_training_deferred_weight": 1.0,
        },
        denied_promotion_reasons=[
            "gpu_training_not_run",
            "provider_bringup_not_run",
        ],
        source_refs={
            "canonical_lower_wm_refs": {
                "perception_grounding": {"artifact_path": "perception.json"},
                "sim_synth_physics": {"artifact_path": "sim.json"},
                "embodiment_actuation": {"artifact_path": "embodiment.json"},
            },
            "counterfactual_eval_path": "counterfactual.json",
            "value_target_pack_path": "value_targets.json",
        },
    )


def _manifest(rows_path: str) -> EconomicWMTrainingCorpusManifest:
    return EconomicWMTrainingCorpusManifest(
        corpus_id="ewm_corpus_resource",
        scaffold_id="ewm_scaffold_resource",
        row_count=1,
        benchmark_ready_count=1,
        shadow_only_count=0,
        rows_path=rows_path,
        readiness_class="scaffold_ready_training_blocked",
        ready_for_training=False,
        promotion_eligible=False,
        training_blockers=["gpu_training_not_run", "provider_bringup_not_run"],
        row_ids=["ewm_row_resource"],
    )


def test_resource_surfaces_define_compute_receipt_slots() -> None:
    manifest, receipts, contracts, runbooks, telemetry = (
        build_economic_wm_resource_surfaces(
            corpus_manifest=_manifest("rows.jsonl"),
            rows=[_row()],
            receipts_path="receipts.jsonl",
            contracts_path="contracts.jsonl",
            degraded_runbooks_path="runbooks.jsonl",
            telemetry_surfaces_path="telemetry.jsonl",
        )
    )

    assert manifest.version == "economic_wm_resource_ingestion_manifest_v1"
    assert manifest.status == "ok"
    assert manifest.ready_for_phase5_local_prep is True
    assert manifest.ready_for_training is False
    assert manifest.promotion_eligible is False
    assert manifest.reward_math_mutation is False
    assert manifest.receipt_count == 1
    assert manifest.contract_count == 1
    assert manifest.runbook_count == 1
    assert manifest.telemetry_surface_count == 1
    assert "capacity_receipts" in manifest.economic_wm_ingestion_slots
    assert "battery_receipts" in manifest.economic_wm_ingestion_slots
    assert "inferential_work_order_budget" in manifest.allocatable_budget_objects

    receipt = receipts[0]
    assert receipt.authority_class == "resource_receipt_schema_only"
    assert receipt.capacity_units["gpu_training_budget"] == 0.0
    assert receipt.latency_ms["shadow_planner_cycle_ms"] > 0.0
    assert receipt.thermal_headroom["gpu_training_headroom"] == 0.0
    assert receipt.battery_reserve["minimum_reserve_fraction"] > 0.0
    assert receipt.telemetry_quality["gpu_runtime_truth"] == 0.0
    assert receipt.promotion_eligible is False

    contract = contracts[0]
    assert contract.compute_plane == "local_cpu_with_future_companion_slot"
    assert contract.live_policy_control is False
    assert contract.reward_math_mutation is False
    assert contract.control_split["policy_runtime"] == "no_live_policy_control"

    assert "live_policy_control" in runbooks[0].denied_modes
    assert telemetry[0].ready_for_shadow_execution is True
    assert telemetry[0].work_order_backlog["train_gpu_model"] == 0.0


def test_prepare_resource_surfaces_script_roundtrip(tmp_path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(str(rows_path))
    row = _row()
    save_economic_wm_training_corpus(
        manifest_path=manifest_path,
        rows_path=rows_path,
        manifest=manifest,
        rows=[row],
    )

    payload = run_prepare_economic_wm_resource_surfaces(
        output_dir=tmp_path / "resource_surfaces",
        corpus_manifest_path=manifest_path,
        rows_path=rows_path,
        run_rows_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["ready_for_phase5_local_prep"] is True
    assert payload["ready_for_training"] is False
    assert payload["promotion_eligible"] is False
    assert Path(payload["artifact_refs"]["markdown_path"]).exists()

    loaded = load_economic_wm_resource_ingestion_manifest(
        payload["artifact_refs"]["manifest_path"]
    )
    receipts = load_economic_wm_resource_receipts(
        payload["artifact_refs"]["receipts_path"]
    )
    contracts = load_economic_wm_companion_compute_contracts(
        payload["artifact_refs"]["contracts_path"]
    )
    runbooks = load_economic_wm_degraded_mode_runbooks(
        payload["artifact_refs"]["degraded_runbooks_path"]
    )
    telemetry = load_economic_wm_queue_telemetry_surfaces(
        payload["artifact_refs"]["telemetry_surfaces_path"]
    )
    assert loaded.manifest_id == payload["manifest_id"]
    assert len(receipts) == len(contracts) == len(runbooks) == len(telemetry) == 1

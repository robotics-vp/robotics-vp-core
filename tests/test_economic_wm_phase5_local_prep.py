from __future__ import annotations

from pathlib import Path

from scripts.economic_world_model.prepare_economic_wm_phase5_local_prep import (
    run_prepare_economic_wm_phase5_local_prep,
)
from src.world_model.economic_world_model import (
    EconomicWMReplayFeatureRow,
    EconomicWMTrainingCorpusManifest,
    build_economic_wm_lower_wm_consumption_preflight,
    build_economic_wm_phase5_local_prep,
    build_economic_wm_resource_surfaces,
    load_economic_wm_counterfactual_value_join_rows,
    load_economic_wm_datapack_composition_rows,
    load_economic_wm_phase5_local_prep_manifest,
    load_economic_wm_temporal_window_rows,
    save_economic_wm_lower_wm_consumption_outputs,
    save_economic_wm_resource_surfaces,
    save_economic_wm_training_corpus,
)


def _row(idx: int, *, benchmark_ready: bool) -> EconomicWMReplayFeatureRow:
    return EconomicWMReplayFeatureRow(
        row_id=f"ewm_row_phase5_{idx}",
        source_episode_id=f"video_phase5_{idx}:proposal_{idx}",
        video_id=f"video_phase5_{idx}",
        proposal_id=f"proposal_{idx}",
        readiness_regime="scaffold_ready_training_blocked",
        benchmark_ready=benchmark_ready,
        shadow_only=not benchmark_ready,
        local_materialization_eligible=True,
        gpu_training_eligible=False,
        feature_vector={
            "benchmark_gate_ready": 1.0 if benchmark_ready else 0.0,
            "provider_friction": 1.0,
        },
        target_vector={
            "benchmark_training_weight": 1.0 if benchmark_ready else 0.0,
            "shadow_gap_weight": 0.0 if benchmark_ready else 1.0,
            "provider_bringup_gap_weight": 1.0,
            "gpu_training_deferred_weight": 1.0,
        },
        denied_promotion_reasons=[
            "gpu_training_not_run",
            "provider_bringup_not_run",
        ],
        source_refs={
            "runtime_packet_path": f"artifacts/video_phase5_{idx}/runtime_packet.json",
            "counterfactual_eval_path": f"artifacts/video_phase5_{idx}/counterfactual_eval.json",
            "value_target_pack_path": f"artifacts/video_phase5_{idx}/value_targets.json",
            "value_ledger_receipt_path": f"artifacts/video_phase5_{idx}/value_ledger.json",
        },
    )


def _rows() -> list[EconomicWMReplayFeatureRow]:
    return [_row(0, benchmark_ready=True), _row(1, benchmark_ready=False)]


def _manifest(rows_path: str) -> EconomicWMTrainingCorpusManifest:
    return EconomicWMTrainingCorpusManifest(
        corpus_id="ewm_corpus_phase5",
        scaffold_id="ewm_scaffold_phase5",
        row_count=2,
        benchmark_ready_count=1,
        shadow_only_count=1,
        rows_path=rows_path,
        readiness_class="scaffold_ready_training_blocked",
        ready_for_training=False,
        promotion_eligible=False,
        training_blockers=["gpu_training_not_run", "provider_bringup_not_run"],
        row_ids=["ewm_row_phase5_0", "ewm_row_phase5_1"],
    )


def _deps(tmp_path):
    rows = _rows()
    corpus = _manifest("rows.jsonl")
    lower_preflight, consumption_rows = (
        build_economic_wm_lower_wm_consumption_preflight(
            corpus_manifest=corpus,
            rows=rows,
            output_dir=tmp_path / "lower",
            consumption_rows_path=tmp_path / "lower" / "consumption.jsonl",
            compile_missing_refs=True,
        )
    )
    resource_manifest, receipts, contracts, runbooks, telemetry = (
        build_economic_wm_resource_surfaces(
            corpus_manifest=corpus,
            rows=rows,
            receipts_path=tmp_path / "resource" / "receipts.jsonl",
            contracts_path=tmp_path / "resource" / "contracts.jsonl",
            degraded_runbooks_path=tmp_path / "resource" / "runbooks.jsonl",
            telemetry_surfaces_path=tmp_path / "resource" / "telemetry.jsonl",
        )
    )
    return (
        corpus,
        rows,
        lower_preflight,
        consumption_rows,
        resource_manifest,
        receipts,
        telemetry,
    )


def test_phase5_local_prep_builds_composition_windows_and_joins(tmp_path) -> None:
    (
        corpus,
        rows,
        lower_preflight,
        consumption_rows,
        resource_manifest,
        receipts,
        telemetry,
    ) = _deps(tmp_path)

    manifest, compositions, joins, windows = build_economic_wm_phase5_local_prep(
        corpus_manifest=corpus,
        rows=rows,
        lower_wm_preflight=lower_preflight,
        canonical_consumption_rows=consumption_rows,
        resource_manifest=resource_manifest,
        resource_receipts=receipts,
        queue_telemetry_surfaces=telemetry,
        composition_rows_path="compositions.jsonl",
        counterfactual_value_joins_path="joins.jsonl",
        temporal_windows_path="windows.jsonl",
        window_size=2,
    )

    assert manifest.version == "economic_wm_phase5_local_prep_manifest_v1"
    assert manifest.status == "ok"
    assert manifest.ready_for_trainer_scaffold is True
    assert manifest.ready_for_gpu_training is False
    assert manifest.promotion_eligible is False
    assert manifest.reward_math_mutation is False
    assert manifest.composition_row_count == 2
    assert manifest.counterfactual_value_join_count == 2
    assert manifest.temporal_window_count == 1

    first = compositions[0]
    assert first.authority_class == "datapack_composition_row_only"
    assert first.material_provenance_composition["perception_grounding_state"] == 1.0
    assert first.material_provenance_composition["resource_budget_receipt"] == 1.0
    assert first.functional_contribution_composition["counterfactual_value_join"] == 1.0
    assert first.feature_vector["composition_lower_wm_ref_fraction"] == 1.0
    assert first.target_vector["target_resource_budget_weight"] == 1.0

    assert joins[0].join_status == "structural_join_ready"
    assert joins[0].ready_for_trainer_scaffold is True
    assert windows[0].benchmark_ready_count == 1
    assert windows[0].shadow_only_count == 1
    assert windows[0].ready_for_trainer_scaffold is True


def test_phase5_local_prep_script_roundtrip(tmp_path) -> None:
    (
        corpus,
        rows,
        lower_preflight,
        consumption_rows,
        resource_manifest,
        receipts,
        telemetry,
    ) = _deps(tmp_path)
    rows_path = tmp_path / "rows.jsonl"
    corpus_path = tmp_path / "corpus.json"
    lower_path = tmp_path / "lower" / "preflight.json"
    consumption_path = tmp_path / "lower" / "consumption.jsonl"
    resource_path = tmp_path / "resource" / "manifest.json"
    save_economic_wm_training_corpus(
        manifest_path=corpus_path,
        rows_path=rows_path,
        manifest=corpus,
        rows=rows,
    )
    save_economic_wm_lower_wm_consumption_outputs(
        preflight_path=lower_path,
        consumption_rows_path=consumption_path,
        preflight=lower_preflight,
        consumption_rows=consumption_rows,
    )
    save_economic_wm_resource_surfaces(
        manifest_path=resource_path,
        manifest=resource_manifest,
        receipts=receipts,
        contracts=[],
        runbooks=[],
        telemetry_surfaces=telemetry,
    )

    payload = run_prepare_economic_wm_phase5_local_prep(
        output_dir=tmp_path / "phase5",
        corpus_manifest_path=corpus_path,
        rows_path=rows_path,
        lower_wm_preflight_path=lower_path,
        canonical_consumption_rows_path=consumption_path,
        resource_manifest_path=resource_path,
        resource_receipts_path=resource_manifest.receipts_path,
        queue_telemetry_surfaces_path=resource_manifest.telemetry_surfaces_path,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["ready_for_trainer_scaffold"] is True
    assert payload["ready_for_gpu_training"] is False
    assert payload["promotion_eligible"] is False
    assert Path(payload["artifact_refs"]["markdown_path"]).exists()

    loaded = load_economic_wm_phase5_local_prep_manifest(
        payload["artifact_refs"]["manifest_path"]
    )
    compositions = load_economic_wm_datapack_composition_rows(
        payload["artifact_refs"]["composition_rows_path"]
    )
    joins = load_economic_wm_counterfactual_value_join_rows(
        payload["artifact_refs"]["counterfactual_value_joins_path"]
    )
    windows = load_economic_wm_temporal_window_rows(
        payload["artifact_refs"]["temporal_windows_path"]
    )
    assert loaded.manifest_id == payload["manifest_id"]
    assert len(compositions) == len(joins) == 2
    assert len(windows) == 1

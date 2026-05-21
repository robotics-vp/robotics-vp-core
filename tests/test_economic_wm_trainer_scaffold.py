from __future__ import annotations

import json
from pathlib import Path

from scripts.train_economic_world_model_v0 import (
    run_train_economic_world_model_v0_scaffold,
)
from src.world_model.economic_world_model import (
    EconomicWMReplayFeatureRow,
    EconomicWMTrainingCorpusManifest,
    build_economic_wm_lower_wm_consumption_preflight,
    build_economic_wm_neural_architecture_manifest,
    build_economic_wm_phase5_local_prep,
    build_economic_wm_resource_surfaces,
    save_economic_wm_neural_architecture_manifest,
    save_economic_wm_phase5_local_prep,
)


def _row(idx: int, *, benchmark_ready: bool) -> EconomicWMReplayFeatureRow:
    return EconomicWMReplayFeatureRow(
        row_id=f"ewm_row_trainer_{idx}",
        source_episode_id=f"video_trainer_{idx}:proposal_{idx}",
        video_id=f"video_trainer_{idx}",
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
            "counterfactual_eval_path": f"artifacts/video_trainer_{idx}/counterfactual_eval.json",
            "value_target_pack_path": f"artifacts/video_trainer_{idx}/value_targets.json",
            "value_ledger_receipt_path": f"artifacts/video_trainer_{idx}/value_ledger.json",
        },
    )


def _artifacts(tmp_path) -> tuple[Path, Path]:
    rows = [_row(0, benchmark_ready=True), _row(1, benchmark_ready=False)]
    corpus = EconomicWMTrainingCorpusManifest(
        corpus_id="ewm_corpus_trainer",
        scaffold_id="ewm_scaffold_trainer",
        row_count=2,
        benchmark_ready_count=1,
        shadow_only_count=1,
        rows_path="rows.jsonl",
        readiness_class="scaffold_ready_training_blocked",
        ready_for_training=False,
        promotion_eligible=False,
        training_blockers=["gpu_training_not_run", "provider_bringup_not_run"],
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
    neural = build_economic_wm_neural_architecture_manifest(lower_wm_preflight=lower)
    neural_path = tmp_path / "neural" / "manifest.json"
    save_economic_wm_neural_architecture_manifest(neural_path, neural)
    return phase5_path, neural_path


def test_trainer_scaffold_emits_shape_checked_non_training_manifest(tmp_path) -> None:
    phase5_path, neural_path = _artifacts(tmp_path)

    payload = run_train_economic_world_model_v0_scaffold(
        output_dir=tmp_path / "trainer",
        phase5_prep_path=phase5_path,
        neural_manifest_path=neural_path,
        run_dependencies_if_missing=False,
    )

    assert payload["version"] == "economic_wm_trainer_scaffold_manifest_v1"
    assert payload["authority_class"] == "trainer_scaffold_only"
    assert payload["dataset_contract_ready"] is True
    assert payload["cpu_smoke_forward_passed"] is True
    assert payload["losses_defined"] is True
    assert payload["training_executed"] is False
    assert payload["weights_written"] is False
    assert payload["ready_for_gpu_training"] is False
    assert payload["promotion_eligible"] is False
    assert payload["reward_math_mutation"] is False
    assert "gpu_training_not_run" in payload["blockers"]

    dataset = json.loads(
        Path(payload["artifact_refs"]["dataset_contract_path"]).read_text()
    )
    model_config = json.loads(
        Path(payload["artifact_refs"]["model_component_config_path"]).read_text()
    )
    smoke = json.loads(
        Path(payload["artifact_refs"]["cpu_smoke_forward_path"]).read_text()
    )
    assert dataset["shape_contracts"]["composition_feature_dim"] > 0
    assert dataset["shape_contracts"]["temporal_feature_dim"] > 0
    assert model_config["component_count"] == 6
    assert all(not item["training_enabled"] for item in model_config["components"])
    assert smoke["cpu_smoke_forward_passed"] is True
    assert all(item["shape_check_passed"] for item in smoke["component_reports"])
    assert Path(payload["artifact_refs"]["markdown_path"]).exists()

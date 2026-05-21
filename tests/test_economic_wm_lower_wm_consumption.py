from __future__ import annotations

from pathlib import Path

from scripts.economic_world_model.prepare_economic_wm_lower_wm_consumption_preflight import (
    run_prepare_economic_wm_lower_wm_consumption_preflight,
)
from src.world_model.economic_world_model import (
    EconomicWMReplayFeatureRow,
    EconomicWMTrainingCorpusManifest,
    build_economic_wm_lower_wm_consumption_preflight,
    load_economic_wm_canonical_consumption_rows,
    load_economic_wm_lower_wm_consumption_preflight,
    save_economic_wm_training_corpus,
)


def _sample_row(row_id: str = "ewm_row_test") -> EconomicWMReplayFeatureRow:
    return EconomicWMReplayFeatureRow(
        row_id=row_id,
        source_episode_id="video_test:proposal_0",
        video_id="video_test",
        proposal_id="proposal_0",
        readiness_regime="scaffold_ready_training_blocked",
        benchmark_ready=True,
        shadow_only=False,
        local_materialization_eligible=True,
        gpu_training_eligible=False,
        feature_vector={
            "benchmark_gate_ready": 1.0,
            "replay_export_flow": 1.0,
            "provider_friction": 1.0,
            "gpu_training_friction": 1.0,
        },
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
            "runtime_packet_path": "artifacts/video_test/runtime_packet.json",
            "governance_trace_path": "artifacts/video_test/governance_trace.json",
            "value_target_pack_path": "artifacts/video_test/value_targets.json",
        },
        metadata={"boundary": "test row only"},
    )


def _manifest(rows_path: str | Path) -> EconomicWMTrainingCorpusManifest:
    return EconomicWMTrainingCorpusManifest(
        corpus_id="ewm_corpus_test",
        scaffold_id="ewm_scaffold_test",
        row_count=1,
        benchmark_ready_count=1,
        shadow_only_count=0,
        rows_path=str(rows_path),
        readiness_class="scaffold_ready_training_blocked",
        ready_for_training=False,
        promotion_eligible=False,
        training_blockers=["gpu_training_not_run", "provider_bringup_not_run"],
        row_ids=["ewm_row_test"],
        metadata={"boundary": "test manifest only"},
    )


def test_lower_wm_consumption_preflight_compiles_canonical_reference_pack(
    tmp_path,
) -> None:
    row = _sample_row()
    preflight, consumption_rows = build_economic_wm_lower_wm_consumption_preflight(
        corpus_manifest=_manifest("rows.jsonl"),
        rows=[row],
        output_dir=tmp_path / "out",
        consumption_rows_path=tmp_path / "out" / "consumption_rows.jsonl",
        compile_missing_refs=True,
    )

    assert preflight.status == "ok"
    assert preflight.all_required_wms_referenced is True
    assert preflight.ready_for_neural_manifest is True
    assert preflight.ready_for_training is False
    assert preflight.promotion_eligible is False
    assert preflight.compiled_reference_count == 3
    assert preflight.direct_reference_count == 0
    assert preflight.summary_only_reference_count == 0

    consumption_row = consumption_rows[0]
    assert consumption_row.ready_for_neural_manifest is True
    canonical_refs = consumption_row.training_row["source_refs"][
        "canonical_lower_wm_refs"
    ]
    assert set(canonical_refs) == {
        "perception_grounding",
        "sim_synth_physics",
        "embodiment_actuation",
    }
    for ref in consumption_row.canonical_refs:
        assert ref.reference_status == "compiled_local_reference"
        assert ref.satisfied is True
        assert ref.summary_only is False
        assert Path(ref.artifact_path).exists()
        assert ref.observed_version == ref.expected_version


def test_lower_wm_consumption_preflight_fails_without_canonical_refs_when_compile_disabled(
    tmp_path,
) -> None:
    preflight, consumption_rows = build_economic_wm_lower_wm_consumption_preflight(
        corpus_manifest=_manifest("rows.jsonl"),
        rows=[_sample_row()],
        output_dir=tmp_path / "out",
        consumption_rows_path=tmp_path / "out" / "consumption_rows.jsonl",
        compile_missing_refs=False,
    )

    assert preflight.status == "failed"
    assert preflight.all_required_wms_referenced is False
    assert preflight.ready_for_neural_manifest is False
    assert preflight.missing_reference_count == 3
    assert preflight.summary_only_reference_count == 3
    assert consumption_rows[0].ready_for_neural_manifest is False
    assert "perception_grounding_canonical_state_ref_missing" in preflight.blockers
    assert "sim_synth_physics_canonical_state_ref_missing" in preflight.blockers
    assert "embodiment_actuation_canonical_state_ref_missing" in preflight.blockers


def test_prepare_lower_wm_consumption_preflight_script_roundtrip(tmp_path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    manifest_path = tmp_path / "manifest.json"
    row = _sample_row()
    save_economic_wm_training_corpus(
        manifest_path=manifest_path,
        rows_path=rows_path,
        manifest=_manifest(rows_path),
        rows=[row],
    )

    payload = run_prepare_economic_wm_lower_wm_consumption_preflight(
        output_dir=tmp_path / "lower_wm_preflight",
        corpus_manifest_path=manifest_path,
        rows_path=rows_path,
        run_rows_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["all_required_wms_referenced"] is True
    assert payload["ready_for_neural_manifest"] is True
    assert payload["ready_for_training"] is False
    assert payload["promotion_eligible"] is False
    assert payload["aggregate_counts"]["compiled_reference_count"] == 3.0
    assert Path(payload["artifact_refs"]["preflight_path"]).exists()
    assert Path(payload["artifact_refs"]["consumption_rows_path"]).exists()
    assert Path(payload["artifact_refs"]["markdown_path"]).exists()

    loaded_preflight = load_economic_wm_lower_wm_consumption_preflight(
        payload["artifact_refs"]["preflight_path"]
    )
    loaded_rows = load_economic_wm_canonical_consumption_rows(
        payload["artifact_refs"]["consumption_rows_path"]
    )
    assert loaded_preflight.preflight_id == payload["preflight_id"]
    assert len(loaded_rows) == 1
    assert (
        loaded_rows[0].canonical_refs[0].reference_status == "compiled_local_reference"
    )

from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.sweep_economic_wm_lower_wm_maturity import (
    run_sweep_economic_wm_lower_wm_maturity,
)
from src.world_model.economic_world_model import (
    EconomicWMCanonicalConsumptionRow,
    EconomicWMDatapackCompositionRow,
    EconomicWMLowerWMConsumptionPreflight,
    EconomicWMLowerWMReference,
    EconomicWMPhase5LocalPrepManifest,
    EconomicWMResourceIngestionManifest,
    build_economic_wm_lower_wm_maturity_sweep,
    load_economic_wm_lower_wm_maturity_rows,
    load_economic_wm_lower_wm_maturity_sweep,
    save_economic_wm_lower_wm_consumption_outputs,
    save_economic_wm_phase5_local_prep,
    save_economic_wm_resource_surfaces,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _state_ref(tmp_path, key: str, version: str) -> EconomicWMLowerWMReference:
    path = tmp_path / f"{key}.json"
    _write_json(path, {"version": version, "state_id": f"state_{key}"})
    return EconomicWMLowerWMReference(
        wm_key=key,
        expected_version=version,
        artifact_path=str(path),
        observed_version=version,
        state_id=f"state_{key}",
        reference_status="direct_source_reference",
        direct_reference=True,
        summary_only=False,
    )


def _fixtures(tmp_path):
    refs = [
        _state_ref(
            tmp_path, "perception_grounding", "perception_grounding_world_state_v1"
        ),
        _state_ref(tmp_path, "sim_synth_physics", "sim_synth_physics_world_state_v1"),
        _state_ref(
            tmp_path, "embodiment_actuation", "embodiment_actuation_world_state_v1"
        ),
    ]
    consumption = EconomicWMCanonicalConsumptionRow(
        consumption_row_id="consumption_a",
        source_row_id="row_maturity",
        source_episode_id="episode_maturity",
        canonical_refs=refs,
        ready_for_neural_manifest=True,
    )
    recon_path = tmp_path / "reconstruction_grounding.json"
    benchmark_path = tmp_path / "benchmark_gate.json"
    teacher_path = tmp_path / "teacher_trace.json"
    _write_json(
        recon_path,
        {
            "version": "reconstruction_grounding_report_v1",
            "training_eligible": True,
            "benchmark_ready": True,
            "quality": {
                "calibration_complete": 1.0,
                "calibration_score": 1.0,
            },
            "metadata": {"scene_tracks_backend": "real"},
        },
    )
    _write_json(benchmark_path, {"version": "benchmark_gate_v1", "ready": True})
    _write_json(
        teacher_path,
        {
            "version": "teacher_trace_v1",
            "summary": {"teacher_confidence_mean": 0.0},
        },
    )
    composition = EconomicWMDatapackCompositionRow(
        composition_row_id="composition_a",
        source_row_id="row_maturity",
        source_episode_id="episode_maturity",
        lower_wm_refs={ref.wm_key: ref.to_dict() for ref in refs},
        resource_receipt_ref="receipt_a",
        counterfactual_value_join_ref="join_a",
        ready_for_trainer_scaffold=True,
        source_refs={
            "reconstruction_grounding_report_path": str(recon_path),
            "benchmark_gate_path": str(benchmark_path),
            "teacher_trace_path": str(teacher_path),
        },
    )
    lower = EconomicWMLowerWMConsumptionPreflight(
        preflight_id="lower_maturity",
        corpus_id="corpus_maturity",
        row_count=1,
        consumption_rows_path=str(tmp_path / "consumption.jsonl"),
        status="ok",
        all_required_wms_referenced=True,
        ready_for_neural_manifest=True,
        direct_reference_count=3,
        consumption_row_ids=["consumption_a"],
    )
    phase5 = EconomicWMPhase5LocalPrepManifest(
        manifest_id="phase5_maturity",
        corpus_id="corpus_maturity",
        lower_wm_preflight_id=lower.preflight_id,
        resource_ingestion_manifest_id="resource_maturity",
        row_count=1,
        composition_row_count=1,
        counterfactual_value_join_count=1,
        temporal_window_count=1,
        composition_rows_path=str(tmp_path / "compositions.jsonl"),
        counterfactual_value_joins_path=str(tmp_path / "joins.jsonl"),
        temporal_windows_path=str(tmp_path / "windows.jsonl"),
        status="ok",
        ready_for_trainer_scaffold=True,
    )
    resource = EconomicWMResourceIngestionManifest(
        manifest_id="resource_maturity",
        corpus_id="corpus_maturity",
        row_count=1,
        receipt_count=1,
        contract_count=1,
        runbook_count=1,
        telemetry_surface_count=1,
        receipts_path=str(tmp_path / "receipts.jsonl"),
        contracts_path=str(tmp_path / "contracts.jsonl"),
        degraded_runbooks_path=str(tmp_path / "runbooks.jsonl"),
        telemetry_surfaces_path=str(tmp_path / "telemetry.jsonl"),
        status="ok",
        ready_for_phase5_local_prep=True,
    )
    return phase5, lower, [consumption], [composition], resource


def test_lower_wm_maturity_sweep_distinguishes_structural_from_production(
    tmp_path,
) -> None:
    phase5, lower, consumptions, compositions, resource = _fixtures(tmp_path)

    sweep, rows = build_economic_wm_lower_wm_maturity_sweep(
        phase5_manifest=phase5,
        lower_wm_preflight=lower,
        consumption_rows=consumptions,
        composition_rows=compositions,
        resource_manifest=resource,
        maturity_rows_path="maturity.jsonl",
    )

    assert sweep.version == "economic_wm_lower_wm_maturity_sweep_v1"
    assert sweep.status == "ok"
    assert sweep.ready_for_phase6_contracts is True
    assert sweep.ready_for_production is False
    assert sweep.promotion_eligible is False
    assert sweep.maturity_row_count == 3
    assert rows[0].artifact_exists is True
    assert rows[0].ready_for_phase6_contracts is True
    assert rows[0].sidecar_scores["calibration_complete"] == 1.0
    assert rows[0].sidecar_scores["real_scene_tracks_joined"] == 1.0
    assert "teacher_runtime_unavailable" in rows[0].blockers


def test_lower_wm_maturity_sweep_script_roundtrip(tmp_path) -> None:
    phase5, lower, consumptions, compositions, resource = _fixtures(tmp_path)
    phase5_path = tmp_path / "phase5.json"
    lower_path = tmp_path / "lower.json"
    consumption_path = tmp_path / "consumption.jsonl"
    resource_path = tmp_path / "resource.json"
    save_economic_wm_phase5_local_prep(
        manifest_path=phase5_path,
        manifest=phase5,
        composition_rows=compositions,
        counterfactual_value_joins=[],
        temporal_windows=[],
    )
    save_economic_wm_lower_wm_consumption_outputs(
        preflight_path=lower_path,
        consumption_rows_path=consumption_path,
        preflight=lower,
        consumption_rows=consumptions,
    )
    save_economic_wm_resource_surfaces(
        manifest_path=resource_path,
        manifest=resource,
        receipts=[],
        contracts=[],
        runbooks=[],
        telemetry_surfaces=[],
    )

    payload = run_sweep_economic_wm_lower_wm_maturity(
        output_dir=tmp_path / "sweep",
        phase5_prep_path=phase5_path,
        lower_wm_preflight_path=lower_path,
        canonical_consumption_rows_path=consumption_path,
        resource_manifest_path=resource_path,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["ready_for_phase6_contracts"] is True
    assert payload["promotion_eligible"] is False
    assert Path(payload["artifact_refs"]["markdown_path"]).exists()
    loaded = load_economic_wm_lower_wm_maturity_sweep(
        payload["artifact_refs"]["sweep_path"]
    )
    rows = load_economic_wm_lower_wm_maturity_rows(
        payload["artifact_refs"]["maturity_rows_path"]
    )
    assert loaded.sweep_id == payload["sweep_id"]
    assert len(rows) == 3

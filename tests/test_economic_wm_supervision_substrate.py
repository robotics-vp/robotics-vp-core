from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.prepare_economic_wm_supervision_substrate import (
    run_prepare_economic_wm_supervision_substrate,
)
from src.economics.counterfactual_eval import build_counterfactual_eval
from src.economics.value_targets import build_value_target_pack
from src.world_model.economic_world_model import (
    EconomicWMCounterfactualValueJoinRow,
    EconomicWMPhase5LocalPrepManifest,
    build_economic_wm_supervision_substrate,
    load_economic_wm_supervision_manifest,
    load_economic_wm_supervision_records,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fixtures(tmp_path):
    cf = build_counterfactual_eval(
        run_id="run_sup",
        episode_id="episode_sup",
        timestamp="1700000000.0",
        runtime_packet_id="runtime_sup",
        objective_profile_id="objective_sup",
        baseline_value=1.0,
        branch_values=[
            {"label": "collect_more_data", "expected_net_value": 1.4},
            {"label": "defer_provider", "expected_net_value": 0.8},
        ],
    )
    pack = build_value_target_pack(
        run_id="run_sup",
        episode_id="episode_sup",
        runtime_packet_id="runtime_sup",
        base_value=1.0,
        recommended_value=1.4,
        disagreement=0.5,
        coverage=0.75,
        counterfactual_eval_id=cf.eval_id,
    )
    cf_path = tmp_path / "cf.json"
    pack_path = tmp_path / "value_targets.json"
    ledger_path = tmp_path / "ledger.json"
    _write_json(cf_path, cf.to_dict())
    _write_json(pack_path, pack.to_dict())
    _write_json(
        ledger_path, {"receipt_id": "ledger_sup", "version": "value_ledger_receipt_v1"}
    )
    join = EconomicWMCounterfactualValueJoinRow(
        join_row_id="join_sup",
        source_row_id="row_sup",
        source_episode_id="episode_sup",
        counterfactual_eval_ref=str(cf_path),
        value_target_pack_ref=str(pack_path),
        value_ledger_ref=str(ledger_path),
        join_status="structural_join_ready",
        ready_for_trainer_scaffold=True,
    )
    return join


def _phase5(tmp_path, join: EconomicWMCounterfactualValueJoinRow):
    joins_path = tmp_path / "joins.jsonl"
    joins_path.write_text(json.dumps(join.to_dict()) + "\n", encoding="utf-8")
    manifest = EconomicWMPhase5LocalPrepManifest(
        manifest_id="phase5_sup",
        corpus_id="corpus_sup",
        lower_wm_preflight_id="lower_sup",
        resource_ingestion_manifest_id="resource_sup",
        row_count=1,
        composition_row_count=1,
        counterfactual_value_join_count=1,
        temporal_window_count=1,
        composition_rows_path=str(tmp_path / "compositions.jsonl"),
        counterfactual_value_joins_path=str(joins_path),
        temporal_windows_path=str(tmp_path / "windows.jsonl"),
        status="ok",
        ready_for_trainer_scaffold=True,
    )
    path = tmp_path / "phase5.json"
    _write_json(path, manifest.to_dict())
    return manifest, path


def test_supervision_substrate_loads_typed_counterfactual_and_value_targets(
    tmp_path,
) -> None:
    join = _fixtures(tmp_path)
    phase5, _phase5_path = _phase5(tmp_path, join)

    manifest, records = build_economic_wm_supervision_substrate(
        phase5_manifest=phase5,
        join_rows=[join],
        records_path="records.jsonl",
    )

    assert manifest.version == "economic_wm_supervision_manifest_v1"
    assert manifest.status == "ok"
    assert manifest.ready_for_shadow_outcome_loop is True
    assert manifest.ready_for_training is False
    assert manifest.promotion_eligible is False
    assert records[0].counterfactual_eval_id
    assert records[0].value_target_pack_id
    assert records[0].value_ledger_receipt_id == "ledger_sup"
    assert records[0].recommended_action == "collect_more_data"
    assert records[0].candidate_count == 3
    assert records[0].value_target_count == 3
    assert records[0].target_kind_counts["route"] == 1.0
    assert records[0].counterfactual_delta_summary["max_delta_value_vs_noop"] > 0.0


def test_supervision_substrate_script_roundtrip(tmp_path) -> None:
    join = _fixtures(tmp_path)
    phase5_manifest, phase5_path = _phase5(tmp_path, join)

    payload = run_prepare_economic_wm_supervision_substrate(
        output_dir=tmp_path / "supervision",
        phase5_prep_path=phase5_path,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["ready_for_shadow_outcome_loop"] is True
    assert payload["promotion_eligible"] is False
    assert Path(payload["artifact_refs"]["markdown_path"]).exists()
    loaded = load_economic_wm_supervision_manifest(
        payload["artifact_refs"]["manifest_path"]
    )
    records = load_economic_wm_supervision_records(
        payload["artifact_refs"]["records_path"]
    )
    assert loaded.manifest_id == payload["manifest_id"]
    assert len(records) == 1

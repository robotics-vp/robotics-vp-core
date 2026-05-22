from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.prepare_phase6_transport_scaffold import (
    run_prepare_phase6_transport_scaffold,
)
from src.world_model.economic_world_model import (
    EconomicWMLowerWMMaturityRow,
    EconomicWMLowerWMMaturitySweep,
    EconomicWMPhase5LocalPrepManifest,
    save_economic_wm_lower_wm_maturity_sweep,
    save_economic_wm_phase5_local_prep,
)
from src.world_model.transport import (
    ROW_FAMILIES,
    build_per_wm_transformer_registry,
    build_wm_transport_contract_pack,
    build_wm_transport_roundtrip_receipts,
    build_wm_transport_training_rows,
    load_per_wm_transformer_registry,
    load_wm_transport_bridge_contracts,
    load_wm_transport_phase6_scaffold_report,
    load_wm_transport_roundtrip_receipts,
    load_wm_transport_training_manifest,
    load_wm_transport_training_rows,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _maturity_row(tmp_path: Path, source_row_id: str, wm_key: str, version: str):
    state_path = tmp_path / f"{source_row_id}_{wm_key}.json"
    _write_json(state_path, {"version": version, "state_id": f"state_{wm_key}"})
    return EconomicWMLowerWMMaturityRow(
        maturity_row_id=f"maturity_{source_row_id}_{wm_key}",
        source_row_id=source_row_id,
        source_episode_id="episode_transport",
        wm_key=wm_key,
        artifact_path=str(state_path),
        observed_version=version,
        state_id=f"state_{wm_key}",
        reference_status="direct_source_reference",
        direct_reference=True,
        artifact_exists=True,
        sidecar_scores={
            "calibration_complete": 1.0,
            "real_scene_tracks_joined": 1.0,
            "resource_receipt_present": 1.0,
        },
        maturity_score=0.9,
        maturity_class="local_structural_mature",
        ready_for_phase6_contracts=True,
        ready_for_production=False,
        blockers=["teacher_runtime_unavailable"],
    )


def _fixtures(tmp_path: Path):
    rows = [
        _maturity_row(
            tmp_path,
            "source_transport",
            "perception_grounding",
            "perception_grounding_world_state_v1",
        ),
        _maturity_row(
            tmp_path,
            "source_transport",
            "sim_synth_physics",
            "sim_synth_physics_world_state_v1",
        ),
        _maturity_row(
            tmp_path,
            "source_transport",
            "embodiment_actuation",
            "embodiment_actuation_world_state_v1",
        ),
    ]
    sweep = EconomicWMLowerWMMaturitySweep(
        sweep_id="maturity_sweep_transport",
        phase5_manifest_id="phase5_transport",
        lower_wm_preflight_id="lower_transport",
        resource_manifest_id="resource_transport",
        maturity_row_count=3,
        structural_ready_count=3,
        production_ready_count=0,
        maturity_rows_path=str(tmp_path / "maturity_rows.jsonl"),
        status="ok",
        ready_for_phase6_contracts=True,
        ready_for_production=False,
    )
    phase5 = EconomicWMPhase5LocalPrepManifest(
        manifest_id="phase5_transport",
        corpus_id="corpus_transport",
        lower_wm_preflight_id="lower_transport",
        resource_ingestion_manifest_id="resource_transport",
        row_count=1,
        composition_row_count=1,
        counterfactual_value_join_count=1,
        temporal_window_count=1,
        composition_rows_path=str(tmp_path / "composition_rows.jsonl"),
        counterfactual_value_joins_path=str(tmp_path / "joins.jsonl"),
        temporal_windows_path=str(tmp_path / "windows.jsonl"),
        status="ok",
        ready_for_trainer_scaffold=True,
    )
    return phase5, sweep, rows


def test_phase6_contracts_require_adjacent_bridge_and_per_wm_receivers(tmp_path):
    phase5, sweep, maturity_rows = _fixtures(tmp_path)
    pack, contracts = build_wm_transport_contract_pack(
        maturity_sweep=sweep,
        maturity_rows=maturity_rows,
        phase5_manifest=phase5,
        contract_path=tmp_path / "contracts.jsonl",
    )

    assert pack.version == "wm_transport_contract_pack_v1"
    assert pack.status == "ok"
    assert pack.contract_count == 4
    assert pack.ready_for_phase6_rows is True
    assert {contract.bridge_key for contract in contracts} == {
        "perception_grounding_to_sim_synth_physics",
        "sim_synth_physics_to_embodiment_actuation",
        "embodiment_actuation_to_economic",
        "lower_wm_bundle_to_economic",
    }
    assert all(contract.adjacent_allowed for contract in contracts)
    assert all(contract.receiver_required for contract in contracts)
    assert all(contract.exporter_required for contract in contracts)
    assert not any(contract.raw_hidden_state_transport for contract in contracts)
    assert all(contract.structurally_valid for contract in contracts)


def test_phase6_roundtrip_and_training_rows_cover_required_families(tmp_path):
    phase5, sweep, maturity_rows = _fixtures(tmp_path)
    pack, contracts = build_wm_transport_contract_pack(
        maturity_sweep=sweep,
        maturity_rows=maturity_rows,
        phase5_manifest=phase5,
        contract_path=tmp_path / "contracts.jsonl",
    )
    registry = build_per_wm_transformer_registry(
        contract_pack_id=pack.pack_id,
        contracts=contracts,
    )
    receipts = build_wm_transport_roundtrip_receipts(
        contracts=contracts,
        transformer_registry=registry,
    )
    manifest, rows = build_wm_transport_training_rows(
        contract_pack=pack,
        contracts=contracts,
        transformer_registry=registry,
        roundtrip_receipts=receipts,
        rows_path=tmp_path / "rows.jsonl",
    )

    assert registry.status == "ok"
    assert registry.exporter_count >= 4
    assert registry.receiver_count >= 3
    assert len(receipts) == 4
    assert min(receipt.aggregate_score for receipt in receipts) > 0.9
    assert manifest.status == "ok"
    assert manifest.row_count == len(ROW_FAMILIES) * len(contracts)
    assert set(manifest.row_family_counts) == set(ROW_FAMILIES)
    assert all(count == len(contracts) for count in manifest.row_family_counts.values())
    assert all(row.ready_for_trainer_scaffold for row in rows)
    assert not any(row.ready_for_training for row in rows)
    assert not any(row.promotion_eligible for row in rows)
    assert not any(row.reward_math_mutation for row in rows)


def test_phase6_transport_scaffold_script_roundtrip(tmp_path):
    phase5, sweep, maturity_rows = _fixtures(tmp_path)
    phase5_path = tmp_path / "phase5.json"
    sweep_path = tmp_path / "sweep.json"
    save_economic_wm_phase5_local_prep(
        manifest_path=phase5_path,
        manifest=phase5,
        composition_rows=[],
        counterfactual_value_joins=[],
        temporal_windows=[],
    )
    save_economic_wm_lower_wm_maturity_sweep(
        sweep_path=sweep_path,
        sweep=sweep,
        maturity_rows=maturity_rows,
    )

    payload = run_prepare_phase6_transport_scaffold(
        output_dir=tmp_path / "phase6",
        phase5_prep_path=phase5_path,
        maturity_sweep_path=sweep_path,
        maturity_rows_path=sweep.maturity_rows_path,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["contract_count"] == 4
    assert payload["roundtrip_receipt_count"] == 4
    assert payload["training_row_count"] == len(ROW_FAMILIES) * 4
    assert payload["ready_for_phase6_3_neural_scaffold"] is True
    assert payload["training_executed"] is False
    assert payload["provider_executed"] is False
    assert payload["hardware_executed"] is False
    assert payload["live_policy_control"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["promotion_eligible"] is False

    refs = payload["artifact_refs"]
    assert Path(refs["markdown_path"]).exists()
    assert len(load_wm_transport_bridge_contracts(refs["contracts_path"])) == 4
    assert load_per_wm_transformer_registry(refs["registry_path"]).status == "ok"
    assert (
        len(load_wm_transport_roundtrip_receipts(refs["roundtrip_receipts_path"])) == 4
    )
    assert (
        load_wm_transport_training_manifest(refs["training_manifest_path"]).status
        == "ok"
    )
    assert (
        len(load_wm_transport_training_rows(refs["training_rows_path"]))
        == len(ROW_FAMILIES) * 4
    )
    assert load_wm_transport_phase6_scaffold_report(refs["report_path"]).status == "ok"

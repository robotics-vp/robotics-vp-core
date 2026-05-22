from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.build_phase6_transport_neural_manifest import (
    run_build_phase6_transport_neural_manifest,
)
from scripts.economic_world_model.prepare_phase6_transport_scaffold import (
    run_prepare_phase6_transport_scaffold,
)
from scripts.train_wm_transport_bridge_v0 import (
    run_train_wm_transport_bridge_v0_scaffold,
)
from src.world_model.economic_world_model import (
    EconomicWMLowerWMMaturityRow,
    EconomicWMLowerWMMaturitySweep,
    EconomicWMPhase5LocalPrepManifest,
    save_economic_wm_lower_wm_maturity_sweep,
    save_economic_wm_phase5_local_prep,
)
from src.world_model.transport import (
    load_wm_transport_loss_ledger,
    load_wm_transport_neural_architecture_manifest,
    load_wm_transport_trainer_dataset_contract,
    load_wm_transport_trainer_scaffold_manifest,
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
        source_episode_id="episode_transport_63",
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


def _materialize_phase6_scaffold(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = [
        _maturity_row(
            tmp_path,
            "source_transport_63",
            "perception_grounding",
            "perception_grounding_world_state_v1",
        ),
        _maturity_row(
            tmp_path,
            "source_transport_63",
            "sim_synth_physics",
            "sim_synth_physics_world_state_v1",
        ),
        _maturity_row(
            tmp_path,
            "source_transport_63",
            "embodiment_actuation",
            "embodiment_actuation_world_state_v1",
        ),
    ]
    sweep = EconomicWMLowerWMMaturitySweep(
        sweep_id="maturity_sweep_transport_63",
        phase5_manifest_id="phase5_transport_63",
        lower_wm_preflight_id="lower_transport_63",
        resource_manifest_id="resource_transport_63",
        maturity_row_count=3,
        structural_ready_count=3,
        production_ready_count=0,
        maturity_rows_path=str(tmp_path / "maturity_rows.jsonl"),
        status="ok",
        ready_for_phase6_contracts=True,
        ready_for_production=False,
    )
    phase5 = EconomicWMPhase5LocalPrepManifest(
        manifest_id="phase5_transport_63",
        corpus_id="corpus_transport_63",
        lower_wm_preflight_id="lower_transport_63",
        resource_ingestion_manifest_id="resource_transport_63",
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
        maturity_rows=rows,
    )
    scaffold_dir = tmp_path / "phase6_scaffold"
    run_prepare_phase6_transport_scaffold(
        output_dir=scaffold_dir,
        phase5_prep_path=phase5_path,
        maturity_sweep_path=sweep_path,
        maturity_rows_path=sweep.maturity_rows_path,
        run_dependencies_if_missing=False,
    )
    return scaffold_dir, phase5_path, sweep_path


def test_phase63_neural_manifest_and_loss_ledger_are_scaffold_only(tmp_path) -> None:
    scaffold_dir, _, _ = _materialize_phase6_scaffold(tmp_path)
    payload = run_build_phase6_transport_neural_manifest(
        output_dir=tmp_path / "neural",
        scaffold_dir=scaffold_dir,
        run_dependencies_if_missing=False,
    )

    assert payload["component_count"] == 8
    assert payload["loss_count"] == 14
    assert payload["ready_for_trainer_scaffold"] is True
    assert payload["ready_for_gpu_training"] is False
    assert payload["training_executed"] is False
    assert payload["weights_written"] is False
    assert payload["promotion_eligible"] is False

    manifest = load_wm_transport_neural_architecture_manifest(
        payload["artifact_refs"]["neural_manifest_path"]
    )
    ledger = load_wm_transport_loss_ledger(payload["artifact_refs"]["loss_ledger_path"])
    assert manifest.architecture_stage == "phase6_3_neural_scaffold"
    assert manifest.ready_for_trainer_scaffold is True
    assert manifest.raw_hidden_state_transport is False
    assert manifest.live_policy_control is False
    assert "isomorphic_transport_bridge" in {
        component.component_key for component in manifest.components
    }
    assert "target_receiver_transformer_bank" in {
        component.component_key for component in manifest.components
    }
    assert ledger.status == "ok"
    assert ledger.direct_policy_rl is False
    assert any(definition.uses_rl_style_signal for definition in ledger.definitions)
    assert not any(definition.direct_policy_rl for definition in ledger.definitions)


def test_phase63_trainer_scaffold_emits_cpu_smoke_without_training(tmp_path) -> None:
    scaffold_dir, _, _ = _materialize_phase6_scaffold(tmp_path)
    run_build_phase6_transport_neural_manifest(
        output_dir=tmp_path / "neural",
        scaffold_dir=scaffold_dir,
        run_dependencies_if_missing=False,
    )

    payload = run_train_wm_transport_bridge_v0_scaffold(
        output_dir=tmp_path / "trainer",
        neural_dir=tmp_path / "neural",
        scaffold_dir=scaffold_dir,
        run_dependencies_if_missing=False,
    )

    assert payload["authority_class"] == "transport_trainer_scaffold_only"
    assert payload["dataset_contract_ready"] is True
    assert payload["losses_defined"] is True
    assert payload["cpu_smoke_forward_passed"] is True
    assert payload["ready_for_training"] is False
    assert payload["ready_for_gpu_training"] is False
    assert payload["training_executed"] is False
    assert payload["weights_written"] is False
    assert payload["provider_executed"] is False
    assert payload["hardware_executed"] is False
    assert payload["live_policy_control"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["promotion_eligible"] is False

    dataset = load_wm_transport_trainer_dataset_contract(
        payload["artifact_refs"]["dataset_contract_path"]
    )
    trainer = load_wm_transport_trainer_scaffold_manifest(
        payload["artifact_refs"]["trainer_manifest_path"]
    )
    assert dataset.row_count == 32
    assert dataset.feature_dim > 0
    assert dataset.target_dim > 0
    assert trainer.trainer_scaffold_id == payload["trainer_scaffold_id"]

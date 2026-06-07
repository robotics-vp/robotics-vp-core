from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.build_phase6_transport_neural_manifest import (
    run_build_phase6_transport_neural_manifest,
)
from scripts.economic_world_model.prepare_phase6_transport_scaffold import (
    run_prepare_phase6_transport_scaffold,
)
from scripts.economic_world_model.run_phase6_transport_advisory_runtime import (
    run_phase6_transport_advisory_runtime,
)
from scripts.economic_world_model.audit_phase6_transport_closure import (
    run_audit_phase6_transport_closure,
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
from src.world_model.economic_world_model.shadow_outcomes import (
    EconomicWMShadowOutcomeReceipt,
)
from src.world_model.transport import (
    DENIED_TRANSPORT_RUNTIME_AUTHORITIES,
    load_wm_transport_advisory_runtime_report,
    load_wm_transport_decomposed_eval_reports,
    load_wm_transport_invocations,
    load_wm_transport_phase6_closure_audit,
    load_wm_transport_proposals,
    load_wm_transport_receipts,
    load_wm_transport_unitree_event_spine_joins,
)
from src.runtime.event_spine import RuntimeEvent, event_spine_sidecar_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _maturity_row(tmp_path: Path, source_row_id: str, wm_key: str, version: str):
    state_path = tmp_path / f"{source_row_id}_{wm_key}.json"
    _write_json(state_path, {"version": version, "state_id": f"state_{wm_key}"})
    return EconomicWMLowerWMMaturityRow(
        maturity_row_id=f"maturity_{source_row_id}_{wm_key}",
        source_row_id=source_row_id,
        source_episode_id="episode_transport_64",
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


def _materialize_phase64_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = [
        _maturity_row(
            tmp_path,
            "source_transport_64",
            "perception_grounding",
            "perception_grounding_world_state_v1",
        ),
        _maturity_row(
            tmp_path,
            "source_transport_64",
            "sim_synth_physics",
            "sim_synth_physics_world_state_v1",
        ),
        _maturity_row(
            tmp_path,
            "source_transport_64",
            "embodiment_actuation",
            "embodiment_actuation_world_state_v1",
        ),
    ]
    sweep = EconomicWMLowerWMMaturitySweep(
        sweep_id="maturity_sweep_transport_64",
        phase5_manifest_id="phase5_transport_64",
        lower_wm_preflight_id="lower_transport_64",
        resource_manifest_id="resource_transport_64",
        maturity_row_count=3,
        structural_ready_count=3,
        production_ready_count=0,
        maturity_rows_path=str(tmp_path / "maturity_rows.jsonl"),
        status="ok",
        ready_for_phase6_contracts=True,
        ready_for_production=False,
    )
    phase5 = EconomicWMPhase5LocalPrepManifest(
        manifest_id="phase5_transport_64",
        corpus_id="corpus_transport_64",
        lower_wm_preflight_id="lower_transport_64",
        resource_ingestion_manifest_id="resource_transport_64",
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
    neural_dir = tmp_path / "phase6_neural"
    trainer_dir = tmp_path / "phase6_trainer"
    run_prepare_phase6_transport_scaffold(
        output_dir=scaffold_dir,
        phase5_prep_path=phase5_path,
        maturity_sweep_path=sweep_path,
        maturity_rows_path=sweep.maturity_rows_path,
        run_dependencies_if_missing=False,
    )
    run_build_phase6_transport_neural_manifest(
        output_dir=neural_dir,
        scaffold_dir=scaffold_dir,
        run_dependencies_if_missing=False,
    )
    run_train_wm_transport_bridge_v0_scaffold(
        output_dir=trainer_dir,
        neural_dir=neural_dir,
        scaffold_dir=scaffold_dir,
        run_dependencies_if_missing=False,
    )
    return scaffold_dir, neural_dir, trainer_dir


def _shadow_outcome_path(tmp_path: Path) -> Path:
    receipt = EconomicWMShadowOutcomeReceipt(
        receipt_id="shadow_outcome_transport_64",
        work_order_id="shadow_work_order_transport_64",
        allocation_label="close_shadow_gap_replay",
        recommended_action="request_lower_wm_gap_closure_receipts",
        observed_effects={
            "local_structural_loop_closed": 1.0,
            "supervision_record_coverage": 1.0,
            "value_target_pack_coverage": 1.0,
            "value_ledger_receipt_coverage": 1.0,
            "provider_outcome_coverage": 0.0,
            "hardware_outcome_coverage": 0.0,
        },
        expected_effects={"expected_value": 0.6},
        comparison_metrics={
            "counterfactual_accuracy_observed": 0.0,
            "pareto_quality_observed": 0.0,
            "promotion_grade_evidence_observed": 0.0,
        },
        evidence_refs={"work_order_id": "shadow_work_order_transport_64"},
        blockers=["promotion_grade_shadow_benchmarks_missing"],
    )
    path = tmp_path / "shadow_outcomes.jsonl"
    _write_jsonl(path, [receipt.to_dict()])
    return path


def _unitree_event_spine_path(tmp_path: Path) -> Path:
    events = [
        RuntimeEvent.from_components(
            run_id="unitree_phase64_fixture",
            episode_id="unitree_phase64_episode_000",
            timestamp="2026-06-07T00:00:00+00:00",
            event_kind="unitree_trace_import_bundle_recorded",
            sequence_idx=0,
            scope={"robot_family": "unitree_g1", "posture": "bipedal_whole_body"},
            runtime_packet_id=None,
            contract_id="unitree_phase64_fixture_contract",
            receipt_label_refs=["trace_replay_receipt_001"],
            artifact_refs={"replay_ref": "unitree_replay_steps_v1.jsonl"},
            provenance={"source": "phase64_test_fixture"},
            metadata={
                "hardware_executed": False,
                "provider_executed": False,
                "promotion_eligible": False,
            },
        ),
        RuntimeEvent.from_components(
            run_id="unitree_phase64_fixture",
            episode_id="unitree_phase64_episode_000",
            timestamp="2026-06-07T00:00:01+00:00",
            event_kind="unitree_runtime_blocker_probe_recorded",
            sequence_idx=1,
            scope={"robot_family": "unitree_g1", "posture": "bipedal_whole_body"},
            runtime_packet_id=None,
            contract_id="unitree_phase64_fixture_contract",
            receipt_label_refs=["blocker_probe_receipt_001"],
            artifact_refs={"blocker_ref": "phase4_unitree_blocker_probe.json"},
            provenance={"source": "phase64_test_fixture"},
            metadata={
                "hardware_executed": False,
                "provider_executed": False,
                "promotion_eligible": False,
            },
        ),
    ]
    path = tmp_path / "unitree_event_spine.json"
    _write_json(
        path,
        event_spine_sidecar_payload(
            run_id="unitree_phase64_fixture",
            events=events,
        ),
    )
    return path


def test_phase64_runtime_emits_advisory_surfaces_and_decomposed_eval(tmp_path):
    scaffold_dir, neural_dir, trainer_dir = _materialize_phase64_inputs(tmp_path)
    shadow_path = _shadow_outcome_path(tmp_path)

    payload = run_phase6_transport_advisory_runtime(
        output_dir=tmp_path / "phase64_runtime",
        scaffold_dir=scaffold_dir,
        neural_dir=neural_dir,
        trainer_dir=trainer_dir,
        shadow_outcomes_path=shadow_path,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["proposal_count"] == 4
    assert payload["invocation_count"] == 4
    assert payload["receipt_count"] == 4
    assert payload["eval_report_count"] == 4
    assert payload["joined_shadow_outcome_count"] == 2
    assert payload["ready_for_decomposed_eval"] is True
    assert payload["ready_for_training"] is False
    assert payload["training_executed"] is False
    assert payload["weights_written"] is False
    assert payload["live_policy_control"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["promotion_eligible"] is False

    refs = payload["artifact_refs"]
    report = load_wm_transport_advisory_runtime_report(refs["report_path"])
    proposals = load_wm_transport_proposals(refs["proposals_path"])
    invocations = load_wm_transport_invocations(refs["invocations_path"])
    receipts = load_wm_transport_receipts(refs["receipts_path"])
    evals = load_wm_transport_decomposed_eval_reports(refs["eval_reports_path"])

    assert report.report_id == payload["report_id"]
    assert all(proposal.advisory_only for proposal in proposals)
    assert all(
        set(DENIED_TRANSPORT_RUNTIME_AUTHORITIES).issubset(
            set(proposal.denied_authority)
        )
        for proposal in proposals
    )
    assert not any(invocation.target_receiver_bypassed for invocation in invocations)
    assert not any(receipt.training_executed for receipt in receipts)
    assert not any(receipt.weights_written for receipt in receipts)
    assert not any(receipt.provider_executed for receipt in receipts)
    assert not any(receipt.hardware_executed for receipt in receipts)
    assert not any(receipt.live_policy_control for receipt in receipts)
    assert not any(receipt.reward_math_mutation for receipt in receipts)
    assert not any(receipt.promotion_eligible for receipt in receipts)
    assert all(receipt.receiver_actionable for receipt in receipts)
    assert all(eval_report.bridge_only_score > 0.0 for eval_report in evals)
    assert all(eval_report.receiver_only_score > 0.0 for eval_report in evals)
    assert all("interaction_effect" in eval_report.terms for eval_report in evals)
    assert any(
        eval_report.downstream_only_score > 0.0
        and eval_report.shadow_outcome_join_status
        == "joined_local_structural_shadow_outcome"
        for eval_report in evals
    )


def test_phase64_runtime_keeps_shadow_join_slots_open_when_outcomes_missing(tmp_path):
    scaffold_dir, neural_dir, trainer_dir = _materialize_phase64_inputs(tmp_path)

    payload = run_phase6_transport_advisory_runtime(
        output_dir=tmp_path / "phase64_runtime_missing_shadow",
        scaffold_dir=scaffold_dir,
        neural_dir=neural_dir,
        trainer_dir=trainer_dir,
        shadow_outcomes_path=tmp_path / "missing_shadow_outcomes.jsonl",
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["joined_shadow_outcome_count"] == 0
    receipts = load_wm_transport_receipts(payload["artifact_refs"]["receipts_path"])
    economic_slots = [
        receipt.shadow_outcome_slot
        for receipt in receipts
        if receipt.target_wm == "economic"
    ]
    assert len(economic_slots) == 2
    assert all(
        slot.join_status == "awaiting_shadow_outcome_receipt"
        for slot in economic_slots
    )
    assert not any(slot.promotion_eligible for slot in economic_slots)


def test_phase64_runtime_threads_unitree_event_spine_refs_into_eval(tmp_path):
    scaffold_dir, neural_dir, trainer_dir = _materialize_phase64_inputs(tmp_path)
    event_spine_path = _unitree_event_spine_path(tmp_path)

    payload = run_phase6_transport_advisory_runtime(
        output_dir=tmp_path / "phase64_runtime_unitree_events",
        scaffold_dir=scaffold_dir,
        neural_dir=neural_dir,
        trainer_dir=trainer_dir,
        shadow_outcomes_path=_shadow_outcome_path(tmp_path),
        unitree_event_spine_path=event_spine_path,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["aggregate_counts"]["unitree_event_spine_join_count"] == 4.0
    assert payload["aggregate_counts"]["joined_unitree_event_spine_count"] == 4.0
    assert payload["aggregate_counts"]["unitree_event_count"] == 2.0
    assert payload["metadata"]["unitree_event_spine_joined"] is True

    refs = payload["artifact_refs"]
    joins = load_wm_transport_unitree_event_spine_joins(
        refs["unitree_event_spine_joins_path"]
    )
    proposals = load_wm_transport_proposals(refs["proposals_path"])
    receipts = load_wm_transport_receipts(refs["receipts_path"])
    evals = load_wm_transport_decomposed_eval_reports(refs["eval_reports_path"])

    assert len(joins) == 4
    assert all(join.join_status == "joined_unitree_event_spine_ref" for join in joins)
    assert all(join.event_spine_ref == str(event_spine_path) for join in joins)
    assert not any(join.provider_executed for join in joins)
    assert not any(join.hardware_executed for join in joins)
    assert not any(join.live_policy_control for join in joins)
    assert not any(join.reward_math_mutation for join in joins)
    assert not any(join.promotion_eligible for join in joins)
    assert all(
        proposal.metadata["unitree_event_spine_join_status"]
        == "joined_unitree_event_spine_ref"
        for proposal in proposals
    )
    assert all(
        receipt.metadata["unitree_event_spine_ref"] == str(event_spine_path)
        for receipt in receipts
    )
    assert all(
        eval_report.metadata["unitree_event_lower_wm_label_only"] is True
        for eval_report in evals
    )


def test_phase6_closure_audit_confirms_only_evidence_blockers_remain(tmp_path):
    scaffold_dir, neural_dir, trainer_dir = _materialize_phase64_inputs(tmp_path)
    runtime_dir = tmp_path / "phase64_runtime"
    run_phase6_transport_advisory_runtime(
        output_dir=runtime_dir,
        scaffold_dir=scaffold_dir,
        neural_dir=neural_dir,
        trainer_dir=trainer_dir,
        shadow_outcomes_path=_shadow_outcome_path(tmp_path),
        run_dependencies_if_missing=False,
    )

    payload = run_audit_phase6_transport_closure(
        output_dir=tmp_path / "phase6_closure",
        scaffold_dir=scaffold_dir,
        neural_dir=neural_dir,
        trainer_dir=trainer_dir,
        runtime_dir=runtime_dir,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["local_phase6_structurally_closed"] is True
    assert payload["missing_local_runtime_contracts"] == []
    assert set(payload["remaining_evidence_blockers"]) == {
        "cross_wm_corpus_density_not_proven",
        "gpu_bridge_receiver_training_not_run",
        "topology_latency_benchmarks_not_run",
        "provider_or_hardware_transport_evidence_missing",
        "promotion_grade_downstream_benchmark_missing",
    }
    assert payload["contract_count"] == 4
    assert payload["advisory_proposal_count"] == 4
    assert payload["decomposed_eval_report_count"] == 4
    assert payload["ready_for_training"] is False
    assert payload["training_executed"] is False
    assert payload["weights_written"] is False
    assert payload["provider_executed"] is False
    assert payload["hardware_executed"] is False
    assert payload["live_policy_control"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["promotion_eligible"] is False

    report = load_wm_transport_phase6_closure_audit(
        payload["artifact_refs"]["report_path"]
    )
    assert report.audit_id == payload["audit_id"]
    assert "advisory_runtime_proposals_invocations_receipts" in (
        report.closed_local_surfaces
    )

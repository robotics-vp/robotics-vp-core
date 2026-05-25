from __future__ import annotations

from pathlib import Path

from scripts.economic_world_model.evaluate_phase7_meta_governance_shadow import (
    run_evaluate_phase7_meta_governance_shadow,
)
from scripts.economic_world_model.prepare_phase7_meta_regal_control_scaffold import (
    run_prepare_phase7_meta_regal_control_scaffold,
)
from scripts.economic_world_model.wire_phase7_meta_regal_runtime_shadow import (
    run_wire_phase7_meta_regal_runtime_shadow,
)
from src.world_model.humanoid_readiness import (
    Phase35465LocalClosureAudit,
    Phase65MetaNodeNeuralizationReport,
    load_phase7_conflict_join_eval_reports,
    load_phase7_control_field_eval_reports,
    load_phase7_meta_governance_evaluation_report,
    load_phase7_outcome_join_rows,
    load_phase7_pareto_regime_eval_reports,
    save_phase35465_local_closure_audit,
)
from src.world_model.humanoid_readiness.common import write_json


def _fake_phase65_report(phase65_dir: Path) -> Path:
    report = Phase65MetaNodeNeuralizationReport(
        report_id="test_phase65_meta_node_report",
        phase35_report_id="test_phase35",
        phase4_report_id="test_phase4",
        phase6_closure_audit_id="test_phase6_closure",
        status="ok",
        node_state_count=5,
        trajectory_receipt_count=5,
        intervention_receipt_count=5,
        counterfactual_target_count=5,
        robustness_report_count=5,
        promotion_gate_count=5,
        local_meta_node_scaffold_complete=True,
        ready_for_phase7_scaffold=True,
        remaining_blockers=["counterfactual_meta_node_corpus_density_missing"],
    )
    path = phase65_dir / "phase65_meta_node_neuralization_report_v1.json"
    write_json(path, report.to_dict())
    return path


def _fake_closure_audit(closure_dir: Path) -> Path:
    audit = Phase35465LocalClosureAudit(
        audit_id="test_phase35_4_65_closure",
        phase35_report_id="test_phase35",
        phase35_bipedal_readiness_audit_id="test_phase35_bipedal",
        phase4_report_id="test_phase4",
        phase4_downstream_controller_report_id="test_phase4_downstream",
        phase4_unitree_bringup_readiness_report_id="test_phase4_bringup",
        phase4_unitree_local_harness_report_id="test_phase4_harness",
        phase4_unitree_runtime_bridge_report_id="test_phase4_runtime_bridge",
        phase4_unitree_blocker_stress_probe_report_id="test_phase4_blockers",
        phase65_report_id="test_phase65_meta_node_report",
        status="ok",
        local_phase35_complete=True,
        local_phase35_bipedal_readiness_complete=True,
        local_phase4_complete=True,
        local_phase4_downstream_controller_complete=True,
        local_phase4_unitree_bringup_readiness_complete=True,
        local_phase4_unitree_local_harness_complete=True,
        local_phase4_unitree_runtime_bridge_complete=True,
        local_phase4_unitree_blocker_stress_probe_complete=True,
        local_phase65_complete=True,
        all_local_structures_complete=True,
        ready_for_phase7_scaffold=True,
        closed_local_surfaces=["phase65_meta_node_state"],
        remaining_evidence_blockers=[
            "gpu_training_provider_hardware_evidence_missing"
        ],
    )
    path = closure_dir / "phase35_4_65_local_closure_audit_v1.json"
    save_phase35465_local_closure_audit(path, audit)
    return path


def test_phase7_meta_governance_eval_and_outcome_join_slots(tmp_path):
    phase65_dir = tmp_path / "phase65"
    closure_dir = tmp_path / "closure"
    scaffold_dir = tmp_path / "phase7_scaffold"
    runtime_dir = tmp_path / "phase7_shadow_runtime"
    eval_dir = tmp_path / "phase7_eval"
    _fake_phase65_report(phase65_dir)
    _fake_closure_audit(closure_dir)
    run_prepare_phase7_meta_regal_control_scaffold(
        output_dir=scaffold_dir,
        phase65_dir=phase65_dir,
        closure_dir=closure_dir,
        run_dependencies_if_missing=False,
    )
    run_wire_phase7_meta_regal_runtime_shadow(
        output_dir=runtime_dir,
        phase7_scaffold_dir=scaffold_dir,
        episodes=1,
        timestamp_base="2026-05-25T00:00:00+00:00",
        run_id="phase7_eval_test",
        run_dependencies_if_missing=False,
    )

    payload = run_evaluate_phase7_meta_governance_shadow(
        output_dir=eval_dir,
        phase7_runtime_dir=runtime_dir,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["control_field_eval_count"] == 7
    assert payload["conflict_join_eval_count"] == 6
    assert payload["pareto_regime_eval_count"] == 1
    assert payload["outcome_join_row_count"] == 14
    assert payload["phase7_event_count"] == 13
    assert payload["phase7_decision_count"] == 13
    assert payload["control_field_only_eval_complete"] is True
    assert payload["conflict_join_eval_complete"] is True
    assert payload["pareto_regime_eval_complete"] is True
    assert payload["outcome_join_slots_complete"] is True
    assert payload["local_meta_governance_eval_complete"] is True
    assert payload["replay_export_ready"] is True
    assert payload["phase7_authority_granted"] is False
    assert payload["live_dispatch_allowed"] is False
    assert payload["hard_veto_dispatch"] is False
    assert payload["training_executed"] is False
    assert payload["weights_written"] is False
    assert payload["provider_executed"] is False
    assert payload["hardware_executed"] is False
    assert payload["unitree_sim_runtime_executed"] is False
    assert payload["live_policy_control"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["promotion_eligible"] is False
    assert not any(payload["denied_gates"].values())

    report = load_phase7_meta_governance_evaluation_report(
        eval_dir / "phase7_meta_governance_evaluation_report_v1.json"
    )
    field_evals = load_phase7_control_field_eval_reports(
        eval_dir / "phase7_control_field_eval_reports_v1.jsonl"
    )
    conflict_evals = load_phase7_conflict_join_eval_reports(
        eval_dir / "phase7_conflict_join_eval_reports_v1.jsonl"
    )
    regime_evals = load_phase7_pareto_regime_eval_reports(
        eval_dir / "phase7_pareto_regime_eval_reports_v1.jsonl"
    )
    rows = load_phase7_outcome_join_rows(
        eval_dir / "phase7_outcome_join_rows_v1.jsonl"
    )

    assert report.local_meta_governance_eval_complete is True
    assert all(item.eval_status == "ok" for item in field_evals)
    assert all(item.metrics["live_dispatch_denied"] == 1.0 for item in field_evals)
    assert all(item.metrics["reward_mutation_denied"] == 1.0 for item in field_evals)
    assert all(item.eval_status == "ok" for item in conflict_evals)
    assert all(
        item.metrics["hard_veto_dispatch_denied"] == 1.0
        for item in conflict_evals
    )
    assert regime_evals[0].regime_key in {
        "nominal_bipedal_with_shadow_conflicts",
        "operator_or_deployment_review_shadow",
        "deployment_truth_or_safety_blocked_shadow",
    }
    assert "veto_constraint" in regime_evals[0].composition_modes_seen
    assert all(row.replay_export_ready for row in rows)
    assert all(row.training_target_only for row in rows)
    assert all(not row.weights_written for row in rows)
    assert all(not row.promotion_eligible for row in rows)
    assert {
        "control_field_shadow_outcome_join",
        "conflict_join_shadow_outcome_join",
        "pareto_regime_shadow_outcome_join",
    }.issubset({row.row_family for row in rows})

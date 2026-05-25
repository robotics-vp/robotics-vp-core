from __future__ import annotations

from pathlib import Path

from scripts.economic_world_model.prepare_phase7_meta_regal_control_scaffold import (
    run_prepare_phase7_meta_regal_control_scaffold,
)
from src.world_model.humanoid_readiness import (
    DENIED_PHASE7_AUTHORITIES,
    Phase35465LocalClosureAudit,
    Phase65MetaNodeNeuralizationReport,
    load_phase7_admissible_region_specs,
    load_phase7_composition_mode_specs,
    load_phase7_conflict_override_receipts,
    load_phase7_control_field_slots,
    load_phase7_governance_node_surfaces,
    load_phase7_meta_regal_control_scaffold_report,
    load_phase7_promotion_gates,
    load_phase7_training_row_slots,
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
        closed_local_surfaces=[
            "phase35_bipedal_readiness",
            "phase4_unitree_harnesses",
            "phase65_meta_node_state",
        ],
        remaining_evidence_blockers=[
            "gpu_training_provider_hardware_evidence_missing"
        ],
    )
    path = closure_dir / "phase35_4_65_local_closure_audit_v1.json"
    save_phase35465_local_closure_audit(path, audit)
    return path


def test_phase7_meta_regal_control_scaffold_outputs_and_denied_gates(tmp_path):
    phase65_dir = tmp_path / "phase65"
    closure_dir = tmp_path / "closure"
    output_dir = tmp_path / "phase7"
    _fake_phase65_report(phase65_dir)
    _fake_closure_audit(closure_dir)

    payload = run_prepare_phase7_meta_regal_control_scaffold(
        output_dir=output_dir,
        phase65_dir=phase65_dir,
        closure_dir=closure_dir,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["local_phase7_scaffold_complete"] is True
    assert payload["ready_for_runtime_wiring"] is True
    assert payload["runtime_wiring_executed"] is False
    assert payload["phase7_authority_granted"] is False
    assert payload["live_control_authority"] is False
    assert payload["training_executed"] is False
    assert payload["weights_written"] is False
    assert payload["provider_executed"] is False
    assert payload["hardware_executed"] is False
    assert payload["unitree_sim_runtime_executed"] is False
    assert payload["live_policy_control"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["promotion_eligible"] is False
    assert not any(payload["denied_gates"].values())

    assert payload["governance_node_surface_count"] >= 7
    assert payload["composition_mode_count"] == 5
    assert payload["conflict_override_receipt_count"] >= 5
    assert payload["admissible_region_count"] >= 5
    assert payload["control_field_slot_count"] >= 6
    assert payload["training_row_slot_count"] >= 5
    assert payload["promotion_gate_count"] >= 5

    report = load_phase7_meta_regal_control_scaffold_report(
        output_dir / "phase7_meta_regal_control_scaffold_report_v1.json"
    )
    surfaces = load_phase7_governance_node_surfaces(
        output_dir / "phase7_governance_node_surfaces_v1.jsonl"
    )
    modes = load_phase7_composition_mode_specs(
        output_dir / "phase7_composition_mode_specs_v1.jsonl"
    )
    conflicts = load_phase7_conflict_override_receipts(
        output_dir / "phase7_conflict_override_receipts_v1.jsonl"
    )
    regions = load_phase7_admissible_region_specs(
        output_dir / "phase7_admissible_region_specs_v1.jsonl"
    )
    fields = load_phase7_control_field_slots(
        output_dir / "phase7_control_field_slots_v1.jsonl"
    )
    rows = load_phase7_training_row_slots(
        output_dir / "phase7_training_row_slots_v1.jsonl"
    )
    gates = load_phase7_promotion_gates(output_dir / "phase7_promotion_gates_v1.jsonl")

    assert report.local_phase7_scaffold_complete is True
    assert report.ready_for_runtime_wiring is True
    assert report.phase7_authority_granted is False

    node_keys = {surface.node_key for surface in surfaces}
    assert "economic_allocation_governance" in node_keys
    assert "safety_constraint_governance" in node_keys
    assert "deployment_truth_governance" in node_keys
    assert "embodiment_limit_governance" in node_keys
    assert all(surface.advisory_only for surface in surfaces)
    assert all(surface.training_aware for surface in surfaces)
    assert all(not surface.bounded_helper_ready for surface in surfaces)
    assert all(
        set(DENIED_PHASE7_AUTHORITIES).issubset(surface.denied_authority)
        for surface in surfaces
    )

    mode_keys = {mode.mode_key for mode in modes}
    assert mode_keys == {
        "pareto_relation",
        "lexicographic_priority",
        "veto_constraint",
        "advisory_evidence",
        "confidence_weighted",
    }
    assert any(mode.hard_constraint_semantics for mode in modes)
    assert all(mode.shadow_only for mode in modes)

    assert any(
        conflict.conflict_key == "safety_vs_economic_throughput"
        and conflict.composition_mode == "veto_constraint"
        for conflict in conflicts
    )
    assert all(conflict.shadow_only for conflict in conflicts)

    region_keys = {region.regime_key for region in regions}
    assert "nominal_bipedal_shadow" in region_keys
    assert "degraded_stable_base_fallback" in region_keys
    assert "deployment_truth_blocked" in region_keys
    assert all(region.evaluation_only for region in regions)
    assert all(not region.promotion_eligible for region in regions)

    assert all(field.shadow_only for field in fields)
    assert all(not field.live_dispatch_allowed for field in fields)
    assert all(not field.reward_math_mutation for field in fields)
    assert all(not field.promotion_eligible for field in fields)
    posture_field = next(
        field for field in fields if field.field_key == "embodiment_mode_demote_field"
    )
    assert posture_field.field_schema["primary_posture"] == "bipedal_whole_body"
    assert (
        posture_field.field_schema["fallback_posture"]
        == "stable_base_mobile_manipulator"
    )
    assert posture_field.field_schema["fixed_base_tabletop"] == (
        "curriculum_regression_only"
    )

    assert all(row.replay_export_ready for row in rows)
    assert all(row.training_target_only for row in rows)
    assert all(not row.weights_written for row in rows)
    assert all(not row.promotion_eligible for row in rows)
    assert all(gate.gate_status == "denied" for gate in gates)
    assert all(not gate.authority_granted for gate in gates)
    assert all(not gate.promotion_eligible for gate in gates)

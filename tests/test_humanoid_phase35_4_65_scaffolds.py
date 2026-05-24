from __future__ import annotations

from pathlib import Path

from scripts.economic_world_model.audit_phase35_4_65_local_closure import (
    run_audit_phase35_4_65_local_closure,
)
from scripts.economic_world_model.prepare_phase35_humanoid_capacity_env_refit import (
    run_prepare_phase35_humanoid_capacity_env_refit,
)
from scripts.economic_world_model.audit_phase35_bipedal_readiness import (
    run_audit_phase35_bipedal_readiness,
)
from scripts.economic_world_model.prepare_phase4_deployment_enabler_sweep import (
    run_prepare_phase4_deployment_enabler_sweep,
)
from scripts.economic_world_model.prepare_phase4_downstream_controller_scaffold import (
    run_prepare_phase4_downstream_controller_scaffold,
)
from scripts.economic_world_model.prepare_phase65_meta_node_neuralization import (
    run_prepare_phase65_meta_node_neuralization,
)
from src.world_model.humanoid_readiness import (
    DENIED_LOCAL_AUTHORITIES,
    load_meta_node_promotion_gates,
    load_meta_node_robustness_reports,
    load_meta_node_states,
    load_phase35_capacity_bands,
    load_phase35_env_taxonomy_receipts,
    load_phase35_schema_deltas,
    load_phase4_contract_surfaces,
    load_phase4_stub_surfaces,
)
from src.world_model.transport import (
    WMTransportPhase6ClosureAuditReport,
    save_wm_transport_phase6_closure_audit,
)


def _fake_phase6_closure(phase6_dir: Path) -> Path:
    path = phase6_dir / "wm_transport_phase6_closure_audit_v1.json"
    report = WMTransportPhase6ClosureAuditReport(
        audit_id="test_phase6_closure",
        scaffold_report_id="scaffold_test",
        neural_manifest_id="neural_test",
        loss_ledger_id="loss_test",
        trainer_scaffold_id="trainer_test",
        advisory_runtime_report_id="runtime_test",
        status="ok",
        local_phase6_structurally_closed=True,
        missing_local_runtime_contracts=[],
        remaining_evidence_blockers=[
            "cross_wm_corpus_density_not_proven",
            "gpu_bridge_receiver_training_not_run",
        ],
        closed_local_surfaces=["phase6_transport_local_surfaces"],
        contract_count=4,
        transformer_count=3,
        training_row_count=16,
        roundtrip_receipt_count=4,
        neural_component_count=8,
        loss_count=14,
        advisory_proposal_count=4,
        advisory_receipt_count=4,
        decomposed_eval_report_count=4,
        joined_shadow_outcome_count=2,
    )
    save_wm_transport_phase6_closure_audit(path, report)
    return path


def test_phase35_phase4_phase65_local_scaffolds_and_gates(tmp_path):
    phase35_dir = tmp_path / "phase35"
    bipedal_chassis_dir = tmp_path / "bipedal_chassis"
    phase35_bipedal_readiness_dir = tmp_path / "phase35_bipedal_readiness"
    phase4_dir = tmp_path / "phase4"
    phase4_downstream_controller_dir = tmp_path / "phase4_downstream_controller"
    phase65_dir = tmp_path / "phase65"
    phase6_dir = tmp_path / "phase6_closure"
    closure_dir = tmp_path / "closure"
    _fake_phase6_closure(phase6_dir)

    phase35 = run_prepare_phase35_humanoid_capacity_env_refit(
        output_dir=phase35_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
    )
    assert phase35["status"] == "ok"
    assert phase35["local_structural_refit_complete"] is True
    assert phase35["capacity_band_count"] == 5
    assert phase35["schema_delta_count"] >= 10
    assert phase35["ready_for_training"] is False
    assert phase35["unitree_sim_runtime_executed"] is False
    assert phase35["promotion_eligible"] is False
    assert not any(phase35["denied_gates"][key] for key in DENIED_LOCAL_AUTHORITIES)

    capacity_bands = load_phase35_capacity_bands(
        phase35_dir / "humanoid_phase35_capacity_band_contracts_v1.jsonl"
    )
    schema_deltas = load_phase35_schema_deltas(
        phase35_dir / "humanoid_phase35_schema_delta_contracts_v1.jsonl"
    )
    env_taxonomy = load_phase35_env_taxonomy_receipts(
        phase35_dir / "humanoid_phase35_env_taxonomy_receipts_v1.jsonl"
    )
    assert {band.band_name for band in capacity_bands} >= {
        "onboard_reflex_reserve",
        "companion_realtime_assist",
        "offline_gpu_training",
    }
    assert any(delta.surface_name == "whole_body_proprioception" for delta in schema_deltas)
    assert any(
        delta.surface_name == "stable_base_fallback_action"
        and delta.posture_scope == "stable_base_mobile_manipulator"
        for delta in schema_deltas
    )
    assert {receipt.posture_tag for receipt in env_taxonomy} == {
        "bipedal_whole_body",
        "stable_base_mobile_manipulator",
        "fixed_base_tabletop",
    }

    phase35_bipedal_readiness = run_audit_phase35_bipedal_readiness(
        output_dir=phase35_bipedal_readiness_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        run_dependencies_if_missing=False,
    )
    assert phase35_bipedal_readiness["status"] == "ok"
    assert phase35_bipedal_readiness["phase35_no_gpu_no_hardware_prepared"] is True
    assert phase35_bipedal_readiness["whole_body_replay_row_count"] == 3
    assert phase35_bipedal_readiness["ready_for_training"] is False
    assert phase35_bipedal_readiness["promotion_eligible"] is False

    phase4 = run_prepare_phase4_deployment_enabler_sweep(
        output_dir=phase4_dir,
        phase35_dir=phase35_dir,
        run_dependencies_if_missing=False,
    )
    assert phase4["status"] == "ok"
    assert phase4["local_non_hardware_scaffold_complete"] is True
    assert phase4["contract_surface_count"] == 15
    assert phase4["stub_surface_count"] == 3
    assert phase4["phase_counts"]["4A"] == 5
    assert phase4["phase_counts"]["4E"] == 5
    assert phase4["phase_counts"]["4F"] == 5
    assert phase4["live_policy_control"] is False
    assert phase4["promotion_eligible"] is False

    contracts = load_phase4_contract_surfaces(
        phase4_dir / "humanoid_phase4_contract_surfaces_v1.jsonl"
    )
    stubs = load_phase4_stub_surfaces(
        phase4_dir / "humanoid_phase4_stub_surfaces_v1.jsonl"
    )
    assert all(contract.replay_export_posture == "sidecar_planning_only" for contract in contracts)
    assert all("promotion" in contract.denied_authority for contract in contracts)
    assert all(stub.explicit_stub and stub.planning_only for stub in stubs)
    assert {stub.phase_key for stub in stubs} == {"4B", "4C", "4D"}

    phase4_downstream = run_prepare_phase4_downstream_controller_scaffold(
        output_dir=phase4_downstream_controller_dir,
        phase4_dir=phase4_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=phase35_bipedal_readiness_dir,
        run_dependencies_if_missing=False,
    )
    assert phase4_downstream["status"] == "ok"
    assert phase4_downstream["local_downstream_controller_scaffold_complete"] is True
    assert phase4_downstream["unitree_bridge_contract_present"] is True
    assert phase4_downstream["g1pilot_fallback_contract_present"] is True
    assert phase4_downstream["dry_run_controller_present"] is True
    assert phase4_downstream["hardware_dispatch_enabled"] is False
    assert phase4_downstream["ros2_publish_attempted"] is False
    assert phase4_downstream["unitree_sdk2_write_enabled"] is False
    assert phase4_downstream["promotion_eligible"] is False

    phase65 = run_prepare_phase65_meta_node_neuralization(
        output_dir=phase65_dir,
        phase35_dir=phase35_dir,
        phase4_dir=phase4_dir,
        phase6_closure_dir=phase6_dir,
        run_dependencies_if_missing=False,
    )
    assert phase65["status"] == "ok"
    assert phase65["local_meta_node_scaffold_complete"] is True
    assert phase65["node_state_count"] == 5
    assert phase65["counterfactual_target_count"] == 5
    assert phase65["robustness_report_count"] == 5
    assert phase65["promotion_gate_count"] == 5
    assert phase65["ready_for_phase7_scaffold"] is True
    assert phase65["phase7_authority_granted"] is False
    assert phase65["training_executed"] is False
    assert phase65["weights_written"] is False
    assert phase65["promotion_eligible"] is False

    states = load_meta_node_states(phase65_dir / "meta_node_states_v1.jsonl")
    robustness = load_meta_node_robustness_reports(
        phase65_dir / "meta_node_robustness_reports_v1.jsonl"
    )
    gates = load_meta_node_promotion_gates(
        phase65_dir / "meta_node_promotion_gates_v1.jsonl"
    )
    assert {state.node_family for state in states} >= {
        "economic_allocation_guard",
        "transport_quality_guard",
        "humanoid_posture_guard",
        "deployment_resource_guard",
        "operator_recovery_guard",
    }
    assert all("phase7_control_wm_authority" in state.denied_authority for state in states)
    assert all(
        report.metrics["deployment_robustness_evidence"] == 0.0
        for report in robustness
    )
    assert all(gate.gate_status == "denied" for gate in gates)
    assert not any(gate.phase7_authority_granted for gate in gates)

    closure = run_audit_phase35_4_65_local_closure(
        output_dir=closure_dir,
        phase35_dir=phase35_dir,
        phase35_bipedal_readiness_dir=phase35_bipedal_readiness_dir,
        phase4_dir=phase4_dir,
        phase4_downstream_controller_dir=phase4_downstream_controller_dir,
        phase65_dir=phase65_dir,
        run_dependencies_if_missing=False,
    )
    assert closure["status"] == "ok"
    assert closure["local_phase35_complete"] is True
    assert closure["local_phase35_bipedal_readiness_complete"] is True
    assert closure["local_phase4_complete"] is True
    assert closure["local_phase4_downstream_controller_complete"] is True
    assert closure["local_phase65_complete"] is True
    assert closure["all_local_structures_complete"] is True
    assert closure["ready_for_phase7_scaffold"] is True
    assert closure["phase7_authority_granted"] is False
    assert closure["hardware_executed"] is False
    assert closure["reward_math_mutation"] is False
    assert closure["promotion_eligible"] is False
    assert "phase35_whole_body_replay_row_slots" in closure["closed_local_surfaces"]
    assert "phase4_dry_run_command_frames" in closure["closed_local_surfaces"]
    assert "phase65_denied_promotion_gates" in closure["closed_local_surfaces"]

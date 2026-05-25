from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.adapt_phase7_governance_node_signals import (
    run_adapt_phase7_governance_node_signals,
)
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
    EXPECTED_PHASE7_GOVERNANCE_NODE_KEYS,
    Phase35465LocalClosureAudit,
    Phase65MetaNodeNeuralizationReport,
    load_phase7_governance_node_signal_receipts,
    save_phase35465_local_closure_audit,
)
from src.world_model.humanoid_readiness.common import write_json, write_jsonl


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


def _write_lower_wm_signal_fixture(root: Path) -> None:
    write_json(
        root
        / "phase35_humanoid_capacity_env_refit"
        / "humanoid_phase35_refit_report_v1.json",
        {
            "report_id": "phase35_refit_test",
            "status": "ok",
            "bipedal_chassis_joint_count": 29,
            "bipedal_chassis_joint_limit_envelope_count": 29,
            "hardware_executed": False,
            "training_executed": False,
            "weights_written": False,
        },
    )
    write_json(
        root
        / "phase35_bipedal_readiness_audit"
        / "phase35_bipedal_readiness_audit_v1.json",
        {
            "audit_id": "phase35_bipedal_audit_test",
            "status": "ok",
            "balance_geometry_report_count": 3,
            "joint_vector_validation_receipt_count": 2,
            "whole_body_replay_row_count": 3,
            "hardware_executed": False,
        },
    )
    write_json(
        root
        / "phase4_downstream_controller_scaffold"
        / "phase4_downstream_controller_scaffold_report_v1.json",
        {
            "report_id": "phase4_downstream_test",
            "status": "ok",
            "safety_receipt_count": 6,
            "command_frame_count": 6,
            "hardware_executed": False,
            "live_policy_control": False,
        },
    )
    write_json(
        root
        / "phase4_unitree_bringup_readiness"
        / "phase4_unitree_bringup_readiness_report_v1.json",
        {
            "report_id": "phase4_bringup_test",
            "status": "ok",
            "dependency_verified_count": 8,
            "operator_recovery_runbook_count": 4,
            "hardware_executed": False,
            "live_policy_control": False,
        },
    )
    write_json(
        root
        / "phase4_unitree_local_harnesses"
        / "phase4_unitree_local_harness_report_v1.json",
        {
            "report_id": "phase4_local_harness_test",
            "status": "ok",
            "stale_validation_receipt_count": 4,
            "watchdog_demotion_receipt_count": 1,
            "hardware_executed": False,
        },
    )
    write_json(
        root
        / "phase4_unitree_runtime_evidence_bridge"
        / "phase4_unitree_runtime_evidence_bridge_report_v1.json",
        {
            "report_id": "phase4_runtime_bridge_test",
            "status": "ok",
            "ros2_runtime_readiness_receipt_count": 2,
            "safety_envelope_expansion_receipt_count": 5,
            "operator_recovery_drill_receipt_count": 4,
            "operator_recovery_scenario_count": 4,
            "hardware_executed": False,
            "live_stream_observed": False,
            "live_policy_control": False,
        },
    )
    write_json(
        root
        / "phase4_unitree_blocker_stress_probes"
        / "phase4_unitree_blocker_stress_probe_report_v1.json",
        {
            "report_id": "phase4_blocker_probe_test",
            "status": "ok",
            "succeeded_probe_count": 8,
            "mujoco_model_stress_success_count": 5,
            "hardware_executed": False,
        },
    )
    write_json(
        root
        / "phase6_transport_advisory_runtime"
        / "wm_transport_advisory_runtime_report_v1.json",
        {
            "report_id": "phase6_advisory_test",
            "status": "ok",
            "eval_report_count": 20,
            "joined_shadow_outcome_count": 10,
            "shadow_join_slot_count": 20,
            "training_executed": False,
            "weights_written": False,
            "reward_math_mutation": False,
            "promotion_eligible": False,
        },
    )
    write_json(
        root
        / "phase6_transport_closure_audit"
        / "wm_transport_phase6_closure_audit_v1.json",
        {
            "audit_id": "phase6_closure_test",
            "status": "ok",
            "training_executed": False,
            "weights_written": False,
            "hardware_executed": False,
        },
    )
    write_json(
        root
        / "phase65_meta_node_neuralization"
        / "phase65_meta_node_neuralization_report_v1.json",
        {
            "report_id": "phase65_test",
            "status": "ok",
            "node_state_count": 5,
            "training_executed": False,
            "weights_written": False,
        },
    )
    write_json(
        root / "phase7_meta_regal_shadow_runtime" / "summary.json",
        {
            "run_id": "phase7_signal_fixture",
            "mean_net_customer_rate": 20.0,
            "total_data_share_credit": 6.0,
            "mean_reward_total": 1.6,
        },
    )
    write_json(
        root
        / "phase7_meta_governance_eval"
        / "phase7_meta_governance_evaluation_report_v1.json",
        {
            "report_id": "phase7_eval_test",
            "status": "ok",
            "control_field_eval_count": 14,
            "reward_math_mutation": False,
            "promotion_eligible": False,
        },
    )
    jsonl_paths = [
        "phase35_bipedal_readiness_audit/balance_geometry_reports_v1.jsonl",
        "phase35_bipedal_readiness_audit/joint_vector_validation_receipts_v1.jsonl",
        "phase35_bipedal_readiness_audit/whole_body_replay_rows_v1.jsonl",
        "phase4_downstream_controller_scaffold/controller_safety_receipts_v1.jsonl",
        "phase4_downstream_controller_scaffold/low_level_command_frames_v1.jsonl",
        "phase4_unitree_bringup_readiness/unitree_safety_preflight_receipts_v1.jsonl",
        "phase4_unitree_bringup_readiness/unitree_operator_recovery_runbooks_v1.jsonl",
        "phase4_unitree_local_harnesses/unitree_mock_receiver_receipts_v1.jsonl",
        "phase4_unitree_local_harnesses/unitree_stale_data_validation_receipts_v1.jsonl",
        "phase4_unitree_local_harnesses/unitree_watchdog_demotion_receipts_v1.jsonl",
        "phase4_unitree_local_harnesses/unitree_safety_state_transitions_v1.jsonl",
        "phase4_unitree_local_harnesses/unitree_trace_replay_receipts_v1.jsonl",
        "phase4_unitree_runtime_evidence_bridge/unitree_ros2_runtime_readiness_receipts_v1.jsonl",
        "phase4_unitree_runtime_evidence_bridge/unitree_operator_recovery_drill_receipts_v1.jsonl",
        "phase4_unitree_runtime_evidence_bridge/unitree_operator_recovery_drill_transitions_v1.jsonl",
        "phase4_unitree_runtime_evidence_bridge/unitree_safety_envelope_expansion_receipts_v1.jsonl",
        "phase4_unitree_blocker_stress_probes/unitree_blocker_stress_probe_receipts_v1.jsonl",
        "phase4_unitree_blocker_stress_probes/unitree_mujoco_model_stress_receipts_v1.jsonl",
        "phase6_transport_advisory_runtime/wm_transport_decomposed_eval_reports_v1.jsonl",
        "phase65_meta_node_neuralization/meta_node_trajectory_receipts_v1.jsonl",
        "phase65_meta_node_neuralization/meta_node_robustness_reports_v1.jsonl",
        "phase7_meta_governance_eval/phase7_outcome_join_rows_v1.jsonl",
        "phase7_meta_governance_eval/phase7_control_field_eval_reports_v1.jsonl",
    ]
    for index, relative in enumerate(jsonl_paths):
        write_jsonl(
            root / relative,
            [{"receipt_id": f"fixture_receipt_{index}", "report_id": f"fixture_report_{index}"}],
        )
    write_jsonl(
        root
        / "phase7_meta_regal_shadow_runtime"
        / "phase7_control_field_runtime_receipts.jsonl",
        [{"receipt_id": "fixture_phase7_field_receipt"}],
    )
    write_jsonl(
        root
        / "phase7_meta_regal_shadow_runtime"
        / "phase7_conflict_runtime_join_receipts.jsonl",
        [{"receipt_id": "fixture_phase7_conflict_receipt"}],
    )


def test_phase7_signal_adapters_feed_runtime_from_lower_wm_receipts(tmp_path):
    phase65_dir = tmp_path / "phase65"
    closure_dir = tmp_path / "closure"
    scaffold_dir = tmp_path / "phase7_scaffold"
    adapter_dir = tmp_path / "phase7_signal_adapters"
    runtime_dir = tmp_path / "phase7_shadow_runtime"
    eval_dir = tmp_path / "phase7_eval"
    lower_root = tmp_path / "lower_artifacts"
    _fake_phase65_report(phase65_dir)
    _fake_closure_audit(closure_dir)
    _write_lower_wm_signal_fixture(lower_root)
    run_prepare_phase7_meta_regal_control_scaffold(
        output_dir=scaffold_dir,
        phase65_dir=phase65_dir,
        closure_dir=closure_dir,
        run_dependencies_if_missing=False,
    )

    payload = run_adapt_phase7_governance_node_signals(
        output_dir=adapter_dir,
        phase7_scaffold_dir=scaffold_dir,
        lower_artifact_root=lower_root,
        phase7_runtime_dir=lower_root / "phase7_meta_regal_shadow_runtime",
        phase7_eval_dir=lower_root / "phase7_meta_governance_eval",
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["adapter_count"] == len(EXPECTED_PHASE7_GOVERNANCE_NODE_KEYS)
    assert payload["signal_receipt_count"] == len(EXPECTED_PHASE7_GOVERNANCE_NODE_KEYS)
    assert payload["all_eight_nodes_signal_backed"] is True
    assert payload["shadow_runtime_feed_ready"] is True
    assert payload["missing_source_artifact_count"] == 0
    assert payload["phase7_authority_granted"] is False
    assert payload["live_dispatch_allowed"] is False
    assert payload["training_executed"] is False
    assert payload["weights_written"] is False
    assert payload["promotion_eligible"] is False
    assert not any(payload["denied_gates"].values())

    receipts = load_phase7_governance_node_signal_receipts(
        adapter_dir / "phase7_governance_node_signal_receipts_v1.jsonl"
    )
    assert {receipt.node_key for receipt in receipts} == set(
        EXPECTED_PHASE7_GOVERNANCE_NODE_KEYS
    )
    assert all(receipt.lower_wm_receipt_backed for receipt in receipts)
    assert all(receipt.shadow_only for receipt in receipts)
    assert all(not receipt.live_dispatch_allowed for receipt in receipts)

    runtime_payload = run_wire_phase7_meta_regal_runtime_shadow(
        output_dir=runtime_dir,
        phase7_scaffold_dir=scaffold_dir,
        phase7_signal_adapter_dir=adapter_dir,
        episodes=1,
        timestamp_base="2026-05-25T00:00:00+00:00",
        run_id="phase7_signal_adapter_runtime_test",
        run_dependencies_if_missing=False,
    )

    phase7 = runtime_payload["phase7_meta_regal_shadow"]
    assert phase7["node_signal_receipt_count"] == len(
        EXPECTED_PHASE7_GOVERNANCE_NODE_KEYS
    )
    assert phase7["lower_wm_signal_backed"] is True
    assert phase7["local_shadow_runtime_wiring_complete"] is True
    assert phase7["live_dispatch_allowed"] is False
    assert phase7["training_executed"] is False
    assert phase7["promotion_eligible"] is False

    event_spine = json.loads((runtime_dir / "event_spine.json").read_text())
    phase7_events = [
        event
        for event in event_spine["events"]
        if str(event["event_kind"]).startswith("phase7_")
    ]
    assert len(phase7_events) == 13
    assert all(event["metadata"]["lower_wm_signal_backed"] for event in phase7_events)
    assert all(event["metadata"]["node_signal_receipt_ids"] for event in phase7_events)

    eval_payload = run_evaluate_phase7_meta_governance_shadow(
        output_dir=eval_dir,
        phase7_runtime_dir=runtime_dir,
        run_dependencies_if_missing=False,
    )
    assert eval_payload["status"] == "ok"
    field_evals = (
        eval_dir / "phase7_control_field_eval_reports_v1.jsonl"
    ).read_text().splitlines()
    assert field_evals
    assert all(
        json.loads(line)["metrics"]["lower_wm_signal_backed"] == 1.0
        for line in field_evals
    )

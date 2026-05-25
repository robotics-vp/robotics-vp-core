from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.prepare_phase7_meta_regal_control_scaffold import (
    run_prepare_phase7_meta_regal_control_scaffold,
)
from scripts.economic_world_model.wire_phase7_meta_regal_runtime_shadow import (
    run_wire_phase7_meta_regal_runtime_shadow,
)
from src.world_model.humanoid_readiness import (
    Phase35465LocalClosureAudit,
    Phase65MetaNodeNeuralizationReport,
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


def test_phase7_shadow_runtime_wiring_emits_event_spine_without_authority(tmp_path):
    phase65_dir = tmp_path / "phase65"
    closure_dir = tmp_path / "closure"
    scaffold_dir = tmp_path / "phase7_scaffold"
    shadow_dir = tmp_path / "phase7_shadow_runtime"
    _fake_phase65_report(phase65_dir)
    _fake_closure_audit(closure_dir)
    run_prepare_phase7_meta_regal_control_scaffold(
        output_dir=scaffold_dir,
        phase65_dir=phase65_dir,
        closure_dir=closure_dir,
        run_dependencies_if_missing=False,
    )

    payload = run_wire_phase7_meta_regal_runtime_shadow(
        output_dir=shadow_dir,
        phase7_scaffold_dir=scaffold_dir,
        episodes=1,
        timestamp_base="2026-05-25T00:00:00+00:00",
        run_id="phase7_shadow_test",
        run_dependencies_if_missing=False,
    )

    phase7 = payload["phase7_meta_regal_shadow"]
    assert phase7["enabled"] is True
    assert phase7["episode_report_count"] == 1
    assert phase7["control_field_runtime_receipt_count"] == 7
    assert phase7["conflict_runtime_join_receipt_count"] == 6
    assert phase7["shadow_event_spine_wiring_executed"] is True
    assert phase7["decision_ledger_wiring_executed"] is True
    assert phase7["local_shadow_runtime_wiring_complete"] is True
    assert phase7["phase7_authority_granted"] is False
    assert phase7["live_dispatch_allowed"] is False
    assert phase7["hard_veto_dispatch"] is False
    assert phase7["training_executed"] is False
    assert phase7["weights_written"] is False
    assert phase7["provider_executed"] is False
    assert phase7["hardware_executed"] is False
    assert phase7["unitree_sim_runtime_executed"] is False
    assert phase7["live_policy_control"] is False
    assert phase7["reward_math_mutation"] is False
    assert phase7["promotion_eligible"] is False

    event_spine = json.loads((shadow_dir / "event_spine.json").read_text())
    decision_ledger = json.loads((shadow_dir / "decision_ledger.json").read_text())
    phase7_events = [
        event
        for event in event_spine["events"]
        if str(event["event_kind"]).startswith("phase7_")
    ]
    phase7_decisions = [
        decision
        for decision in decision_ledger["decisions"]
        if str(decision["decision_kind"]).startswith("phase7_")
    ]
    assert len(phase7_events) == 13
    assert len(phase7_decisions) == 13
    assert {
        "phase7_control_field_shadow_emitted",
        "phase7_conflict_override_shadow_joined",
    }.issubset({event["event_kind"] for event in phase7_events})
    assert {
        "phase7_control_field_shadow_recorded",
        "phase7_conflict_override_shadow_recorded",
    }.issubset({decision["decision_kind"] for decision in phase7_decisions})

    for event in phase7_events:
        metadata = event["metadata"]
        assert metadata["shadow_only"] is True
        assert metadata["live_dispatch_allowed"] is False
        assert metadata["phase7_authority_granted"] is False
        assert metadata["reward_math_mutation"] is False
        assert metadata["promotion_eligible"] is False

    wiring = json.loads((shadow_dir / "phase7_shadow_runtime_wiring.json").read_text())
    assert wiring["summary"]["control_field_runtime_receipt_count"] == 7
    assert wiring["summary"]["conflict_runtime_join_receipt_count"] == 6
    assert wiring["summary"]["local_shadow_runtime_wiring_complete"] is True
    assert wiring["reports"][0]["local_shadow_runtime_wiring_complete"] is True
    assert wiring["reports"][0]["live_dispatch_allowed"] is False
    assert wiring["reports"][0]["phase7_authority_granted"] is False

    field_receipts = (
        shadow_dir / "phase7_control_field_runtime_receipts.jsonl"
    ).read_text().strip().splitlines()
    conflict_receipts = (
        shadow_dir / "phase7_conflict_runtime_join_receipts.jsonl"
    ).read_text().strip().splitlines()
    assert len(field_receipts) == 7
    assert len(conflict_receipts) == 6

from __future__ import annotations

from pathlib import Path

from scripts.economic_world_model.compile_neural_trainability_audit import (
    run_compile_neural_trainability_audit,
)
from src.world_model.economic_world_model.neural_trainability_audit import (
    FOLLOWUP_BLOCKERS,
    FOLLOWUP_PLANES,
    build_neural_trainability_audit,
    load_neural_trainability_audit_report,
    load_neural_trainability_components,
    load_neural_trainability_followups,
    validate_neural_trainability_audit,
)


def test_neural_trainability_audit_covers_wm_components_and_backlog() -> None:
    report, components, followups = build_neural_trainability_audit()
    by_id = {component.component_id: component for component in components}

    assert report.status == "ok_neural_trainability_audit_non_training"
    assert report.component_count == len(components)
    assert report.followup_count == len(followups)
    assert report.ready_for_training_count == 0
    assert report.promotion_eligible_count == 0
    assert report.training_executed is False
    assert report.weights_written is False
    assert report.provider_executed is False
    assert report.gpu_executed is False
    assert report.hardware_executed is False
    assert report.phase7_authority_granted is False
    assert report.promotion_eligible is False
    assert not any(report.denied_gates.values())

    required_components = {
        "perception_evidence_fusion_seam",
        "perception_vjepa_temporal_alignment",
        "vision_backbone_projection_head",
        "sim_synth_predictive_vjepa_component",
        "embodiment_phase34_neural_architecture",
        "economic_world_model_neural_components",
        "wm_transport_bridge_receiver_trainer",
        "phase65_meta_node_trainer",
        "phase7_signal_adapter_consumer",
        "bio_neuro_trainability_bundle",
        "orchestrator_semantic_runtime_trainers",
        "vla_openvla_recap_heads",
        "rl_hrl_curriculum_policy_family",
    }
    assert required_components <= set(by_id)
    assert any(
        component.component_id.startswith("training_backlog_")
        for component in components
    )
    assert "trainer/runtime lane" in report.surface_role_counts
    assert "lower-WM producer" in report.surface_role_counts
    assert "provider/hardware adapter" in report.surface_role_counts

    transport = by_id["wm_transport_bridge_receiver_trainer"]
    assert "wm_transport_unitree_event_spine_joins_v1" in transport.receipt_refs
    assert transport.ready_for_training is False
    assert transport.promotion_eligible is False
    assert transport.source_backlog_rows


def test_neural_trainability_followups_are_executable_and_fail_closed() -> None:
    report, components, followups = build_neural_trainability_audit()
    component_ids = {component.component_id for component in components}

    assert report.plane_counts["runpod_provider"] > 0
    assert report.plane_counts["runpod_train"] > 0
    assert report.plane_counts["hardware_runtime"] > 0
    assert report.plane_counts["local"] == 0
    assert report.blocker_counts["provider"] > 0
    assert report.blocker_counts["gpu"] > 0
    assert report.blocker_counts["hardware"] > 0
    assert report.blocker_counts["benchmark_missing"] > 0

    for row in followups:
        assert row.component_id in component_ids
        assert row.plane in FOLLOWUP_PLANES
        assert row.blocker in FOLLOWUP_BLOCKERS
        assert row.action
        assert row.target
        assert row.verify_receipt
        assert row.promotion_eligible is False

    assert any(
        row.component_id == "phase7_signal_adapter_consumer"
        and row.plane == "runpod_train"
        and row.blocker == "data"
        for row in followups
    )
    assert not any(row.plane == "local" for row in followups)
    assert any(
        row.component_id == "embodiment_phase34_neural_architecture"
        and row.plane == "hardware_runtime"
        for row in followups
    )

    validation = validate_neural_trainability_audit(
        report=report,
        components=components,
        followups=followups,
    )
    assert validation["status"] == "ok"
    assert validation["error_count"] == 0
    assert validation["safe_for_training"] is False
    assert validation["safe_for_promotion"] is False


def test_compile_neural_trainability_audit_writes_loadable_artifacts(
    tmp_path: Path,
) -> None:
    payload = run_compile_neural_trainability_audit(output_dir=tmp_path)

    assert payload["status"] == "ok_neural_trainability_audit_non_training"
    assert payload["component_count"] >= 20
    assert payload["followup_count"] >= payload["component_count"]
    assert payload["ready_for_training_count"] == 0
    assert payload["promotion_eligible_count"] == 0
    assert payload["validation"]["status"] == "ok"
    assert payload["validation"]["safe_for_training"] is False
    assert payload["validation"]["safe_for_promotion"] is False

    refs = payload["artifact_refs"]
    report = load_neural_trainability_audit_report(refs["report_path"])
    components = load_neural_trainability_components(refs["components_path"])
    followups = load_neural_trainability_followups(refs["followups_path"])

    assert report.audit_id == payload["audit_id"]
    assert len(components) == payload["component_count"]
    assert len(followups) == payload["followup_count"]
    assert Path(refs["markdown_path"]).exists()
    assert Path(refs["validation_path"]).exists()
    assert any(
        component.component_id == "bio_neuro_trainability_bundle"
        for component in components
    )
    assert any(row.verify_receipt == "openvla_provider_receipt_v1" for row in followups)

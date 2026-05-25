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
    Phase7GovernanceNodeSignalAdapter,
    Phase7GovernanceNodeSignalReceipt,
    Phase7GovernanceSignalAdapterReport,
    build_phase7_meta_composition_hypernetwork_scaffold,
    load_phase7_composition_mode_specs,
    load_phase7_conflict_join_eval_reports,
    load_phase7_conflict_override_receipts,
    load_phase7_control_field_eval_reports,
    load_phase7_control_field_slots,
    load_phase7_governance_node_surfaces,
    load_phase7_hypernetwork_conditioning_specs,
    load_phase7_hypernetwork_cpu_smoke_forward,
    load_phase7_hypernetwork_dataset_contract,
    load_phase7_hypernetwork_model_config,
    load_phase7_hypernetwork_output_heads,
    load_phase7_meta_composition_hypernetwork_scaffold_report,
    load_phase7_meta_composition_losses,
    load_phase7_meta_governance_evaluation_report,
    load_phase7_meta_regal_control_scaffold_report,
    load_phase7_outcome_join_rows,
    load_phase7_pareto_regime_eval_reports,
    load_phase7_promotion_gates,
    load_phase7_training_row_slots,
    save_phase35465_local_closure_audit,
    save_phase7_meta_composition_hypernetwork_scaffold,
)
from src.world_model.humanoid_readiness.common import load_json, write_json


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
        remaining_evidence_blockers=["gpu_training_provider_hardware_evidence_missing"],
    )
    path = closure_dir / "phase35_4_65_local_closure_audit_v1.json"
    save_phase35465_local_closure_audit(path, audit)
    return path


def _fake_signal_artifacts(surfaces):
    adapters = [
        Phase7GovernanceNodeSignalAdapter(
            adapter_id=f"adapter_{surface.node_key}",
            surface_id=surface.surface_id,
            node_key=surface.node_key,
            domain_key=surface.domain_key,
            source_artifact_refs={"fixture": "lower_wm_fixture"},
            source_receipt_ids=[f"receipt_{surface.node_key}"],
            source_receipt_families=["fixture_lower_wm_receipt"],
            metrics={"source_receipt_count": 1.0},
            signal_slots={"confidence_source": "fixture"},
        )
        for surface in surfaces
    ]
    receipts = [
        Phase7GovernanceNodeSignalReceipt(
            signal_id=f"signal_{surface.node_key}",
            adapter_id=f"adapter_{surface.node_key}",
            surface_id=surface.surface_id,
            node_key=surface.node_key,
            domain_key=surface.domain_key,
            signal_key=f"{surface.node_key}_signal",
            source_receipt_ids=[f"receipt_{surface.node_key}"],
            source_artifact_refs={"fixture": "lower_wm_fixture"},
            confidence=0.5,
            candidate_outputs={"surface_output_refs": surface.output_refs},
            hard_constraint_candidate=surface.hard_constraint_capable,
        )
        for surface in surfaces
    ]
    report = Phase7GovernanceSignalAdapterReport(
        report_id="phase7_signal_adapter_fixture",
        phase7_scaffold_report_id="phase7_scaffold_fixture",
        status="ok",
        governance_node_surface_count=len(surfaces),
        adapter_count=len(adapters),
        signal_receipt_count=len(receipts),
        source_artifact_count=8,
        missing_source_artifact_count=0,
        lower_wm_receipt_backed_node_count=len(receipts),
        all_eight_nodes_signal_backed=True,
        shadow_runtime_feed_ready=True,
        local_signal_adapter_complete=True,
        remaining_blockers=["labeled_governance_signal_outcomes_missing"],
    )
    return report, adapters, receipts


def test_phase7_meta_composition_hypernetwork_scaffold(tmp_path):
    phase65_dir = tmp_path / "phase65"
    closure_dir = tmp_path / "closure"
    scaffold_dir = tmp_path / "phase7_scaffold"
    runtime_dir = tmp_path / "phase7_runtime"
    eval_dir = tmp_path / "phase7_eval"
    output_dir = tmp_path / "phase7_hypernetwork"
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
        run_id="phase7_hypernetwork_test",
        run_dependencies_if_missing=False,
    )
    run_evaluate_phase7_meta_governance_shadow(
        output_dir=eval_dir,
        phase7_runtime_dir=runtime_dir,
        run_dependencies_if_missing=False,
    )

    phase7_report = load_phase7_meta_regal_control_scaffold_report(
        scaffold_dir / "phase7_meta_regal_control_scaffold_report_v1.json"
    )
    surfaces = load_phase7_governance_node_surfaces(
        scaffold_dir / "phase7_governance_node_surfaces_v1.jsonl"
    )
    signal_report, signal_adapters, signal_receipts = _fake_signal_artifacts(surfaces)

    (
        report,
        conditioning_specs,
        output_heads,
        losses,
        dataset_contract,
        model_config,
        smoke_forward,
    ) = build_phase7_meta_composition_hypernetwork_scaffold(
        phase7_report=phase7_report,
        surfaces=surfaces,
        modes=load_phase7_composition_mode_specs(
            scaffold_dir / "phase7_composition_mode_specs_v1.jsonl"
        ),
        conflicts=load_phase7_conflict_override_receipts(
            scaffold_dir / "phase7_conflict_override_receipts_v1.jsonl"
        ),
        control_fields=load_phase7_control_field_slots(
            scaffold_dir / "phase7_control_field_slots_v1.jsonl"
        ),
        training_rows=load_phase7_training_row_slots(
            scaffold_dir / "phase7_training_row_slots_v1.jsonl"
        ),
        promotion_gates=load_phase7_promotion_gates(
            scaffold_dir / "phase7_promotion_gates_v1.jsonl"
        ),
        signal_report=signal_report,
        signal_adapters=signal_adapters,
        signal_receipts=signal_receipts,
        eval_report=load_phase7_meta_governance_evaluation_report(
            eval_dir / "phase7_meta_governance_evaluation_report_v1.json"
        ),
        field_evals=load_phase7_control_field_eval_reports(
            eval_dir / "phase7_control_field_eval_reports_v1.jsonl"
        ),
        conflict_evals=load_phase7_conflict_join_eval_reports(
            eval_dir / "phase7_conflict_join_eval_reports_v1.jsonl"
        ),
        regime_evals=load_phase7_pareto_regime_eval_reports(
            eval_dir / "phase7_pareto_regime_eval_reports_v1.jsonl"
        ),
        outcome_rows=load_phase7_outcome_join_rows(
            eval_dir / "phase7_outcome_join_rows_v1.jsonl"
        ),
        runtime_summary=load_json(runtime_dir / "summary.json"),
        artifact_refs={"fixture": "phase7_hypernetwork_test"},
    )
    save_phase7_meta_composition_hypernetwork_scaffold(
        output_dir,
        report,
        conditioning_specs,
        output_heads,
        losses,
        dataset_contract,
        model_config,
        smoke_forward,
    )

    assert report.status == "ok"
    assert report.local_hypernetwork_scaffold_complete is True
    assert report.conditioning_wiring_complete is True
    assert report.future_meta_composition_explicit is True
    assert report.cpu_smoke_forward_passed is True
    assert report.training_executed is False
    assert report.weights_written is False
    assert report.live_dispatch_allowed is False
    assert report.hard_veto_dispatch is False
    assert report.reward_math_mutation is False
    assert report.promotion_eligible is False
    assert not any(report.denied_gates.values())

    loaded_report = load_phase7_meta_composition_hypernetwork_scaffold_report(
        output_dir / "phase7_meta_composition_hypernetwork_scaffold_report_v1.json"
    )
    loaded_specs = load_phase7_hypernetwork_conditioning_specs(
        output_dir / "phase7_hypernetwork_conditioning_specs_v1.jsonl"
    )
    loaded_heads = load_phase7_hypernetwork_output_heads(
        output_dir / "phase7_hypernetwork_output_heads_v1.jsonl"
    )
    loaded_losses = load_phase7_meta_composition_losses(
        output_dir / "phase7_meta_composition_losses_v1.json"
    )
    loaded_dataset = load_phase7_hypernetwork_dataset_contract(
        output_dir / "phase7_hypernetwork_dataset_contract_v1.json"
    )
    loaded_model = load_phase7_hypernetwork_model_config(
        output_dir / "phase7_hypernetwork_model_config_v1.json"
    )
    loaded_smoke = load_phase7_hypernetwork_cpu_smoke_forward(
        output_dir / "phase7_hypernetwork_cpu_smoke_forward_v1.json"
    )

    assert loaded_report.future_meta_composition_explicit is True
    assert {spec.conditioning_key for spec in loaded_specs} == {
        "node_signal_conditioning",
        "conflict_context_conditioning",
        "pareto_regime_conditioning",
        "shadow_outcome_conditioning",
        "runtime_truth_and_denial_conditioning",
    }
    pareto_spec = next(
        spec
        for spec in loaded_specs
        if spec.conditioning_key == "pareto_regime_conditioning"
    )
    assert "economic WM remains one governance voice" in pareto_spec.meta_composition_semantics
    assert {head.head_key for head in loaded_heads}.issuperset(
        {"node_gate_logits", "pareto_regime_parameters", "veto_candidate_calibration"}
    )
    assert {loss.loss_key for loss in loaded_losses}.issuperset(
        {
            "composition_mode_classification_loss",
            "pareto_frontier_regime_loss",
            "promotion_denial_regularizer",
        }
    )
    assert loaded_dataset.ready_for_cpu_smoke_forward is True
    assert "meta_composition_hypernetwork" in {
        component["component_key"] for component in loaded_model.components
    }
    assert loaded_model.future_meta_composition_wiring["current_wiring"]
    assert loaded_smoke.smoke_forward_passed is True
    assert loaded_smoke.weights_written is False

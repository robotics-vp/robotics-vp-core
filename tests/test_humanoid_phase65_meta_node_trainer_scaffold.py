from __future__ import annotations

from pathlib import Path

from scripts.economic_world_model.build_phase65_meta_node_trainer_scaffold import (
    run_build_phase65_meta_node_trainer_scaffold,
)
from src.world_model.humanoid_readiness import (
    MetaNodeCounterfactualTarget,
    MetaNodeInterventionReceipt,
    MetaNodePromotionGate,
    MetaNodeRobustnessReport,
    MetaNodeState,
    MetaNodeTrajectoryReceipt,
    Phase65MetaNodeNeuralizationReport,
    load_phase65_meta_node_cpu_smoke_forward,
    load_phase65_meta_node_loss_definitions,
    load_phase65_meta_node_model_component_config,
    load_phase65_meta_node_trainer_dataset_contract,
    load_phase65_meta_node_trainer_scaffold_manifest,
)
from src.world_model.humanoid_readiness.common import write_json, write_jsonl


def _write_phase65_fixture(root: Path) -> None:
    report = Phase65MetaNodeNeuralizationReport(
        report_id="phase65_trainer_test",
        phase35_report_id="phase35_test",
        phase4_report_id="phase4_test",
        phase6_closure_audit_id="phase6_test",
        status="ok",
        node_state_count=2,
        trajectory_receipt_count=2,
        intervention_receipt_count=2,
        counterfactual_target_count=2,
        robustness_report_count=2,
        promotion_gate_count=2,
        local_meta_node_scaffold_complete=True,
        ready_for_phase7_scaffold=True,
        remaining_blockers=["counterfactual_meta_node_corpus_density_missing"],
    )
    states = [
        MetaNodeState(
            node_id="meta_node_test_a",
            node_family="test_a",
            activation_scope="economic_allocation_and_resource_routing",
            posture_scope="bipedal_whole_body",
            input_refs=["input_a"],
            target_refs=["target_a"],
            neighbor_node_ids=["meta_node_test_b"],
            confidence_prior=0.5,
            activation_strength_prior=0.25,
        ),
        MetaNodeState(
            node_id="meta_node_test_b",
            node_family="test_b",
            activation_scope="operator_handoff_and_recovery_scope",
            posture_scope="bipedal_whole_body",
            input_refs=["input_b"],
            target_refs=["target_b"],
            neighbor_node_ids=["meta_node_test_a"],
            confidence_prior=0.4,
            activation_strength_prior=0.2,
        ),
    ]
    trajectories = [
        MetaNodeTrajectoryReceipt(
            receipt_id=f"trajectory_{state.node_id}",
            node_id=state.node_id,
            trajectory_events=["activation_candidate_created", "promotion_gate_denied"],
        )
        for state in states
    ]
    interventions = [
        MetaNodeInterventionReceipt(
            receipt_id=f"intervention_{state.node_id}",
            node_id=state.node_id,
            intervention_kind="veto",
            rationale="test",
            target_refs=state.target_refs,
        )
        for state in states
    ]
    targets = [
        MetaNodeCounterfactualTarget(
            target_id=f"target_{state.node_id}",
            node_id=state.node_id,
            target_family="activation_timing_strength_and_downstream_effect",
            label_slots={"activation_timing": "awaiting_postmortem_label"},
            downstream_effect_slots={"governance_satisfaction": None},
        )
        for state in states
    ]
    robustness = [
        MetaNodeRobustnessReport(
            report_id=f"robustness_{state.node_id}",
            node_id=state.node_id,
            metrics={
                "surface_completeness": 1.0,
                "activation_calibration_evidence": 0.0,
                "neighbor_consistency_benchmark_evidence": 0.0,
                "deployment_robustness_evidence": 0.0,
            },
        )
        for state in states
    ]
    gates = [
        MetaNodePromotionGate(
            gate_id=f"gate_{state.node_id}",
            node_id=state.node_id,
            requested_authority="phase7_control_wm_authority",
            missing_evidence=["gpu_meta_node_training_not_run"],
        )
        for state in states
    ]

    write_json(root / "phase65_meta_node_neuralization_report_v1.json", report.to_dict())
    write_jsonl(root / "meta_node_states_v1.jsonl", [item.to_dict() for item in states])
    write_jsonl(
        root / "meta_node_trajectory_receipts_v1.jsonl",
        [item.to_dict() for item in trajectories],
    )
    write_jsonl(
        root / "meta_node_intervention_receipts_v1.jsonl",
        [item.to_dict() for item in interventions],
    )
    write_jsonl(
        root / "meta_node_counterfactual_targets_v1.jsonl",
        [item.to_dict() for item in targets],
    )
    write_jsonl(
        root / "meta_node_robustness_reports_v1.jsonl",
        [item.to_dict() for item in robustness],
    )
    write_jsonl(root / "meta_node_promotion_gates_v1.jsonl", [item.to_dict() for item in gates])


def test_phase65_meta_node_trainer_scaffold_contracts(tmp_path):
    phase65_dir = tmp_path / "phase65"
    output_dir = tmp_path / "phase65_trainer"
    _write_phase65_fixture(phase65_dir)

    payload = run_build_phase65_meta_node_trainer_scaffold(
        output_dir=output_dir,
        phase65_dir=phase65_dir,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["dataset_contract_ready"] is True
    assert payload["losses_defined"] is True
    assert payload["model_config_ready"] is True
    assert payload["cpu_smoke_forward_passed"] is True
    assert payload["ready_for_training"] is False
    assert payload["ready_for_gpu_training"] is False
    assert payload["phase7_authority_granted"] is False
    assert payload["training_executed"] is False
    assert payload["weights_written"] is False
    assert payload["live_policy_control"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["promotion_eligible"] is False
    assert not any(payload["denied_gates"].values())

    manifest = load_phase65_meta_node_trainer_scaffold_manifest(
        output_dir / "phase65_meta_node_trainer_scaffold_manifest_v1.json"
    )
    dataset = load_phase65_meta_node_trainer_dataset_contract(
        output_dir / "phase65_meta_node_trainer_dataset_contract_v1.json"
    )
    losses = load_phase65_meta_node_loss_definitions(
        output_dir / "phase65_meta_node_loss_definitions_v1.json"
    )
    model_config = load_phase65_meta_node_model_component_config(
        output_dir / "phase65_meta_node_model_component_config_v1.json"
    )
    smoke = load_phase65_meta_node_cpu_smoke_forward(
        output_dir / "phase65_meta_node_cpu_smoke_forward_v1.json"
    )

    assert manifest.cpu_smoke_forward_passed is True
    assert dataset.feature_dim > 0
    assert dataset.target_dim > 0
    assert dataset.ready_for_cpu_smoke_forward is True
    assert {loss.loss_key for loss in losses}.issuperset(
        {
            "activation_timing_loss",
            "counterfactual_downstream_effect_loss",
            "promotion_denial_regularizer",
        }
    )
    assert all(loss.optimization_status == "defined_not_optimized" for loss in losses)
    assert all(not loss.direct_policy_rl for loss in losses)
    assert model_config.component_count >= 4
    assert all(not component["weights_written"] for component in model_config.components)
    assert smoke.smoke_forward_passed is True
    assert smoke.weights_written is False

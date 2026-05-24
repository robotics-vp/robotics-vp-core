from __future__ import annotations

from scripts.economic_world_model.audit_phase35_bipedal_readiness import (
    run_audit_phase35_bipedal_readiness,
)
from scripts.economic_world_model.prepare_phase35_humanoid_capacity_env_refit import (
    run_prepare_phase35_humanoid_capacity_env_refit,
)
from scripts.economic_world_model.prepare_phase4_deployment_enabler_sweep import (
    run_prepare_phase4_deployment_enabler_sweep,
)
from scripts.economic_world_model.prepare_phase4_downstream_controller_scaffold import (
    run_prepare_phase4_downstream_controller_scaffold,
)
from src.world_model.humanoid_readiness import (
    DENIED_DOWNSTREAM_CONTROLLER_AUTHORITIES,
    load_controller_bridge_targets,
    load_controller_invocations,
    load_controller_mode_specs,
    load_controller_receipts,
    load_controller_safety_receipts,
    load_downstream_controller_proposals,
    load_low_level_command_frames,
    load_phase4_downstream_controller_scaffold_report,
)


def test_phase4_downstream_controller_scaffold_emits_dry_run_receipts(tmp_path):
    phase35_dir = tmp_path / "phase35"
    bipedal_chassis_dir = tmp_path / "bipedal_chassis"
    readiness_dir = tmp_path / "phase35_bipedal_readiness"
    phase4_dir = tmp_path / "phase4"
    controller_dir = tmp_path / "phase4_downstream_controller"

    run_prepare_phase35_humanoid_capacity_env_refit(
        output_dir=phase35_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
    )
    run_audit_phase35_bipedal_readiness(
        output_dir=readiness_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        run_dependencies_if_missing=False,
    )
    run_prepare_phase4_deployment_enabler_sweep(
        output_dir=phase4_dir,
        phase35_dir=phase35_dir,
        run_dependencies_if_missing=False,
    )

    payload = run_prepare_phase4_downstream_controller_scaffold(
        output_dir=controller_dir,
        phase4_dir=phase4_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=readiness_dir,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["local_downstream_controller_scaffold_complete"] is True
    assert payload["unitree_bridge_contract_present"] is True
    assert payload["g1pilot_fallback_contract_present"] is True
    assert payload["dry_run_controller_present"] is True
    assert payload["bridge_target_count"] == 5
    assert payload["mode_count"] == 6
    assert payload["proposal_count"] == 6
    assert payload["command_frame_count"] == 6
    assert payload["safety_receipt_count"] == 6
    assert payload["invocation_count"] == 6
    assert payload["controller_receipt_count"] == 6
    assert payload["hardware_dispatch_enabled"] is False
    assert payload["ros2_publish_attempted"] is False
    assert payload["unitree_sdk2_write_enabled"] is False
    assert payload["g1pilot_runtime_invoked"] is False
    assert payload["live_policy_control"] is False
    assert payload["training_executed"] is False
    assert payload["weights_written"] is False
    assert payload["provider_executed"] is False
    assert payload["hardware_executed"] is False
    assert payload["unitree_sim_runtime_executed"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["promotion_eligible"] is False
    assert not any(
        payload["denied_gates"][key]
        for key in DENIED_DOWNSTREAM_CONTROLLER_AUTHORITIES
    )

    report = load_phase4_downstream_controller_scaffold_report(
        controller_dir / "phase4_downstream_controller_scaffold_report_v1.json"
    )
    bridge_targets = load_controller_bridge_targets(
        controller_dir / "controller_bridge_targets_v1.jsonl"
    )
    modes = load_controller_mode_specs(controller_dir / "controller_mode_specs_v1.jsonl")
    proposals = load_downstream_controller_proposals(
        controller_dir / "downstream_controller_proposals_v1.jsonl"
    )
    frames = load_low_level_command_frames(
        controller_dir / "low_level_command_frames_v1.jsonl"
    )
    safety_receipts = load_controller_safety_receipts(
        controller_dir / "controller_safety_receipts_v1.jsonl"
    )
    invocations = load_controller_invocations(
        controller_dir / "controller_invocations_v1.jsonl"
    )
    receipts = load_controller_receipts(controller_dir / "controller_receipts_v1.jsonl")

    assert report.local_downstream_controller_scaffold_complete is True
    assert {target.source_project for target in bridge_targets} >= {
        "unitreerobotics/unitree_ros2",
        "hucebot/g1pilot",
    }
    assert all(target.vendored_code_included is False for target in bridge_targets)
    assert all(target.hardware_dispatch_enabled is False for target in bridge_targets)
    assert {mode.mode_name for mode in modes} >= {
        "hold_pose",
        "joint_pd_tracking",
        "cartesian_upper_body_tracking",
        "stable_base_fallback",
        "operator_teleop_pass_through",
        "e_stop_veto",
    }
    assert all(mode.dry_run_only for mode in modes)
    assert not any(mode.live_authority_allowed for mode in modes)
    assert any(
        proposal.proposal_name == "joint_pd_tracking_clamp_probe"
        for proposal in proposals
    )
    assert any(frame.clamp_applied for frame in frames)
    assert not any(frame.publish_attempted for frame in frames)
    assert not any(frame.hardware_dispatch_enabled for frame in frames)
    assert any(receipt.e_stop_vetoed for receipt in safety_receipts)
    assert all(receipt.stale_data_vetoed for receipt in safety_receipts)
    assert not any(receipt.hardware_dispatch_allowed for receipt in safety_receipts)
    assert all(
        invocation.dispatch_status == "dispatch_denied_dry_run"
        for invocation in invocations
    )
    assert not any(invocation.publish_attempted for invocation in invocations)
    assert not any(invocation.unitree_sdk2_write_enabled for invocation in invocations)
    assert all(receipt.replay_export_ready for receipt in receipts)
    assert not any(receipt.promotion_eligible for receipt in receipts)

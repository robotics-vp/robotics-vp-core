from __future__ import annotations

from scripts.economic_world_model.prepare_phase35_bipedal_chassis_scaffold import (
    run_prepare_phase35_bipedal_chassis_scaffold,
)
from scripts.economic_world_model.prepare_phase35_humanoid_capacity_env_refit import (
    run_prepare_phase35_humanoid_capacity_env_refit,
)
from src.world_model.embodiment_actuation import (
    DENIED_BIPEDAL_CHASSIS_AUTHORITIES,
    load_balance_envelope_receipts,
    load_bipedal_support_states,
    load_humanoid_chassis_profile,
    load_humanoid_frame_tree,
    load_joint_limit_envelopes,
    load_limb_coordinate_frames,
    load_whole_body_action_schema,
    load_whole_body_observation_schema,
)


def test_phase35_bipedal_chassis_scaffold_materializes_whole_body_contracts(
    tmp_path,
):
    output_dir = tmp_path / "bipedal_chassis"
    payload = run_prepare_phase35_bipedal_chassis_scaffold(output_dir=output_dir)

    assert payload["status"] == "ok"
    assert payload["local_structural_scaffold_complete"] is True
    assert payload["controlled_joint_count"] >= 21
    assert payload["joint_limit_envelope_count"] == payload["controlled_joint_count"]
    assert payload["frame_count"] >= 20
    assert payload["support_state_count"] == 3
    assert payload["balance_receipt_count"] == 3
    assert payload["canonical_bipedal_chassis_present"] is True
    assert payload["limb_frame_tree_present"] is True
    assert payload["joint_limit_envelope_present"] is True
    assert payload["whole_body_observation_schema_present"] is True
    assert payload["whole_body_action_schema_present"] is True
    assert payload["balance_envelope_present"] is True
    assert payload["ready_for_unitree_runtime"] is False
    assert payload["hardware_calibrated_limits"] is False
    assert payload["unitree_sim_runtime_executed"] is False
    assert payload["provider_executed"] is False
    assert payload["hardware_executed"] is False
    assert payload["training_executed"] is False
    assert payload["weights_written"] is False
    assert payload["live_policy_control"] is False
    assert payload["reward_math_mutation"] is False
    assert payload["promotion_eligible"] is False
    assert not any(
        payload["denied_gates"][key] for key in DENIED_BIPEDAL_CHASSIS_AUTHORITIES
    )

    chassis = load_humanoid_chassis_profile(
        output_dir / "humanoid_chassis_profile_v1.json"
    )
    frame_tree = load_humanoid_frame_tree(output_dir / "humanoid_frame_tree_v1.json")
    frames = load_limb_coordinate_frames(
        output_dir / "limb_coordinate_frames_v1.jsonl"
    )
    joint_limits = load_joint_limit_envelopes(
        output_dir / "joint_limit_envelopes_v1.jsonl"
    )
    observation_schema = load_whole_body_observation_schema(
        output_dir / "whole_body_observation_schema_v1.json"
    )
    action_schema = load_whole_body_action_schema(
        output_dir / "whole_body_action_schema_v1.json"
    )
    support_states = load_bipedal_support_states(
        output_dir / "bipedal_support_states_v1.jsonl"
    )
    balance_receipts = load_balance_envelope_receipts(
        output_dir / "balance_envelope_receipts_v1.jsonl"
    )

    assert chassis.posture_tag == "bipedal_whole_body"
    assert chassis.controlled_joint_count == 29
    assert chassis.floating_base_dof == 6
    assert chassis.minimum_total_dof == 35
    assert {"left_leg", "right_leg", "left_arm", "right_arm", "waist"}.issubset(
        set(chassis.limb_groups)
    )
    assert frame_tree.status == "ok_contract_only"
    assert frame_tree.cycle_detected is False
    assert not frame_tree.orphan_frame_ids
    assert {"pelvis", "left_foot", "right_foot", "left_hand", "right_hand"}.issubset(
        {frame.frame_id for frame in frames}
    )
    assert len(joint_limits) == chassis.controlled_joint_count
    assert all(limit.lower_rad < limit.upper_rad for limit in joint_limits)
    assert not any(limit.hardware_limit_verified for limit in joint_limits)
    assert all(limit.violation_policy == "emit_receipt_only" for limit in joint_limits)
    assert "balance" in observation_schema.channel_groups
    assert "joint_state" in observation_schema.channel_groups
    assert action_schema.action_dimension == chassis.controlled_joint_count
    assert "stable_base_fallback_required_when_balance_evidence_missing" in (
        action_schema.support_phase_constraints
    )
    assert {state.support_phase for state in support_states} == {
        "double_support",
        "left_single_support",
        "right_single_support",
    }
    assert all(receipt.observational_only for receipt in balance_receipts)
    assert not any(receipt.promotion_eligible for receipt in balance_receipts)
    assert not any(receipt.live_policy_control for receipt in balance_receipts)


def test_phase35_refit_consumes_bipedal_chassis_scaffold(tmp_path):
    payload = run_prepare_phase35_humanoid_capacity_env_refit(
        output_dir=tmp_path / "phase35",
        bipedal_chassis_dir=tmp_path / "bipedal_chassis",
    )

    assert payload["status"] == "ok"
    assert payload["local_structural_refit_complete"] is True
    assert payload["bipedal_chassis_local_scaffold_complete"] is True
    assert payload["bipedal_chassis_joint_count"] >= 21
    assert payload["bipedal_chassis_frame_count"] >= 20
    assert payload["bipedal_chassis_joint_limit_envelope_count"] == (
        payload["bipedal_chassis_joint_count"]
    )
    assert payload["bipedal_balance_receipt_count"] == 3
    assert payload["canonical_bipedal_chassis_present"] is True
    assert payload["limb_frame_tree_present"] is True
    assert payload["joint_limit_envelope_present"] is True
    assert payload["whole_body_observation_schema_present"] is True
    assert payload["whole_body_action_schema_present"] is True
    assert payload["balance_envelope_present"] is True
    assert payload["ready_for_unitree_runtime"] is False
    assert payload["unitree_sim_runtime_executed"] is False
    assert payload["hardware_executed"] is False
    assert payload["promotion_eligible"] is False

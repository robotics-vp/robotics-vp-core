from __future__ import annotations

from pathlib import Path

from scripts.economic_world_model.audit_phase35_bipedal_readiness import (
    run_audit_phase35_bipedal_readiness,
)
from scripts.economic_world_model.prepare_phase35_bipedal_chassis_scaffold import (
    run_prepare_phase35_bipedal_chassis_scaffold,
)
from src.world_model.embodiment_actuation import (
    DENIED_BIPEDAL_READINESS_AUTHORITIES,
    load_balance_geometry_reports,
    load_humanoid_chassis_profile,
    load_humanoid_robot_asset_contract,
    load_joint_vector_validation_receipts,
    load_kinematic_consistency_report,
    load_phase35_bipedal_readiness_audit,
    load_robot_asset_parse_receipts,
    load_whole_body_replay_rows,
)


def test_phase35_bipedal_readiness_audit_closes_no_hardware_surfaces(tmp_path):
    chassis_dir = tmp_path / "bipedal_chassis"
    readiness_dir = tmp_path / "readiness"
    run_prepare_phase35_bipedal_chassis_scaffold(output_dir=chassis_dir)

    payload = run_audit_phase35_bipedal_readiness(
        output_dir=readiness_dir,
        bipedal_chassis_dir=chassis_dir,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["phase35_no_gpu_no_hardware_prepared"] is True
    assert payload["local_asset_ingestion_contract_present"] is True
    assert payload["asset_parse_receipt_count"] == 1
    assert payload["real_asset_parsed"] is False
    assert payload["kinematic_validators_present"] is True
    assert payload["joint_vector_validation_receipt_count"] == 2
    assert payload["balance_geometry_report_count"] == 3
    assert payload["whole_body_replay_row_count"] == 3
    assert payload["ready_for_unitree_runtime"] is False
    assert payload["ready_for_training"] is False
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
        payload["denied_gates"][key] for key in DENIED_BIPEDAL_READINESS_AUTHORITIES
    )

    audit = load_phase35_bipedal_readiness_audit(
        readiness_dir / "phase35_bipedal_readiness_audit_v1.json"
    )
    contract = load_humanoid_robot_asset_contract(
        readiness_dir / "humanoid_robot_asset_contract_v1.json"
    )
    parse_receipts = load_robot_asset_parse_receipts(
        readiness_dir / "robot_asset_parse_receipts_v1.jsonl"
    )
    kinematic = load_kinematic_consistency_report(
        readiness_dir / "kinematic_consistency_report_v1.json"
    )
    joint_receipts = load_joint_vector_validation_receipts(
        readiness_dir / "joint_vector_validation_receipts_v1.jsonl"
    )
    balance_reports = load_balance_geometry_reports(
        readiness_dir / "balance_geometry_reports_v1.jsonl"
    )
    replay_rows = load_whole_body_replay_rows(
        readiness_dir / "whole_body_replay_rows_v1.jsonl"
    )

    assert audit.phase35_no_gpu_no_hardware_prepared is True
    assert contract.asset_status == "assets_unavailable"
    assert contract.parser_status == "not_run_without_assets"
    assert contract.real_asset_parsed is False
    assert parse_receipts[0].status == "unavailable_no_asset_paths"
    assert parse_receipts[0].real_asset_parsed is False
    assert kinematic.status == "ok_contract_only"
    assert kinematic.minimum_21dof_invariant_passed is True
    assert kinematic.action_channel_alignment_passed is True
    assert kinematic.joint_limit_coverage_passed is True
    assert kinematic.frame_tree_acyclic is True
    assert kinematic.frame_tree_orphan_free is True
    assert kinematic.asset_joint_alignment_status == "no_real_asset_to_compare"
    assert {receipt.validation_kind for receipt in joint_receipts} == {
        "neutral_planning_vector",
        "synthetic_limit_violation_probe",
    }
    neutral = next(
        receipt
        for receipt in joint_receipts
        if receipt.validation_kind == "neutral_planning_vector"
    )
    probe = next(
        receipt
        for receipt in joint_receipts
        if receipt.validation_kind == "synthetic_limit_violation_probe"
    )
    assert neutral.status == "ok"
    assert neutral.violation_count == 0
    assert probe.status == "violations_observed"
    assert probe.violation_count == 1
    double_support = next(
        report
        for report in balance_reports
        if report.support_phase == "double_support"
    )
    assert double_support.polygon_area_m2 > 0.0
    assert double_support.com_inside_support is True
    assert double_support.computed_from_measured_streams is False
    assert all(row.posture_tag == "bipedal_whole_body" for row in replay_rows)
    assert not any(row.ready_for_training for row in replay_rows)
    assert not any(row.unitree_sim_runtime_executed for row in replay_rows)
    assert not any(row.provider_executed for row in replay_rows)
    assert not any(row.hardware_executed for row in replay_rows)
    assert not any(row.promotion_eligible for row in replay_rows)


def _write_synthetic_urdf(path: Path, joint_names: list[str]) -> None:
    lines = ['<robot name="synthetic_g1_contract">', '  <link name="base_link"/>']
    for joint_name in joint_names:
        child_name = f"{joint_name}_link"
        lines.extend(
            [
                f'  <link name="{child_name}"/>',
                f'  <joint name="{joint_name}" type="revolute">',
                '    <parent link="base_link"/>',
                f'    <child link="{child_name}"/>',
                '    <limit lower="-1.0" upper="1.0" effort="1.0" velocity="1.0"/>',
                "  </joint>",
            ]
        )
    lines.append("</robot>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_phase35_bipedal_readiness_parses_local_asset_contract(tmp_path):
    chassis_dir = tmp_path / "bipedal_chassis"
    readiness_dir = tmp_path / "readiness"
    run_prepare_phase35_bipedal_chassis_scaffold(output_dir=chassis_dir)
    chassis = load_humanoid_chassis_profile(
        chassis_dir / "humanoid_chassis_profile_v1.json"
    )
    asset_path = tmp_path / "synthetic_g1_contract.urdf"
    _write_synthetic_urdf(asset_path, chassis.joint_names)

    payload = run_audit_phase35_bipedal_readiness(
        output_dir=readiness_dir,
        bipedal_chassis_dir=chassis_dir,
        asset_paths=[asset_path],
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["real_asset_parsed"] is True
    assert payload["phase35_no_gpu_no_hardware_prepared"] is True
    assert payload["hardware_calibrated_limits"] is False
    assert payload["unitree_sim_runtime_executed"] is False
    assert payload["hardware_executed"] is False
    assert payload["training_executed"] is False
    assert payload["promotion_eligible"] is False

    contract = load_humanoid_robot_asset_contract(
        readiness_dir / "humanoid_robot_asset_contract_v1.json"
    )
    parse_receipts = load_robot_asset_parse_receipts(
        readiness_dir / "robot_asset_parse_receipts_v1.jsonl"
    )
    kinematic = load_kinematic_consistency_report(
        readiness_dir / "kinematic_consistency_report_v1.json"
    )
    assert contract.asset_status == "assets_present_unparsed"
    assert contract.parser_status == "ready_to_parse"
    assert contract.real_asset_parsed is False
    assert parse_receipts[0].status == "parsed_local_asset"
    assert parse_receipts[0].asset_format == "urdf"
    assert parse_receipts[0].real_asset_parsed is True
    assert parse_receipts[0].hardware_calibrated_limits is False
    assert set(parse_receipts[0].extracted_joint_names) == set(chassis.joint_names)
    assert kinematic.asset_joint_alignment_status == "aligned"
    assert not kinematic.missing_joint_names
    assert not kinematic.extra_asset_joint_names

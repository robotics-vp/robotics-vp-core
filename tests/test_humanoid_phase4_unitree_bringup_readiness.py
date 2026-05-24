from __future__ import annotations

from pathlib import Path

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
from scripts.economic_world_model.prepare_phase4_unitree_bringup_readiness import (
    run_prepare_phase4_unitree_bringup_readiness,
)
from src.world_model.embodiment_actuation import load_humanoid_chassis_profile
from src.world_model.humanoid_readiness import (
    DENIED_UNITREE_BRINGUP_AUTHORITIES,
    load_unitree_asset_calibration_receipts,
    load_unitree_bringup_block_receipts,
    load_unitree_command_conformance_receipts,
    load_unitree_dependency_targets,
    load_unitree_operator_recovery_runbooks,
    load_unitree_safety_preflight_receipts,
    load_unitree_sim_hardware_evidence_ledgers,
    load_unitree_stream_contracts,
    load_unitree_timing_jitter_probe_receipts,
)


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_synthetic_urdf(path: Path, joint_names: list[str]) -> None:
    lines = ['<robot name="synthetic_g1_29dof_contract">', '  <link name="base_link"/>']
    for joint_name in [*joint_names, "imu_mount_joint", "camera_mount_joint"]:
        child_name = f"{joint_name}_link"
        joint_type = "fixed" if "mount" in joint_name else "revolute"
        lines.extend(
            [
                f'  <link name="{child_name}"/>',
                f'  <joint name="{joint_name}" type="{joint_type}">',
                '    <parent link="base_link"/>',
                f'    <child link="{child_name}"/>',
                '    <limit lower="-1.0" upper="1.0" effort="1.0" velocity="1.0"/>',
                "  </joint>",
            ]
        )
    lines.append("</robot>")
    _write(path, "\n".join(lines) + "\n")


def _fake_unitree_roots(tmp_path: Path, joint_names: list[str]) -> dict[str, str]:
    roots = tmp_path / "unitree_roots"
    sdk2 = roots / "unitree_sdk2"
    _write(sdk2 / "CMakeLists.txt", "cmake_minimum_required(VERSION 3.16)\n")
    (sdk2 / "include/unitree").mkdir(parents=True)
    (sdk2 / "lib").mkdir(parents=True)

    models = roots / "unitree_models"
    (models / "G1/29dof/usd").mkdir(parents=True)
    _write(models / "README.md", "synthetic unitree model root\n")

    rl_gym = roots / "unitree_rl_gym"
    _write_synthetic_urdf(
        rl_gym / "resources/robots/g1_description/g1_29dof.urdf",
        joint_names,
    )
    (rl_gym / "legged_gym").mkdir(parents=True)
    (rl_gym / "deploy").mkdir(parents=True)

    isaaclab = roots / "unitree_sim_isaaclab"
    (isaaclab / "tasks/g1_tasks").mkdir(parents=True)
    (isaaclab / "layeredcontrol").mkdir(parents=True)
    (isaaclab / "tools").mkdir(parents=True)

    il_lerobot = roots / "unitree_IL_lerobot"
    (il_lerobot / "unitree_lerobot/eval_robot").mkdir(parents=True)
    (il_lerobot / "unitree_lerobot/utils").mkdir(parents=True)
    _write(il_lerobot / "pyproject.toml", "[project]\nname='synthetic'\n")

    return {
        "unitree_sdk2": str(sdk2),
        "unitree_models": str(models),
        "unitree_rl_gym": str(rl_gym),
        "unitree_sim_isaaclab": str(isaaclab),
        "unitree_il_lerobot": str(il_lerobot),
        "g1pilot": str(roots / "g1pilot"),
        "unitree_ros2": str(roots / "unitree_ros2"),
        "unitree_mujoco": str(roots / "unitree_mujoco"),
    }


def test_phase4_unitree_bringup_readiness_blocks_are_receipted(tmp_path):
    phase35_dir = tmp_path / "phase35"
    bipedal_chassis_dir = tmp_path / "bipedal_chassis"
    readiness_dir = tmp_path / "phase35_bipedal_readiness"
    phase4_dir = tmp_path / "phase4"
    controller_dir = tmp_path / "phase4_downstream_controller"
    bringup_dir = tmp_path / "phase4_unitree_bringup"

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
    run_prepare_phase4_downstream_controller_scaffold(
        output_dir=controller_dir,
        phase4_dir=phase4_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=readiness_dir,
        run_dependencies_if_missing=False,
    )
    chassis = load_humanoid_chassis_profile(
        bipedal_chassis_dir / "humanoid_chassis_profile_v1.json"
    )
    roots = _fake_unitree_roots(tmp_path, chassis.joint_names)

    payload = run_prepare_phase4_unitree_bringup_readiness(
        output_dir=bringup_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=readiness_dir,
        phase4_downstream_controller_dir=controller_dir,
        local_roots=roots,
        timing_iterations=16,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["local_pre_purchase_prepared"] is True
    assert payload["block_count"] == 9
    assert payload["dependency_target_count"] == 8
    assert payload["dependency_verified_count"] == 5
    assert payload["asset_joint_subset_aligned"] is True
    assert payload["stream_contracts_present"] is True
    assert payload["command_conformance_dry_run_ready"] is True
    assert payload["local_timing_probe_present"] is True
    assert payload["physical_safety_preflight_present"] is True
    assert payload["operator_recovery_runbook_present"] is True
    assert payload["honest_sim_or_hardware_evidence_present"] is False
    assert payload["hardware_dispatch_enabled"] is False
    assert payload["ros2_publish_attempted"] is False
    assert payload["unitree_sdk2_write_enabled"] is False
    assert payload["g1pilot_runtime_invoked"] is False
    assert payload["hardware_executed"] is False
    assert payload["training_executed"] is False
    assert payload["promotion_eligible"] is False
    assert not any(
        payload["denied_gates"][key] for key in DENIED_UNITREE_BRINGUP_AUTHORITIES
    )

    block_receipts = load_unitree_bringup_block_receipts(
        bringup_dir / "unitree_bringup_block_receipts_v1.jsonl"
    )
    dependency_targets = load_unitree_dependency_targets(
        bringup_dir / "unitree_dependency_targets_v1.jsonl"
    )
    asset_receipts = load_unitree_asset_calibration_receipts(
        bringup_dir / "unitree_asset_calibration_receipts_v1.jsonl"
    )
    stream_contracts = load_unitree_stream_contracts(
        bringup_dir / "unitree_stream_contracts_v1.jsonl"
    )
    command_receipts = load_unitree_command_conformance_receipts(
        bringup_dir / "unitree_command_conformance_receipts_v1.jsonl"
    )
    timing_receipts = load_unitree_timing_jitter_probe_receipts(
        bringup_dir / "unitree_timing_jitter_probe_receipts_v1.jsonl"
    )
    safety_receipts = load_unitree_safety_preflight_receipts(
        bringup_dir / "unitree_safety_preflight_receipts_v1.jsonl"
    )
    runbooks = load_unitree_operator_recovery_runbooks(
        bringup_dir / "unitree_operator_recovery_runbooks_v1.jsonl"
    )
    ledgers = load_unitree_sim_hardware_evidence_ledgers(
        bringup_dir / "unitree_sim_hardware_evidence_ledgers_v1.jsonl"
    )

    assert {receipt.block_key for receipt in block_receipts} == {
        "runtime_dependency_manifest",
        "g1pilot_or_fallback_review",
        "robot_asset_calibration_intake",
        "live_stream_interface_contracts",
        "command_interface_conformance",
        "timing_jitter_probe",
        "physical_safety_preflight",
        "operator_estop_recovery_runbook",
        "sim_hardware_evidence_ledger",
    }
    assert all(receipt.local_prepared for receipt in block_receipts)
    assert all(receipt.external_blocked for receipt in block_receipts)
    assert {
        target.local_root_key
        for target in dependency_targets
        if target.verified_local_layout
    } >= {
        "unitree_sdk2",
        "unitree_models",
        "unitree_rl_gym",
        "unitree_sim_isaaclab",
        "unitree_il_lerobot",
    }
    assert {
        target.local_root_key for target in dependency_targets if not target.exists
    } >= {"g1pilot", "unitree_ros2", "unitree_mujoco"}
    assert asset_receipts[0].status == (
        "controlled_joint_subset_aligned_with_extra_asset_joints"
    )
    assert asset_receipts[0].hardware_calibrated_limits is False
    assert asset_receipts[0].extra_asset_joint_names
    assert all(contract.mock_receiver_ready for contract in stream_contracts)
    assert not any(contract.live_stream_observed for contract in stream_contracts)
    assert all(receipt.dry_run_frames_available for receipt in command_receipts)
    assert not any(receipt.ros2_publish_attempted for receipt in command_receipts)
    assert not any(receipt.unitree_sdk2_write_enabled for receipt in command_receipts)
    assert timing_receipts[0].local_perf_counter_probe_executed is True
    assert timing_receipts[0].dds_measured is False
    assert timing_receipts[0].hardware_measured is False
    assert all(receipt.dispatch_veto_default for receipt in safety_receipts)
    assert not any(receipt.hardware_calibrated for receipt in safety_receipts)
    assert all(runbook.local_runbook_present for runbook in runbooks)
    assert not any(runbook.drill_executed for runbook in runbooks)
    assert ledgers[0].honest_sim_executed is False
    assert ledgers[0].hardware_executed is False
    assert set(ledgers[0].local_roots_missing) >= {
        "g1pilot",
        "unitree_ros2",
        "unitree_mujoco",
    }

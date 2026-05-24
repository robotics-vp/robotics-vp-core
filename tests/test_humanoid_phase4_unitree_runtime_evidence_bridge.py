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
from scripts.economic_world_model.prepare_phase4_unitree_local_harnesses import (
    run_prepare_phase4_unitree_local_harnesses,
)
from scripts.economic_world_model.prepare_phase4_unitree_runtime_evidence_bridge import (
    run_prepare_phase4_unitree_runtime_evidence_bridge,
)
from src.world_model.humanoid_readiness import (
    DENIED_UNITREE_RUNTIME_BRIDGE_AUTHORITIES,
    load_mujoco_headless_step_receipts,
    load_mujoco_headless_trace_rows,
    load_operator_recovery_drill_receipts,
    load_operator_recovery_scenarios,
    load_ros2_runtime_readiness_receipts,
    load_safety_envelope_expansion_receipts,
    load_trace_import_adapter_receipts,
)


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_unitree_roots(tmp_path: Path) -> dict[str, str]:
    roots = tmp_path / "unitree_runtime_bridge_roots"
    ros2 = roots / "unitree_ros2"
    ros_tree = ros2 / "cyclonedds_ws/src/unitree"
    _write(ros2 / "setup.sh", "#!/usr/bin/env bash\n")
    _write(ros_tree / "unitree_hg/package.xml", "<package><name>unitree_hg</name></package>\n")
    _write(ros_tree / "unitree_api/package.xml", "<package><name>unitree_api</name></package>\n")
    _write(ros_tree / "unitree_go/package.xml", "<package><name>unitree_go</name></package>\n")
    _write(
        ros_tree / "unitree_hg/msg/LowCmd.msg",
        "\n".join(
            [
                "uint8 mode_pr",
                "uint8 mode_machine",
                "MotorCmd[35] motor_cmd",
                "uint32[4] reserve",
                "uint32 crc",
                "",
            ]
        ),
    )
    _write(
        ros_tree / "unitree_hg/msg/MotorCmd.msg",
        "\n".join(
            [
                "uint8 mode",
                "float32 q",
                "float32 dq",
                "float32 tau",
                "float32 kp",
                "float32 kd",
                "uint32[3] reserve",
                "",
            ]
        ),
    )
    _write(
        ros_tree / "unitree_hg/msg/LowState.msg",
        "\n".join(
            [
                "uint8[2] version",
                "uint8 mode_pr",
                "uint8 mode_machine",
                "uint32 tick",
                "IMUState imu_state",
                "MotorState[35] motor_state",
                "uint8[40] wireless_remote",
                "uint32[4] reserve",
                "uint32 crc",
                "",
            ]
        ),
    )
    _write(
        ros_tree / "unitree_hg/msg/IMUState.msg",
        "\n".join(
            [
                "float32[4] quaternion",
                "float32[3] gyroscope",
                "float32[3] accelerometer",
                "float32[3] rpy",
                "int8 temperature",
                "",
            ]
        ),
    )
    _write(
        ros_tree / "unitree_api/msg/Request.msg",
        "\n".join(["RequestHeader header", "string parameter", "uint8[] binary", ""]),
    )
    _write(
        ros_tree / "unitree_api/msg/RequestHeader.msg",
        "\n".join(["uint32 identity", "int64 lease_id", ""]),
    )
    _write(
        ros_tree / "unitree_go/msg/WirelessController.msg",
        "\n".join(["float32 lx", "float32 ly", "float32 rx", "float32 ry", "uint16 keys", ""]),
    )

    mujoco = roots / "unitree_mujoco"
    _write(mujoco / "simulate_python/unitree_mujoco.py", "print('no app launch')\n")
    _write(mujoco / "simulate_python/unitree_sdk2py_bridge.py", "BRIDGE = True\n")
    _write(
        mujoco / "unitree_robots/g1/scene_29dof.xml",
        '<mujoco model="synthetic_g1_scene"><worldbody /></mujoco>\n',
    )
    _write(
        mujoco / "unitree_robots/g1/g1_29dof.xml",
        '<mujoco model="synthetic_g1"><worldbody /></mujoco>\n',
    )

    g1pilot = roots / "g1pilot"
    _write(g1pilot / "package.xml", "<package><name>g1pilot</name></package>\n")
    _write(g1pilot / "launch/bringup_launcher.launch.py", "# no launch\n")
    _write(g1pilot / "launch/teleoperation_launcher.launch.py", "# no launch\n")
    _write(g1pilot / "g1pilot/__init__.py", "")
    _write(g1pilot / "description_files/urdf/g1.urdf", "<robot name='g1'/>\n")

    return {
        "unitree_ros2": str(ros2),
        "unitree_mujoco": str(mujoco),
        "g1pilot": str(g1pilot),
    }


def test_phase4_unitree_runtime_evidence_bridge_receipts(tmp_path: Path) -> None:
    phase35_dir = tmp_path / "phase35"
    bipedal_chassis_dir = tmp_path / "bipedal_chassis"
    readiness_dir = tmp_path / "phase35_bipedal_readiness"
    phase4_dir = tmp_path / "phase4"
    controller_dir = tmp_path / "phase4_downstream_controller"
    harness_dir = tmp_path / "phase4_unitree_local_harnesses"
    bridge_dir = tmp_path / "phase4_unitree_runtime_bridge"
    roots = _fake_unitree_roots(tmp_path)

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
    run_prepare_phase4_unitree_local_harnesses(
        output_dir=harness_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=readiness_dir,
        phase4_downstream_controller_dir=controller_dir,
        local_roots=roots,
        sample_count=8,
        timing_iterations=8,
        run_dependencies_if_missing=False,
    )

    payload = run_prepare_phase4_unitree_runtime_evidence_bridge(
        output_dir=bridge_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=readiness_dir,
        phase4_downstream_controller_dir=controller_dir,
        phase4_unitree_local_harness_dir=harness_dir,
        local_roots=roots,
        mujoco_steps=4,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["local_runtime_evidence_bridge_complete"] is True
    assert payload["ros2_runtime_preflight_complete"] is True
    assert payload["mujoco_headless_trace_attempt_complete"] is True
    assert payload["trace_ingestion_adapters_complete"] is True
    assert payload["safety_envelope_expansion_complete"] is True
    assert payload["operator_drill_runner_complete"] is True
    assert payload["ros2_runtime_readiness_receipt_count"] == 2
    assert payload["mujoco_headless_step_receipt_count"] == 1
    assert payload["trace_import_adapter_receipt_count"] == 3
    assert payload["safety_envelope_expansion_receipt_count"] == 5
    assert payload["operator_recovery_scenario_count"] == 4
    assert payload["operator_recovery_drill_receipt_count"] == 4
    assert payload["ros2_publish_attempted"] is False
    assert payload["unitree_sdk2_write_enabled"] is False
    assert payload["g1pilot_runtime_invoked"] is False
    assert payload["hardware_executed"] is False
    assert payload["live_policy_control"] is False
    assert payload["training_executed"] is False
    assert payload["promotion_eligible"] is False
    assert not any(
        payload["denied_gates"][key]
        for key in DENIED_UNITREE_RUNTIME_BRIDGE_AUTHORITIES
    )

    ros2 = load_ros2_runtime_readiness_receipts(
        bridge_dir / "unitree_ros2_runtime_readiness_receipts_v1.jsonl"
    )
    assert {receipt.profile_key for receipt in ros2} == {
        "native_ros2_colcon",
        "container_ros2_colcon",
    }
    assert all(receipt.setup_script_present for receipt in ros2)
    assert all(receipt.msg_definition_count >= 7 for receipt in ros2)
    assert not any(receipt.build_executed for receipt in ros2)
    assert not any(receipt.ros2_publish_attempted for receipt in ros2)

    step = load_mujoco_headless_step_receipts(
        bridge_dir / "unitree_mujoco_headless_step_receipts_v1.jsonl"
    )[0]
    rows = load_mujoco_headless_trace_rows(
        bridge_dir / "unitree_mujoco_headless_trace_rows_v1.jsonl"
    )
    assert step.unitree_mujoco_app_launched is False
    assert step.policy_controlled is False
    assert step.ros2_bridge_active is False
    assert step.hardware_executed is False
    if step.step_executed:
        assert step.status == "ok"
        assert step.trace_row_count == 4
        assert len(rows) == 4
        assert all(not row.policy_controlled for row in rows)
    else:
        assert step.status.startswith("blocked_")
        assert len(rows) == 0

    adapters = load_trace_import_adapter_receipts(
        bridge_dir / "unitree_trace_import_adapter_receipts_v1.jsonl"
    )
    by_adapter = {receipt.adapter_key: receipt for receipt in adapters}
    assert by_adapter["jsonl_unitree_trace_bundle"].import_executed is True
    assert by_adapter["jsonl_unitree_trace_bundle"].rows_imported >= 1
    assert by_adapter["rosbag2_unitree_topics"].import_executed is False
    assert by_adapter["mcap_unitree_topics"].import_executed is False
    assert "rosbag2_input_path_missing" in by_adapter["rosbag2_unitree_topics"].blockers
    assert "mcap_input_path_missing" in by_adapter["mcap_unitree_topics"].blockers

    safety = load_safety_envelope_expansion_receipts(
        bridge_dir / "unitree_safety_envelope_expansion_receipts_v1.jsonl"
    )
    assert {receipt.envelope_key for receipt in safety} == {
        "joint_limit_runtime_clamp",
        "self_collision_hook",
        "fall_posture_guard",
        "stop_distance_slot",
        "calibrated_limit_sidecar",
    }
    assert all(receipt.dispatch_veto_default for receipt in safety)
    assert not any(receipt.calibrated_from_hardware for receipt in safety)
    assert not any(receipt.hardware_executed for receipt in safety)

    scenarios = load_operator_recovery_scenarios(
        bridge_dir / "unitree_operator_recovery_scenarios_v1.jsonl"
    )
    drills = load_operator_recovery_drill_receipts(
        bridge_dir / "unitree_operator_recovery_drill_receipts_v1.jsonl"
    )
    assert len(scenarios) == 4
    assert all(scenario.replay_export_required for scenario in scenarios)
    assert all(receipt.local_drill_executed for receipt in drills)
    assert all(receipt.passed for receipt in drills)
    assert not any(receipt.teleop_runtime_executed for receipt in drills)
    assert not any(receipt.command_dispatch_allowed for receipt in drills)
    assert not any(receipt.hardware_executed for receipt in drills)

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
from src.world_model.humanoid_readiness import (
    DENIED_UNITREE_LOCAL_HARNESS_AUTHORITIES,
    load_command_shape_validation_receipts,
    load_contact_traces,
    load_imu_traces,
    load_low_state_traces,
    load_mock_receiver_receipts,
    load_mock_timing_run_receipts,
    load_ros_message_definitions,
    load_runtime_preflight_receipts,
    load_safety_state_transitions,
    load_stale_data_validation_receipts,
    load_synthetic_safety_drill_receipts,
    load_trace_replay_receipts,
    load_watchdog_demotion_receipts,
    load_wireless_estop_traces,
)


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_unitree_local_harness_roots(tmp_path: Path) -> dict[str, str]:
    roots = tmp_path / "unitree_local_harness_roots"
    ros2 = roots / "unitree_ros2"
    ros_tree = ros2 / "cyclonedds_ws/src/unitree"
    _write(ros2 / "setup.sh", "#!/usr/bin/env bash\n")
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
        "\n".join(
            [
                "RequestHeader header",
                "string parameter",
                "uint8[] binary",
                "",
            ]
        ),
    )
    _write(
        ros_tree / "unitree_api/msg/RequestHeader.msg",
        "\n".join(
            [
                "uint32 identity",
                "int64 lease_id",
                "",
            ]
        ),
    )
    _write(
        ros_tree / "unitree_go/msg/WirelessController.msg",
        "\n".join(
            [
                "float32 lx",
                "float32 ly",
                "float32 rx",
                "float32 ry",
                "uint16 keys",
                "",
            ]
        ),
    )

    mujoco = roots / "unitree_mujoco"
    _write(mujoco / "simulate_python/unitree_mujoco.py", "print('no launch')\n")
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


def test_phase4_unitree_local_harnesses_materialize_receipts(tmp_path: Path) -> None:
    phase35_dir = tmp_path / "phase35"
    bipedal_chassis_dir = tmp_path / "bipedal_chassis"
    readiness_dir = tmp_path / "phase35_bipedal_readiness"
    phase4_dir = tmp_path / "phase4"
    controller_dir = tmp_path / "phase4_downstream_controller"
    harness_dir = tmp_path / "phase4_unitree_local_harnesses"

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

    payload = run_prepare_phase4_unitree_local_harnesses(
        output_dir=harness_dir,
        bipedal_chassis_dir=bipedal_chassis_dir,
        phase35_bipedal_readiness_dir=readiness_dir,
        phase4_downstream_controller_dir=controller_dir,
        local_roots=_fake_unitree_local_harness_roots(tmp_path),
        sample_count=8,
        timing_iterations=8,
        run_dependencies_if_missing=False,
    )

    assert payload["status"] == "ok"
    assert payload["local_harnesses_complete"] is True
    assert payload["trace_stream_harness_complete"] is True
    assert payload["command_shape_harness_complete"] is True
    assert payload["mock_timing_watchdog_harness_complete"] is True
    assert payload["safety_recovery_harness_complete"] is True
    assert payload["runtime_preflight_harness_complete"] is True
    assert payload["low_state_trace_count"] == 8
    assert payload["imu_trace_count"] == 8
    assert payload["wireless_estop_trace_count"] == 8
    assert payload["contact_trace_count"] == 8
    assert payload["trace_replay_receipt_count"] == 4
    assert payload["mock_receiver_receipt_count"] == 4
    assert payload["stale_validation_receipt_count"] == 4
    assert payload["ros_message_definition_count"] == 7
    assert payload["command_shape_validation_receipt_count"] == 2
    assert payload["mock_timing_run_receipt_count"] == 1
    assert payload["watchdog_demotion_receipt_count"] == 1
    assert payload["synthetic_safety_drill_receipt_count"] == 1
    assert payload["runtime_preflight_receipt_count"] == 7
    assert payload["live_stream_observed"] is False
    assert payload["ros2_publish_attempted"] is False
    assert payload["unitree_sdk2_write_enabled"] is False
    assert payload["g1pilot_runtime_invoked"] is False
    assert payload["mujoco_launch_executed"] is False
    assert payload["ros2_launch_executed"] is False
    assert payload["hardware_executed"] is False
    assert payload["training_executed"] is False
    assert payload["promotion_eligible"] is False
    assert not any(
        payload["denied_gates"][key]
        for key in DENIED_UNITREE_LOCAL_HARNESS_AUTHORITIES
    )

    assert len(load_low_state_traces(harness_dir / "unitree_low_state_traces_v1.jsonl")) == 8
    assert len(load_imu_traces(harness_dir / "unitree_imu_traces_v1.jsonl")) == 8
    assert len(load_wireless_estop_traces(harness_dir / "unitree_wireless_estop_traces_v1.jsonl")) == 8
    assert len(load_contact_traces(harness_dir / "unitree_contact_traces_v1.jsonl")) == 8

    replay = load_trace_replay_receipts(
        harness_dir / "unitree_trace_replay_receipts_v1.jsonl"
    )
    receivers = load_mock_receiver_receipts(
        harness_dir / "unitree_mock_receiver_receipts_v1.jsonl"
    )
    stale = load_stale_data_validation_receipts(
        harness_dir / "unitree_stale_data_validation_receipts_v1.jsonl"
    )
    assert all(receipt.jsonl_import_verified for receipt in replay)
    assert all(receipt.rosbag_import_ready for receipt in replay)
    assert not any(receipt.rosbag_import_executed for receipt in replay)
    assert all(receipt.receiver_executed for receipt in receivers)
    assert not any(receipt.live_stream_observed for receipt in receivers)
    assert any(receipt.stale_data_veto_required for receipt in stale)

    definitions = load_ros_message_definitions(
        harness_dir / "unitree_ros_message_definitions_v1.jsonl"
    )
    assert {definition.message_name for definition in definitions} >= {
        "LowCmd",
        "MotorCmd",
        "LowState",
        "IMUState",
        "Request",
        "RequestHeader",
        "WirelessController",
    }
    assert all(definition.parsed for definition in definitions)

    command_shapes = load_command_shape_validation_receipts(
        harness_dir / "unitree_command_shape_validation_receipts_v1.jsonl"
    )
    assert {receipt.command_family for receipt in command_shapes} == {
        "low_level_joint_pd",
        "sport_request_degraded_mode",
    }
    assert all(receipt.no_publish_serialization_ready for receipt in command_shapes)
    assert not any(receipt.ros2_publish_attempted for receipt in command_shapes)
    assert not any(receipt.unitree_sdk2_write_enabled for receipt in command_shapes)

    timing = load_mock_timing_run_receipts(
        harness_dir / "unitree_mock_timing_run_receipts_v1.jsonl"
    )
    watchdog = load_watchdog_demotion_receipts(
        harness_dir / "unitree_watchdog_demotion_receipts_v1.jsonl"
    )
    assert timing[0].local_loop_executed is True
    assert timing[0].dds_runtime_observed is False
    assert timing[0].producer_event_count == 8
    assert watchdog[0].demotion_requested is True
    assert watchdog[0].command_dispatch_allowed is False

    transitions = load_safety_state_transitions(
        harness_dir / "unitree_safety_state_transitions_v1.jsonl"
    )
    drills = load_synthetic_safety_drill_receipts(
        harness_dir / "unitree_synthetic_safety_drill_receipts_v1.jsonl"
    )
    assert {transition.to_state for transition in transitions} >= {
        "estop_latched",
        "stable_base_demote_requested",
        "recovery_ready_operator_required",
    }
    assert drills[0].drill_executed_locally is True
    assert drills[0].estop_latched is True
    assert drills[0].stale_data_vetoed is True
    assert drills[0].joint_clamp_observed is True
    assert drills[0].stable_base_demote_requested is True
    assert drills[0].recovery_state_reached is True
    assert drills[0].hardware_executed is False

    preflights = load_runtime_preflight_receipts(
        harness_dir / "unitree_runtime_preflight_receipts_v1.jsonl"
    )
    assert {receipt.target_key for receipt in preflights} >= {
        "unitree_ros2",
        "unitree_mujoco",
        "g1pilot",
    }
    assert any(
        receipt.target_key == "unitree_ros2"
        and receipt.preflight_kind == "source_layout_and_message_files"
        and receipt.status == "ok"
        for receipt in preflights
    )
    assert any(
        receipt.target_key == "unitree_mujoco"
        and receipt.preflight_kind == "source_layout_and_xml_parse"
        and receipt.status == "ok"
        for receipt in preflights
    )
    assert all(receipt.launch_executed is False for receipt in preflights)
    assert all(receipt.runtime_executed is False for receipt in preflights)

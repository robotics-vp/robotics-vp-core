from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.run_cpu_august_gap_tranche import (
    run_cpu_august_gap_tranche,
)
from src.world_model.humanoid_readiness import (
    DENIED_CPU_AUGUST_GAP_AUTHORITIES,
    load_cpu_august_gap_execution_report,
    load_unitree_event_replay_join_rows,
    load_unitree_lower_wm_ingestion_rows,
    load_unitree_ros2_sdk2_build_message_validation_receipts,
)


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_unitree_roots(tmp_path: Path) -> dict[str, str]:
    roots = tmp_path / "unitree_cpu_august_roots"
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
    for name in ("scene_29dof.xml", "g1_29dof.xml", "scene_23dof.xml", "g1_23dof.xml", "scene.xml"):
        _write(
            mujoco / f"unitree_robots/g1/{name}",
            f'<mujoco model="{name.replace(".", "_")}"><worldbody /></mujoco>\n',
        )

    g1pilot = roots / "g1pilot"
    _write(g1pilot / "package.xml", "<package><name>g1pilot</name></package>\n")
    _write(g1pilot / "launch/bringup_launcher.launch.py", "# no launch\n")
    _write(g1pilot / "launch/teleoperation_launcher.launch.py", "# no launch\n")
    _write(g1pilot / "g1pilot/__init__.py", "")
    _write(g1pilot / "description_files/urdf/g1.urdf", "<robot name='g1'/>\n")

    sdk2 = roots / "unitree_sdk2"
    _write(sdk2 / "include/.keep", "")
    _write(sdk2 / "thirdparty/include/.keep", "")

    rl_gym = roots / "unitree_rl_gym"
    _write(rl_gym / "deploy/pre_train/g1/motion.pt", "not a real checkpoint\n")
    _write(rl_gym / "deploy/deploy_mujoco/configs/g1.yaml", "robot: g1\n")
    _write(rl_gym / "resources/robots/g1_description/g1.urdf", "<robot name='g1'/>\n")

    isaaclab = roots / "unitree_sim_isaaclab"
    _write(isaaclab / "source/unitree_isaac/tasks/g1_task.py", "TASK = 'g1'\n")

    lerobot = roots / "unitree_IL_lerobot"
    _write(lerobot / "eval_unitree.py", "print('no eval')\n")
    _write(lerobot / "convert_unitree_data.py", "print('no conversion')\n")

    return {
        "unitree_ros2": str(ros2),
        "unitree_mujoco": str(mujoco),
        "g1pilot": str(g1pilot),
        "unitree_sdk2": str(sdk2),
        "unitree_rl_gym": str(rl_gym),
        "unitree_sim_isaaclab": str(isaaclab),
        "unitree_il_lerobot": str(lerobot),
    }


def test_cpu_august_gap_tranche_materializes_joined_receipts(tmp_path: Path) -> None:
    output = tmp_path / "cpu_august_gap"
    payload = run_cpu_august_gap_tranche(
        output_dir=output,
        bipedal_chassis_dir=tmp_path / "bipedal_chassis",
        phase35_bipedal_readiness_dir=tmp_path / "phase35_readiness",
        phase4_downstream_controller_dir=tmp_path / "phase4_controller",
        phase4_unitree_local_harness_dir=tmp_path / "phase4_local_harness",
        phase4_unitree_runtime_bridge_dir=tmp_path / "phase4_runtime_bridge",
        phase4_unitree_blocker_stress_probe_dir=tmp_path / "phase4_blockers",
        local_roots=_fake_unitree_roots(tmp_path),
        sample_count=4,
        timing_iterations=4,
        mujoco_steps=2,
        stress_steps=2,
        allow_build_attempt=False,
        run_dependencies_if_missing=True,
    )

    assert payload["status"] == "ok"
    assert payload["cpu_august_gap_tranche_complete"] is True
    assert payload["ros2_sdk2_build_message_validation_complete"] is True
    assert payload["trace_import_complete"] is True
    assert payload["command_dry_run_complete"] is True
    assert payload["timing_watchdog_complete"] is True
    assert payload["safety_recovery_complete"] is True
    assert payload["cpu_mujoco_probe_complete"] is True
    assert payload["event_spine_replay_joins_complete"] is True
    assert payload["lower_wm_ingestion_complete"] is True
    assert payload["validation_receipt_count"] == 5
    assert payload["replay_step_count"] == 4
    assert payload["lower_wm_ingestion_row_count"] == 4
    assert payload["ros2_publish_attempted"] is False
    assert payload["unitree_sdk2_write_enabled"] is False
    assert payload["hardware_executed"] is False
    assert payload["training_executed"] is False
    assert payload["phase7_authority_granted"] is False
    assert payload["promotion_eligible"] is False
    assert not any(payload["denied_gates"][key] for key in DENIED_CPU_AUGUST_GAP_AUTHORITIES)

    report = load_cpu_august_gap_execution_report(
        output / "cpu_august_gap_execution_report_v1.json"
    )
    assert report.cpu_august_gap_tranche_complete is True

    validation = load_unitree_ros2_sdk2_build_message_validation_receipts(
        output / "unitree_ros2_sdk2_build_message_validation_receipts_v1.jsonl"
    )
    by_key = {receipt.validation_key: receipt for receipt in validation}
    assert by_key["ros2_static_message_definition_validation"].succeeded is True
    assert by_key["ros2_colcon_build_validation"].build_attempted is False
    assert by_key["unitree_sdk2_cmake_build_validation"].build_attempted is False

    joins = load_unitree_event_replay_join_rows(
        output / "unitree_event_replay_join_rows_v1.jsonl"
    )
    assert {row.join_key for row in joins} >= {
        "build_message_validation",
        "trace_import",
        "command_dry_run",
        "timing_watchdog",
        "safety_recovery",
        "cpu_mujoco_probe",
    }
    assert all(row.join_status == "ok_shadow_join" for row in joins)

    ingestion = load_unitree_lower_wm_ingestion_rows(
        output / "unitree_lower_wm_ingestion_rows_v1.jsonl"
    )
    assert {row.wm_key for row in ingestion} == {
        "embodiment_actuation",
        "sim_synth_physics",
        "perception_grounding",
        "economic_world_model",
    }
    assert all(row.ready_for_economic_shadow for row in ingestion)
    assert not any(row.promotion_eligible for row in ingestion)

    event_spine = json.loads((output / "event_spine.json").read_text(encoding="utf-8"))
    replay_steps = (output / "unitree_replay_steps_v1.jsonl").read_text(encoding="utf-8")
    assert event_spine["event_count"] >= 8
    assert len([line for line in replay_steps.splitlines() if line.strip()]) == 4

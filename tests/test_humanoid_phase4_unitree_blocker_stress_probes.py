from __future__ import annotations

from pathlib import Path

from scripts.economic_world_model.probe_phase4_unitree_blockers import (
    run_probe_phase4_unitree_blockers,
)
from src.world_model.humanoid_readiness import (
    BLOCKER_PROBE_KEYS,
    DENIED_UNITREE_BLOCKER_STRESS_AUTHORITIES,
    load_phase4_unitree_blocker_stress_probe_report,
    load_unitree_blocker_stress_probe_receipts,
    load_unitree_mujoco_model_stress_receipts,
)


def _write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_roots(tmp_path: Path) -> dict[str, str]:
    roots = tmp_path / "unitree_blocker_probe_roots"

    ros2 = roots / "unitree_ros2"
    ros_tree = ros2 / "cyclonedds_ws/src/unitree"
    for pkg in ("unitree_hg", "unitree_go", "unitree_api"):
        _write(
            ros_tree / pkg / "package.xml",
            f"<package><name>{pkg}</name></package>\n",
        )
        _write(ros_tree / pkg / "CMakeLists.txt", "cmake_minimum_required(VERSION 3.16)\n")
    _write(ros_tree / "unitree_hg/msg/LowCmd.msg", "uint8 mode\n")
    _write(ros_tree / "unitree_go/msg/WirelessController.msg", "uint16 keys\n")
    _write(ros_tree / "unitree_api/msg/Request.msg", "string parameter\n")

    mujoco = roots / "unitree_mujoco"
    for name in (
        "scene_29dof.xml",
        "g1_29dof.xml",
        "scene_23dof.xml",
        "g1_23dof.xml",
        "scene.xml",
    ):
        _write(
            mujoco / "unitree_robots/g1" / name,
            '<mujoco model="synthetic_g1"><worldbody /></mujoco>\n',
        )

    g1pilot = roots / "g1pilot"
    _write(g1pilot / "g1pilot/__init__.py", "")
    _write(g1pilot / "launch/teleoperation_launcher.launch.py", "def generate_launch_description():\n    return None\n")
    _write(g1pilot / "launch/bringup_launcher.launch.py", "def generate_launch_description():\n    return None\n")
    _write(g1pilot / "setup.py", "from setuptools import setup\nsetup(name='g1pilot')\n")

    sdk2 = roots / "unitree_sdk2"
    (sdk2 / "include/unitree").mkdir(parents=True)
    (sdk2 / "thirdparty/include/dds").mkdir(parents=True)
    _write(sdk2 / "thirdparty/include/dds/dds.h", "#pragma once\n")
    _write(
        sdk2 / "thirdparty/include/dds/version.h",
        "#pragma once\n#define DDS_VERSION \"test\"\n#define DDS_VERSION_MAJOR 0\n",
    )

    rl_gym = roots / "unitree_rl_gym"
    _write(rl_gym / "deploy/pre_train/g1/motion.pt", "synthetic checkpoint placeholder\n")
    _write(rl_gym / "deploy/deploy_mujoco/configs/g1.yaml", "robot: g1\n")
    _write(rl_gym / "resources/robots/g1_description/g1_29dof.urdf", "<robot name='g1'/>\n")
    _write(rl_gym / "resources/robots/g1_description/g1_29dof.xml", "<mujoco/>\n")

    isaaclab = roots / "unitree_sim_isaaclab"
    _write(isaaclab / "tasks/g1_tasks/pick_place/__init__.py", "")
    _write(isaaclab / "requirements.txt", "isaaclab\n")

    lerobot = roots / "unitree_IL_lerobot"
    for path in (
        "unitree_lerobot/eval_robot/eval_g1.py",
        "unitree_lerobot/eval_robot/eval_g1_sim.py",
        "unitree_lerobot/utils/convert_unitree_json_to_lerobot.py",
        "unitree_lerobot/utils/convert_unitree_json_to_h5.py",
    ):
        _write(lerobot / path, "")

    return {
        "unitree_ros2": str(ros2),
        "unitree_mujoco": str(mujoco),
        "g1pilot": str(g1pilot),
        "unitree_sdk2": str(sdk2),
        "unitree_rl_gym": str(rl_gym),
        "unitree_sim_isaaclab": str(isaaclab),
        "unitree_il_lerobot": str(lerobot),
    }


def test_phase4_unitree_blocker_stress_probe_receipts(tmp_path: Path) -> None:
    output = tmp_path / "phase4_unitree_blocker_stress_probes"
    payload = run_probe_phase4_unitree_blockers(
        output_dir=output,
        local_roots=_fake_roots(tmp_path),
        stress_steps=3,
    )

    assert payload["status"] == "ok"
    assert payload["all_local_probe_attempts_complete"] is True
    assert payload["local_phase4_probe_expansion_complete"] is True
    assert payload["probe_receipt_count"] == len(BLOCKER_PROBE_KEYS)
    assert payload["mujoco_model_stress_receipt_count"] == 5
    assert payload["g1pilot_static_surface_succeeded"] is True
    assert payload["policy_checkpoint_visible"] is True
    assert payload["isaaclab_task_surface_visible"] is True
    assert payload["lerobot_adapter_surface_visible"] is True
    assert not any(
        payload["denied_gates"][key]
        for key in DENIED_UNITREE_BLOCKER_STRESS_AUTHORITIES
    )
    assert payload["ros2_publish_attempted"] is False
    assert payload["unitree_sdk2_write_enabled"] is False
    assert payload["g1pilot_runtime_invoked"] is False
    assert payload["hardware_executed"] is False
    assert payload["live_policy_control"] is False
    assert payload["training_executed"] is False
    assert payload["promotion_eligible"] is False

    report = load_phase4_unitree_blocker_stress_probe_report(
        output / "phase4_unitree_blocker_stress_probe_report_v1.json"
    )
    receipts = load_unitree_blocker_stress_probe_receipts(
        output / "unitree_blocker_stress_probe_receipts_v1.jsonl"
    )
    mujoco = load_unitree_mujoco_model_stress_receipts(
        output / "unitree_mujoco_model_stress_receipts_v1.jsonl"
    )
    assert report.report_id == payload["report_id"]
    assert {receipt.probe_key for receipt in receipts} == set(BLOCKER_PROBE_KEYS)
    assert any(
        receipt.probe_key == "host_ros2_colcon_toolchain"
        and "ros2_runtime_not_installed" in receipt.blockers
        for receipt in receipts
    )
    assert any(
        receipt.probe_key == "unitree_mujoco_g1_model_stress"
        for receipt in receipts
    )
    assert all(receipt.policy_controlled is False for receipt in mujoco)
    assert all(receipt.ros2_bridge_active is False for receipt in mujoco)
    assert all(receipt.hardware_executed is False for receipt in mujoco)

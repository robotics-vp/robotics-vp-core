from __future__ import annotations

import json
from pathlib import Path

from scripts.scan_phase1_runtime_layouts import main as scan_runtime_layouts_main


def test_scan_phase1_runtime_layouts_emits_deployment_and_runtime_packs(tmp_path: Path) -> None:
    unitree_sim_root = tmp_path / "unitree_sim_isaaclab"
    unitree_sim_root.mkdir()
    (unitree_sim_root / "sim_main.py").write_text("", encoding="utf-8")
    (unitree_sim_root / "dds").mkdir()
    (unitree_sim_root / "action_provider").mkdir()
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "g1_policy.onnx").write_text("x", encoding="utf-8")
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    (motion_root / "g1_walk.npz").write_text("x", encoding="utf-8")
    retargeting_root = tmp_path / "retargeting"
    retargeting_root.mkdir()
    (retargeting_root / "g1_retarget.yaml").write_text("{}", encoding="utf-8")

    embodiment_context_path = tmp_path / "embodiment.json"
    embodiment_context_path.write_text(
        json.dumps(
            {
                "unitree_sim_isaaclab_root": str(unitree_sim_root),
                "unitree_sdk2_root": str(sdk_root),
                "unitree_asset_root": str(asset_root),
                "unitree_policy_root": str(policy_root),
                "holosoma_root": str(holosoma_root),
                "holosoma_motion_root": str(motion_root),
                "holosoma_policy_root": str(policy_root),
                "retargeting_root": str(retargeting_root),
                "motion_clip_paths": [str(motion_root / "g1_walk.npz")],
                "robot_asset_manifest": {
                    "unitree_urdf": "/assets/unitree/g1.urdf",
                    "joint_map": "/assets/unitree/joint_map.yaml",
                    "camera_extrinsics": "/assets/unitree/camera.json",
                    "imu_extrinsics": "/assets/unitree/imu.json",
                    "force_torque_calibration": "/assets/unitree/ft.json",
                    "actuator_latency_profile": "/assets/unitree/latency.yaml",
                    "joint_limit_profile": "/assets/unitree/joint_limits.yaml",
                    "safety_watchdog_profile": "/assets/unitree/watchdog.yaml",
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "scan.json"

    result = scan_runtime_layouts_main(
        [
            "--embodiment-context",
            str(embodiment_context_path),
            "--output-path",
            str(output_path),
        ]
    )

    assert Path(result["output_path"]).exists()
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["isaac_deployment_contract"]["sim_launch_ready"] is True
    assert summary["isaac_upstream_runtime_pack"]["pack_status"] == "pack_ready"
    assert summary["holosoma_deployment_contract"]["motion_train_ready"] is True
    assert summary["holosoma_upstream_runtime_pack"]["pack_status"] == "pack_ready"

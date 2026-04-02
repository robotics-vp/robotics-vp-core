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
    (sdk_root / "include").mkdir()
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "g1.usd").write_text("x", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "g1_policy.onnx").write_text("x", encoding="utf-8")
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    (holosoma_root / "holosoma").mkdir()
    (holosoma_root / "holosoma" / "__init__.py").write_text("", encoding="utf-8")
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
    assert summary["isaac_upstream_runtime_pack"]["primary_policy_ref"].endswith("g1_policy.onnx")
    assert summary["isaac_upstream_runtime_pack"]["profile_candidate_counts"]["deploy"] >= 1
    assert (
        summary["isaac_upstream_runtime_pack"]["profile_install_preflight_status"]
        == "install_ready"
    )
    assert summary["isaac_runtime_binding"]["binding_status"] == "binding_ready"
    assert summary["isaac_runtime_binding"]["host_preflight_status"] == "preflight_blocked"
    assert "asset::unitree_robot_description" not in summary["isaac_runtime_binding"][
        "host_preflight_missing_components"
    ]
    assert "asset::whole_body_joint_map" not in summary["isaac_runtime_binding"][
        "host_preflight_missing_components"
    ]
    assert "asset::joint_limit_profile" not in summary["isaac_runtime_binding"][
        "host_preflight_missing_components"
    ]
    assert "asset::actuator_latency_profile" in summary["isaac_runtime_binding"][
        "host_preflight_missing_components"
    ]
    assert "unitree_sim_isaaclab" in summary["scan_summary"]["isaac"]["usable_profiles"]
    assert "unitree_sim_isaaclab" in summary["scan_summary"]["isaac"]["install_ready_profiles"]
    assert summary["scan_summary"]["isaac"]["selected_policy_ref"].endswith(
        "g1_policy.onnx"
    )
    assert summary["scan_summary"]["isaac"]["selected_deploy_config_ref"] == ""
    assert summary["holosoma_deployment_contract"]["motion_train_ready"] is True
    assert summary["holosoma_upstream_runtime_pack"]["pack_status"] == "pack_ready"
    assert summary["holosoma_upstream_runtime_pack"]["existing_motion_sources"]
    assert (
        summary["holosoma_upstream_runtime_pack"]["profile_install_preflight_status"]
        == "install_ready"
    )
    assert summary["holosoma_runtime_binding"]["binding_status"] == "binding_ready"
    assert summary["holosoma_runtime_binding"]["host_preflight_status"] == "preflight_ready"
    assert "holosoma_repo" in summary["scan_summary"]["holosoma"]["usable_profiles"]
    assert "holosoma_motion_bank" in summary["scan_summary"]["holosoma"]["usable_profiles"]
    assert summary["scan_summary"]["holosoma"]["selected_policy_ref"].endswith(
        "g1_policy.onnx"
    )


def test_scan_phase1_runtime_layouts_derives_holosoma_local_surfaces_from_repo(
    tmp_path: Path,
) -> None:
    holosoma_root = tmp_path / "holosoma"
    motion_root = holosoma_root / "src" / "holosoma" / "holosoma" / "data" / "motions"
    policy_root = holosoma_root / "src" / "holosoma_inference" / "holosoma_inference" / "models"
    retargeting_root = holosoma_root / "src" / "holosoma_retargeting"
    motion_root.mkdir(parents=True)
    policy_root.mkdir(parents=True)
    retargeting_root.mkdir(parents=True)
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    (holosoma_root / "scripts").mkdir()
    (motion_root / "g1_walk.npz").write_text("x", encoding="utf-8")
    (policy_root / "g1_policy.onnx").write_text("x", encoding="utf-8")
    (retargeting_root / "g1_retarget.json").write_text("{}", encoding="utf-8")

    embodiment_context_path = tmp_path / "embodiment_holosoma.json"
    embodiment_context_path.write_text(
        json.dumps({"holosoma_root": str(holosoma_root)}),
        encoding="utf-8",
    )
    output_path = tmp_path / "holosoma_scan.json"

    scan_runtime_layouts_main(
        [
            "--embodiment-context",
            str(embodiment_context_path),
            "--output-path",
            str(output_path),
        ]
    )

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    holosoma_summary = summary["scan_summary"]["holosoma"]
    assert "holosoma_motion_bank" in holosoma_summary["usable_profiles"]
    assert holosoma_summary["selected_policy_ref"].endswith("g1_policy.onnx")
    assert "holosoma_motion_root" in holosoma_summary["selected_verified_target_ids"]


def test_scan_phase1_runtime_layouts_derives_unitree_assets_from_public_runtime_roots(
    tmp_path: Path,
) -> None:
    unitree_sim_root = tmp_path / "unitree_sim_isaaclab"
    (unitree_sim_root / "dds").mkdir(parents=True)
    (unitree_sim_root / "action_provider").mkdir()
    (unitree_sim_root / "sim_main.py").write_text(
        'parser.add_argument("--step_hz", type=int, default=100, help="control frequency")\n',
        encoding="utf-8",
    )

    sdk_root = tmp_path / "sdk2"
    (sdk_root / "include").mkdir(parents=True)

    unitree_models_root = tmp_path / "unitree_models"
    unitree_usd = (
        unitree_models_root
        / "G1"
        / "29dof"
        / "usd"
        / "g1_29dof_rev_1_0"
        / "g1_29dof_rev_1_0.usd"
    )
    unitree_usd.parent.mkdir(parents=True)
    unitree_usd.write_text("#usda 1.0\n", encoding="utf-8")

    unitree_rl_root = tmp_path / "unitree_rl_gym"
    policy_ref = unitree_rl_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    policy_ref.parent.mkdir(parents=True)
    policy_ref.write_text("x", encoding="utf-8")
    runtime_report = unitree_rl_root / "deploy" / "deploy_mujoco" / "configs" / "g1.yaml"
    runtime_report.parent.mkdir(parents=True)
    runtime_report.write_text("task: g1\n", encoding="utf-8")
    urdf_ref = unitree_rl_root / "resources" / "robots" / "g1_description" / "g1_29dof.urdf"
    urdf_ref.parent.mkdir(parents=True)
    urdf_ref.write_text(
        '<robot name="g1"><joint name="left_hip_pitch_joint" type="revolute"><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>\n',
        encoding="utf-8",
    )

    humanoidverse_root = tmp_path / "HumanoidVerse"
    joint_config = humanoidverse_root / "humanoidverse" / "config" / "robot" / "g1" / "g1_29dof.yaml"
    joint_config.parent.mkdir(parents=True)
    joint_config.write_text(
        "robot:\n"
        "  dof_names: [left_hip_pitch_joint]\n"
        "  dof_pos_lower_limit_list: [-1.0]\n"
        "  dof_pos_upper_limit_list: [1.0]\n",
        encoding="utf-8",
    )

    embodiment_context_path = tmp_path / "embodiment_isaac.json"
    embodiment_context_path.write_text(
        json.dumps(
            {
                "unitree_sim_isaaclab_root": str(unitree_sim_root),
                "unitree_sdk2_root": str(sdk_root),
                "unitree_asset_root": str(unitree_models_root),
                "unitree_rl_gym_root": str(unitree_rl_root),
                "humanoidverse_root": str(humanoidverse_root),
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "isaac_scan.json"

    scan_runtime_layouts_main(
        [
            "--embodiment-context",
            str(embodiment_context_path),
            "--output-path",
            str(output_path),
        ]
    )

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    isaac_summary = summary["scan_summary"]["isaac"]
    assert isaac_summary["selected_policy_ref"] == str(policy_ref)
    assert "asset::unitree_robot_description" not in isaac_summary["host_preflight_missing_components"]
    assert "asset::whole_body_joint_map" not in isaac_summary["host_preflight_missing_components"]
    assert "asset::joint_limit_profile" not in isaac_summary["host_preflight_missing_components"]
    assert "asset::actuator_latency_profile" in isaac_summary["host_preflight_missing_components"]
    assert "asset::safety_watchdog_profile" in isaac_summary["host_preflight_missing_components"]

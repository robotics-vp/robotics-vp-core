from __future__ import annotations

from src.world_model.sim_synth_physics.asset_manifest import normalize_robot_asset_manifest


def test_normalize_robot_asset_manifest_derives_unitree_assets_from_runtime_targets(
    tmp_path,
) -> None:
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

    unitree_sim_root = tmp_path / "unitree_sim_isaaclab"
    sim_main = unitree_sim_root / "sim_main.py"
    sim_main.parent.mkdir(parents=True)
    sim_main.write_text(
        'parser.add_argument("--step_hz", type=int, default=100, help="control frequency")\n',
        encoding="utf-8",
    )

    xr_root = tmp_path / "xr_teleoperate"
    teleop = xr_root / "teleop" / "teleop_hand_and_arm.py"
    teleop.parent.mkdir(parents=True)
    teleop.write_text(
        "# soft emergency stop function\n# switch to damping mode\n",
        encoding="utf-8",
    )

    normalized = normalize_robot_asset_manifest(
        {},
        runtime_target_contract={
            "targets": [
                {"target_id": "unitree_asset_root", "ref": str(unitree_models_root)},
                {"target_id": "humanoidverse_root", "ref": str(humanoidverse_root)},
                {"target_id": "unitree_sim_isaaclab_root", "ref": str(unitree_sim_root)},
                {"target_id": "xr_teleoperate_root", "ref": str(xr_root)},
            ]
        },
    )

    assert normalized["unitree_robot_description"]["present"] is True
    assert normalized["unitree_robot_description"]["value"] == str(unitree_usd)
    assert normalized["unitree_robot_description"]["local_path_exists"] is True
    assert "unitree_asset_root" in normalized["unitree_robot_description"]["derivation_source"]
    assert normalized["whole_body_joint_map"]["value"] == str(joint_config)
    assert normalized["joint_limit_profile"]["value"] == str(joint_config)
    assert normalized["control_frequency_profile"]["value"] == str(sim_main)
    assert normalized["teleop_recovery_contract"]["value"] == str(teleop)
    assert normalized["actuator_latency_profile"]["present"] is False
    assert normalized["safety_watchdog_profile"]["present"] is False


def test_normalize_robot_asset_manifest_prefers_verified_derived_assets_over_missing_declared_paths(
    tmp_path,
) -> None:
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

    normalized = normalize_robot_asset_manifest(
        {
            "robot_asset_manifest": {
                "joint_map": "/missing/joint_map.yaml",
                "joint_limit_profile": "/missing/joint_limits.yaml",
            }
        },
        runtime_target_contract={
            "targets": [
                {"target_id": "humanoidverse_root", "ref": str(humanoidverse_root)},
            ]
        },
    )

    assert normalized["whole_body_joint_map"]["value"] == str(joint_config)
    assert normalized["whole_body_joint_map"]["explicit_declared_value"] == "/missing/joint_map.yaml"
    assert (
        normalized["whole_body_joint_map"]["selection_reason"]
        == "derived_local_asset_overrides_missing_declared_path"
    )
    assert normalized["joint_limit_profile"]["value"] == str(joint_config)
    assert normalized["joint_limit_profile"]["explicit_declared_value"] == "/missing/joint_limits.yaml"

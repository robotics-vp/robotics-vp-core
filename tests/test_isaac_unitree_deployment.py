from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.isaac_unitree_deployment import (
    build_isaac_unitree_deployment_contract,
)
from src.world_model.sim_synth_physics.runtime_layouts import (
    describe_isaac_policy_contract,
    describe_isaac_runtime_layouts,
)
from src.world_model.sim_synth_physics.runtime_targets import describe_isaac_runtime_targets


def _normalized_manifest(*, include_ft: bool = True) -> dict[str, object]:
    manifest = {
        "unitree_robot_description": {"present": True},
        "whole_body_joint_map": {"present": True},
        "camera_extrinsics": {"present": True},
        "imu_extrinsics": {"present": True},
        "actuator_latency_profile": {"present": True},
        "joint_limit_profile": {"present": True},
        "safety_watchdog_profile": {"present": True},
    }
    manifest["force_torque_calibration"] = {"present": include_ft}
    return manifest


def test_deployment_contract_marks_sim_ready_when_assets_policy_and_targets_exist(tmp_path) -> None:
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

    embodiment_context = {
        "unitree_sim_isaaclab_root": str(unitree_sim_root),
        "unitree_sdk2_root": str(sdk_root),
        "unitree_asset_root": str(asset_root),
        "unitree_policy_root": str(policy_root),
        "active_embodiments": ["unitree_g1"],
    }
    contract = build_isaac_unitree_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=describe_isaac_runtime_targets(embodiment_context),
        runtime_layout_contract=describe_isaac_runtime_layouts(embodiment_context),
        policy_contract=describe_isaac_policy_contract(embodiment_context),
        normalized_asset_manifest=_normalized_manifest(),
    )

    assert contract["robot_variant"] == "unitree_g1"
    assert contract["sim_launch_ready"] is True
    assert contract["preferred_profile"] == "unitree_sim_isaaclab"
    assert "sim_eval" in contract["ready_modes"]


def test_deployment_contract_flags_missing_assets_for_physical_deploy(tmp_path) -> None:
    xr_root = tmp_path / "xr_teleoperate"
    xr_root.mkdir()
    (xr_root / "teleop").mkdir()
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()
    (sdk_root / "include").mkdir()
    sdk2_python_root = tmp_path / "sdk2_python"
    sdk2_python_root.mkdir()
    (sdk2_python_root / "setup.py").write_text("", encoding="utf-8")
    teleimager_root = tmp_path / "teleimager"
    teleimager_root.mkdir()
    (teleimager_root / "README.md").write_text("tele", encoding="utf-8")
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "g1.usd").write_text("x", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "g1_policy.onnx").write_text("x", encoding="utf-8")

    embodiment_context = {
        "xr_teleoperate_root": str(xr_root),
        "unitree_sdk2_root": str(sdk_root),
        "unitree_sdk2_python_root": str(sdk2_python_root),
        "teleimager_root": str(teleimager_root),
        "unitree_asset_root": str(asset_root),
        "unitree_policy_root": str(policy_root),
        "active_embodiments": ["unitree_r1"],
    }
    contract = build_isaac_unitree_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=describe_isaac_runtime_targets(embodiment_context),
        runtime_layout_contract=describe_isaac_runtime_layouts(embodiment_context),
        policy_contract=describe_isaac_policy_contract(embodiment_context),
        normalized_asset_manifest=_normalized_manifest(include_ft=False),
    )

    physical = next(
        row for row in contract["deployment_modes"] if row["mode_id"] == "physical_deploy"
    )
    assert contract["robot_variant"] == "unitree_r1"
    assert contract["teleop_launch_ready"] is True
    assert contract["physical_deploy_ready"] is False
    assert "force_torque_calibration" in physical["missing_preconditions"]


def test_deployment_contract_prefers_lerobot_profile_when_requested(tmp_path) -> None:
    lerobot_root = tmp_path / "unitree_lerobot"
    lerobot_root.mkdir()
    (lerobot_root / "examples").mkdir()
    (lerobot_root / "outputs").mkdir()
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()
    (sdk_root / "include").mkdir()
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "g1.usd").write_text("x", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "g1_policy.onnx").write_text("x", encoding="utf-8")

    embodiment_context = {
        "unitree_lerobot_root": str(lerobot_root),
        "unitree_sdk2_root": str(sdk_root),
        "unitree_asset_root": str(asset_root),
        "unitree_policy_root": str(policy_root),
    }
    contract = build_isaac_unitree_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=describe_isaac_runtime_targets(embodiment_context),
        runtime_layout_contract=describe_isaac_runtime_layouts(embodiment_context),
        policy_contract=describe_isaac_policy_contract(embodiment_context),
        normalized_asset_manifest=_normalized_manifest(),
    )

    assert contract["lerobot_eval_ready"] is True
    assert contract["preferred_profile"] == "unitree_lerobot"


def test_deployment_contract_blocks_sim_when_targets_are_only_path_present(tmp_path) -> None:
    unitree_sim_root = tmp_path / "unitree_sim_isaaclab"
    unitree_sim_root.mkdir()
    (unitree_sim_root / "sim_main.py").write_text("", encoding="utf-8")
    (unitree_sim_root / "dds").mkdir()
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "g1_policy.onnx").write_text("x", encoding="utf-8")
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()
    asset_root = tmp_path / "assets"
    asset_root.mkdir()

    embodiment_context = {
        "unitree_sim_isaaclab_root": str(unitree_sim_root),
        "unitree_sdk2_root": str(sdk_root),
        "unitree_asset_root": str(asset_root),
        "unitree_policy_root": str(policy_root),
    }
    contract = build_isaac_unitree_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=describe_isaac_runtime_targets(embodiment_context),
        runtime_layout_contract=describe_isaac_runtime_layouts(embodiment_context),
        policy_contract=describe_isaac_policy_contract(embodiment_context),
        normalized_asset_manifest=_normalized_manifest(),
    )

    sim_eval = next(row for row in contract["deployment_modes"] if row["mode_id"] == "sim_eval")
    assert contract["sim_launch_ready"] is False
    assert "unitree_sdk2_root" in sim_eval["missing_preconditions"]
    assert "unitree_asset_root" in sim_eval["missing_preconditions"]

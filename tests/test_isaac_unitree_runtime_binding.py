from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.isaac_unitree_deployment import (
    build_isaac_unitree_deployment_contract,
)
from src.world_model.sim_synth_physics.adapters.isaac_unitree_runtime_binding import (
    build_isaac_unitree_runtime_binding,
)
from src.world_model.sim_synth_physics.adapters.isaac_unitree_runtime_pack import (
    build_isaac_unitree_runtime_pack,
)
from src.world_model.sim_synth_physics.runtime_layouts import (
    describe_isaac_policy_contract,
    describe_isaac_runtime_layouts,
)
from src.world_model.sim_synth_physics.runtime_targets import describe_isaac_runtime_targets


def test_isaac_runtime_binding_selects_policy_and_launch_root(tmp_path) -> None:
    sim_root = tmp_path / "unitree_sim_isaaclab"
    sim_root.mkdir()
    (sim_root / "sim_main.py").write_text("", encoding="utf-8")
    (sim_root / "dds").mkdir()
    (sim_root / "action_provider").mkdir()
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    policy_path = policy_root / "g1_policy.onnx"
    policy_path.write_text("x", encoding="utf-8")

    embodiment_context = {
        "unitree_sim_isaaclab_root": str(sim_root),
        "unitree_sdk2_root": str(sdk_root),
        "unitree_asset_root": str(asset_root),
        "unitree_policy_root": str(policy_root),
        "active_embodiments": ["unitree_g1"],
    }
    runtime_target_contract = describe_isaac_runtime_targets(embodiment_context)
    runtime_layout_contract = describe_isaac_runtime_layouts(embodiment_context)
    policy_contract = describe_isaac_policy_contract(embodiment_context)
    deployment_contract = build_isaac_unitree_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        normalized_asset_manifest={
            "unitree_robot_description": {"present": True},
            "whole_body_joint_map": {"present": True},
            "camera_extrinsics": {"present": True},
            "imu_extrinsics": {"present": True},
            "force_torque_calibration": {"present": True},
            "actuator_latency_profile": {"present": True},
            "joint_limit_profile": {"present": True},
            "safety_watchdog_profile": {"present": True},
        },
    )
    runtime_pack = build_isaac_unitree_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        normalized_robot_asset_manifest={"unitree_robot_description": {"present": True}},
    )
    binding = build_isaac_unitree_runtime_binding(
        task_id="peg_in_hole",
        explicit_policy_ref="",
        preferred_profile="unitree_sim_isaaclab",
        launch_specs=[
            {
                "profile_id": "unitree_sim_isaaclab",
                "root": str(sim_root),
                "command": "python sim_main.py --task peg_in_hole --policy /tmp/g1.onnx --headless",
            }
        ],
        runtime_target_contract=runtime_target_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        upstream_runtime_pack=runtime_pack,
    )

    assert binding["binding_status"] == "binding_ready"
    assert binding["selected_profile"] == "unitree_sim_isaaclab"
    assert binding["selected_policy_ref"].endswith("g1_policy.onnx")
    assert binding["selected_launch_root"] == str(sim_root)


def test_isaac_runtime_binding_blocks_when_launch_command_missing(tmp_path) -> None:
    sim_root = tmp_path / "unitree_sim_isaaclab"
    sim_root.mkdir()
    (sim_root / "sim_main.py").write_text("", encoding="utf-8")
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()
    asset_root = tmp_path / "assets"
    asset_root.mkdir()

    embodiment_context = {
        "unitree_sim_isaaclab_root": str(sim_root),
        "unitree_sdk2_root": str(sdk_root),
        "unitree_asset_root": str(asset_root),
    }
    runtime_target_contract = describe_isaac_runtime_targets(embodiment_context)
    runtime_layout_contract = describe_isaac_runtime_layouts(embodiment_context)
    policy_contract = describe_isaac_policy_contract(embodiment_context)
    deployment_contract = build_isaac_unitree_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        normalized_asset_manifest={"unitree_robot_description": {"present": True}},
    )
    runtime_pack = build_isaac_unitree_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        normalized_robot_asset_manifest={"unitree_robot_description": {"present": True}},
    )
    binding = build_isaac_unitree_runtime_binding(
        task_id="peg_in_hole",
        explicit_policy_ref="",
        preferred_profile="unitree_sim_isaaclab",
        launch_specs=[{"profile_id": "unitree_sim_isaaclab", "root": str(sim_root), "command": ""}],
        runtime_target_contract=runtime_target_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        upstream_runtime_pack=runtime_pack,
    )

    assert binding["binding_status"] in {"binding_partial", "binding_blocked"}
    assert "launch_command" in binding["missing_components"]

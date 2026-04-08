from __future__ import annotations

from pathlib import Path

from src.world_model.sim_synth_physics.adapters.holosoma_deployment import (
    build_holosoma_deployment_contract,
)
from src.world_model.sim_synth_physics.adapters.holosoma_runtime_pack import (
    build_holosoma_runtime_pack,
)
from src.world_model.sim_synth_physics.adapters.isaac_unitree_deployment import (
    build_isaac_unitree_deployment_contract,
)
from src.world_model.sim_synth_physics.adapters.isaac_unitree_runtime_pack import (
    build_isaac_unitree_runtime_pack,
)
from src.world_model.sim_synth_physics.runtime_bundles import build_backend_runtime_bundle
from src.world_model.sim_synth_physics.runtime_layouts import (
    describe_holosoma_policy_contract,
    describe_holosoma_runtime_layouts,
    describe_isaac_policy_contract,
    describe_isaac_runtime_layouts,
)
from src.world_model.sim_synth_physics.runtime_targets import (
    describe_holosoma_runtime_targets,
    describe_isaac_runtime_targets,
)


def test_build_isaac_runtime_bundle_prefers_unitree_sim_profile(tmp_path: Path) -> None:
    unitree_sim_root = tmp_path / "unitree_sim_isaaclab"
    unitree_sim_root.mkdir()
    (unitree_sim_root / "sim_main.py").write_text("", encoding="utf-8")
    (unitree_sim_root / "dds").mkdir()
    (unitree_sim_root / "action_provider").mkdir()
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    policy_path = policy_root / "g1_policy.onnx"
    policy_path.write_text("x", encoding="utf-8")
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "g1.usd").write_text("x", encoding="utf-8")
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()
    (sdk_root / "include").mkdir()

    embodiment_context = {
        "unitree_sim_isaaclab_root": str(unitree_sim_root),
        "unitree_policy_root": str(policy_root),
        "unitree_asset_root": str(asset_root),
        "unitree_sdk2_root": str(sdk_root),
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
    upstream_runtime_pack = build_isaac_unitree_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        normalized_robot_asset_manifest={
            "unitree_robot_description": {"present": True, "value": "/assets/g1.usd"},
            "whole_body_joint_map": {"present": True, "value": "/assets/joint_map.yaml"},
        },
    )
    refs, runtime_bundle, launch_spec = build_backend_runtime_bundle(
        backend="isaac",
        task_id="peg_in_hole",
        policy_ref=str(policy_path),
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        robot_asset_manifest={"unitree_usd": "/assets/g1.usd"},
        normalized_robot_asset_manifest={
            "unitree_robot_description": {"present": True, "value": "/assets/g1.usd"}
        },
        deployment_contract=deployment_contract,
        upstream_runtime_pack=upstream_runtime_pack,
        output_root=tmp_path / "bundle",
    )

    assert refs
    assert runtime_bundle["preferred_profile"] == "unitree_sim_isaaclab"
    assert "unitree_sim_isaaclab" in runtime_bundle["ready_profiles"]
    assert "unitree_sim_isaaclab" in runtime_bundle["usable_profiles"]
    assert launch_spec["preferred_profile"] == "unitree_sim_isaaclab"
    assert "sim_main.py" in launch_spec["command"]
    assert launch_spec["policy_ready"] is True
    assert launch_spec["deployment_contract"]["sim_launch_ready"] is True
    assert (
        launch_spec["executable_adapter_request"]["deployment_mode"] == "sim_eval"
    )
    assert (
        launch_spec["executable_adapter_request"]["adapter_entrypoint"]
        == "isaaclab_unitree_sim"
    )
    assert (
        runtime_bundle["executable_adapter_consumer"]["consumer_mode"]
        == "external_sim_launch"
    )
    assert runtime_bundle["output_contract"]["profile_id"] == "unitree_sim_isaaclab"
    assert launch_spec["output_contract"]["profile_id"] == "unitree_sim_isaaclab"
    assert runtime_bundle["output_contract"]["sources"]
    assert runtime_bundle["upstream_runtime_pack"]["pack_status"] == "pack_ready"
    assert runtime_bundle["upstream_runtime_pack"]["primary_policy_ref"] == str(policy_path)
    assert runtime_bundle["upstream_runtime_pack"]["profile_candidate_counts"]["deploy"] >= 1
    assert (
        runtime_bundle["upstream_runtime_pack"]["profile_install_preflight_status"]
        == "install_ready"
    )
    assert runtime_bundle["runtime_binding"]["binding_status"] == "binding_ready"
    assert runtime_bundle["runtime_binding"]["selected_policy_ref"] == str(policy_path)
    assert runtime_bundle["runtime_binding"]["selected_deploy_config"].endswith("sim_main.py")
    assert (
        runtime_bundle["runtime_binding"]["selected_profile_primary_entrypoint_ref"].endswith(
            "sim_main.py"
        )
    )
    assert launch_spec["upstream_runtime_pack"]["ready_surfaces"]
    assert launch_spec["runtime_binding"]["selected_profile"] == "unitree_sim_isaaclab"


def test_build_isaac_runtime_bundle_can_prefer_lerobot_profile(tmp_path: Path) -> None:
    lerobot_root = tmp_path / "unitree_lerobot"
    lerobot_root.mkdir()
    (lerobot_root / "examples").mkdir()
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    policy_path = policy_root / "g1_policy.onnx"
    policy_path.write_text("x", encoding="utf-8")
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()

    embodiment_context = {
        "unitree_lerobot_root": str(lerobot_root),
        "unitree_sdk2_root": str(sdk_root),
        "unitree_asset_root": str(asset_root),
        "unitree_policy_root": str(policy_root),
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

    _, runtime_bundle, launch_spec = build_backend_runtime_bundle(
        backend="isaac",
        task_id="walk_forward",
        policy_ref=str(policy_path),
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        robot_asset_manifest={"unitree_usd": "/assets/g1.usd"},
        normalized_robot_asset_manifest={
            "unitree_robot_description": {"present": True, "value": "/assets/g1.usd"}
        },
        deployment_contract=deployment_contract,
        upstream_runtime_pack=build_isaac_unitree_runtime_pack(
            runtime_target_contract=runtime_target_contract,
            runtime_layout_contract=runtime_layout_contract,
            policy_contract=policy_contract,
            deployment_contract=deployment_contract,
            normalized_robot_asset_manifest={
                "unitree_robot_description": {"present": True, "value": "/assets/g1.usd"}
            },
        ),
        output_root=tmp_path / "bundle",
    )

    assert runtime_bundle["preferred_profile"] == "unitree_lerobot"
    assert launch_spec["preferred_profile"] == "unitree_lerobot"
    assert "eval_policy.py" in launch_spec["command"]
    assert (
        launch_spec["executable_adapter_request"]["deployment_mode"] == "lerobot_eval"
    )
    assert (
        launch_spec["executable_adapter_request"]["adapter_entrypoint"]
        == "unitree_lerobot_eval"
    )
    assert (
        launch_spec["executable_adapter_consumer"]["consumer_mode"]
        == "external_lerobot_eval"
    )
    assert runtime_bundle["upstream_runtime_pack"]["preferred_profile"] == "unitree_lerobot"
    assert runtime_bundle["upstream_runtime_pack"]["primary_policy_ref"] == str(policy_path)
    assert runtime_bundle["runtime_binding"]["selected_profile"] == "unitree_lerobot"


def test_build_holosoma_runtime_bundle_prefers_repo_profile(tmp_path: Path) -> None:
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    (holosoma_root / "holosoma").mkdir()
    (holosoma_root / "holosoma" / "__init__.py").write_text("", encoding="utf-8")
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    motion_clip = motion_root / "g1_walk.npz"
    motion_clip.write_text("x", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    policy_path = policy_root / "policy.ckpt"
    policy_path.write_text("x", encoding="utf-8")
    retargeting_root = tmp_path / "retargeting"
    retargeting_root.mkdir()

    embodiment_context = {
        "holosoma_root": str(holosoma_root),
        "holosoma_motion_root": str(motion_root),
        "holosoma_policy_root": str(policy_root),
        "retargeting_root": str(retargeting_root),
        "motion_clip_paths": [str(motion_clip)],
    }
    runtime_target_contract = describe_holosoma_runtime_targets(embodiment_context)
    runtime_layout_contract = describe_holosoma_runtime_layouts(embodiment_context)
    policy_contract = describe_holosoma_policy_contract(embodiment_context)
    deployment_contract = build_holosoma_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
    )
    upstream_runtime_pack = build_holosoma_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        embodiment_context=embodiment_context,
    )
    refs, runtime_bundle, launch_spec = build_backend_runtime_bundle(
        backend="holosoma",
        task_id="humanoid_wbt_g1",
        policy_ref=str(policy_path),
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        robot_asset_manifest={},
        normalized_robot_asset_manifest={},
        deployment_contract=deployment_contract,
        upstream_runtime_pack=upstream_runtime_pack,
        output_root=tmp_path / "bundle",
    )

    assert refs
    assert runtime_bundle["preferred_profile"] == "holosoma_repo"
    assert "holosoma_repo" in runtime_bundle["usable_profiles"]
    assert launch_spec["preferred_profile"] == "holosoma_repo"
    assert "holosoma.eval" in launch_spec["command"]
    assert launch_spec["policy_ready"] is True
    assert runtime_bundle["executable_adapter_request"]["adapter_family"] == "holosoma"
    assert runtime_bundle["executable_adapter_consumer"]["consumer_mode"] in {
        "local_runtime_binding",
        "external_runtime_launch",
    }
    assert runtime_bundle["output_contract"]["profile_id"] == "holosoma_repo"
    assert runtime_bundle["output_contract"]["sources"]
    assert runtime_bundle["upstream_runtime_pack"]["pack_status"] == "pack_ready"
    assert "policy_surface" in runtime_bundle["upstream_runtime_pack"]["ready_surfaces"]
    assert runtime_bundle["upstream_runtime_pack"]["primary_policy_ref"] == str(policy_path)
    assert runtime_bundle["upstream_runtime_pack"]["existing_motion_sources"] == [str(motion_clip)]
    assert (
        runtime_bundle["upstream_runtime_pack"]["profile_install_preflight_status"]
        == "install_ready"
    )
    assert runtime_bundle["runtime_binding"]["binding_status"] == "binding_ready"
    assert runtime_bundle["runtime_binding"]["selected_profile"] == "holosoma_repo"
    assert (
        runtime_bundle["runtime_binding"]["selected_profile_primary_entrypoint_ref"].endswith(
            "holosoma"
        )
    )

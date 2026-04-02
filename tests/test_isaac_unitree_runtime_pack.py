from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.isaac_unitree_deployment import (
    build_isaac_unitree_deployment_contract,
)
from src.world_model.sim_synth_physics.adapters.isaac_unitree_runtime_pack import (
    build_isaac_unitree_runtime_pack,
)
from src.world_model.sim_synth_physics.runtime_layouts import (
    describe_isaac_policy_contract,
    describe_isaac_runtime_layouts,
)
from src.world_model.sim_synth_physics.runtime_targets import describe_isaac_runtime_targets


def test_isaac_unitree_runtime_pack_tracks_ready_surfaces(tmp_path) -> None:
    sim_root = tmp_path / "unitree_sim_isaaclab"
    sim_root.mkdir()
    (sim_root / "sim_main.py").write_text("", encoding="utf-8")
    (sim_root / "dds").mkdir()
    (sim_root / "action_provider").mkdir()
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "g1_policy.onnx").write_text("x", encoding="utf-8")
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    (asset_root / "g1.usd").write_text("x", encoding="utf-8")
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()
    (sdk_root / "include").mkdir()

    embodiment_context = {
        "unitree_sim_isaaclab_root": str(sim_root),
        "unitree_policy_root": str(policy_root),
        "unitree_asset_root": str(asset_root),
        "unitree_sdk2_root": str(sdk_root),
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
            "unitree_robot_description": {"present": True, "value": "/assets/g1.usd"},
            "whole_body_joint_map": {"present": True, "value": "/assets/joint_map.yaml"},
            "camera_extrinsics": {"present": True, "value": "/assets/cam.json"},
            "imu_extrinsics": {"present": True, "value": "/assets/imu.json"},
            "force_torque_calibration": {"present": True, "value": "/assets/ft.json"},
            "actuator_latency_profile": {"present": True, "value": "/assets/latency.yaml"},
            "joint_limit_profile": {"present": True, "value": "/assets/joint_limits.yaml"},
            "safety_watchdog_profile": {"present": True, "value": "/assets/watchdog.yaml"},
        },
    )
    pack = build_isaac_unitree_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        normalized_robot_asset_manifest={
            "unitree_robot_description": {"present": True, "value": "/assets/g1.usd"},
            "whole_body_joint_map": {"present": True, "value": "/assets/joint_map.yaml"},
        },
    )

    assert pack["pack_status"] == "pack_ready"
    assert "runtime_profile_surface" in pack["ready_surfaces"]
    assert "policy_surface" in pack["ready_surfaces"]
    assert "asset_surface" in pack["ready_surfaces"]
    assert pack["preferred_profile"] == "unitree_sim_isaaclab"
    assert pack["profile_candidate_counts"]["deploy"] >= 1
    assert pack["profile_install_preflight_status"] == "install_ready"
    assert pack["profile_primary_entrypoint_ref"].endswith("sim_main.py")
    assert pack["primary_profile_deploy_ref"].endswith("sim_main.py")
    assert pack["primary_policy_ref"].endswith("g1_policy.onnx")
    assert pack["asset_evidence_summary"]["declared_asset_count"] == 2
    assert pack["asset_evidence_summary"]["verified_asset_count"] == 0
    assert pack["declared_only_asset_ids"] == [
        "unitree_robot_description",
        "whole_body_joint_map",
    ]


def test_isaac_unitree_runtime_pack_stays_partial_when_policy_missing(tmp_path) -> None:
    sim_root = tmp_path / "unitree_sim_isaaclab"
    sim_root.mkdir()
    (sim_root / "sim_main.py").write_text("", encoding="utf-8")
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()
    (sdk_root / "include").mkdir()
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
    pack = build_isaac_unitree_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        normalized_robot_asset_manifest={"unitree_robot_description": {"present": True}},
    )

    assert pack["pack_status"] == "pack_partial"
    assert pack["profile_install_preflight_status"] == "install_partial"
    assert "policy_checkpoint" in pack["missing_components"]
    assert pack["primary_policy_ref"] == ""


def test_isaac_unitree_runtime_pack_uses_verified_targets_for_runtime_surface(tmp_path) -> None:
    sim_root = tmp_path / "unitree_sim_isaaclab"
    sim_root.mkdir()
    (sim_root / "sim_main.py").write_text("", encoding="utf-8")
    (sim_root / "dds").mkdir()
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "g1_policy.onnx").write_text("x", encoding="utf-8")
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()
    asset_root = tmp_path / "assets"
    asset_root.mkdir()

    embodiment_context = {
        "unitree_sim_isaaclab_root": str(sim_root),
        "unitree_policy_root": str(policy_root),
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
    pack = build_isaac_unitree_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        normalized_robot_asset_manifest={"unitree_robot_description": {"present": True}},
    )

    assert sorted(pack["runtime_target_ids"]) == [
        "unitree_policy_root",
        "unitree_sim_isaaclab_root",
    ]
    assert pack["runtime_target_preflight_status"] == "preflight_partial"
    assert sorted(pack["unverified_runtime_target_ids"]) == [
        "unitree_asset_root",
        "unitree_sdk2_root",
    ]


def test_isaac_unitree_runtime_pack_prefers_verified_local_refs(tmp_path) -> None:
    policy_ref = tmp_path / "policies" / "g1_policy.onnx"
    policy_ref.parent.mkdir()
    policy_ref.write_text("x", encoding="utf-8")
    deploy_ref = tmp_path / "unitree_sim_isaaclab" / "sim_main.py"
    deploy_ref.parent.mkdir()
    deploy_ref.write_text("", encoding="utf-8")
    report_ref = tmp_path / "logs" / "eval.json"
    report_ref.parent.mkdir()
    report_ref.write_text("{}", encoding="utf-8")

    pack = build_isaac_unitree_runtime_pack(
        runtime_target_contract={
            "verified_target_ids": ["unitree_sim_isaaclab_root", "unitree_sdk2_root"],
            "runtime_target_preflight_status": "preflight_ready",
        },
        runtime_layout_contract={
            "preferred_profile_order": ["unitree_sim_isaaclab"],
            "profiles": [
                {
                    "profile_id": "unitree_sim_isaaclab",
                    "root": str(deploy_ref.parent),
                    "root_exists": True,
                    "install_preflight_status": "install_ready",
                    "primary_entrypoint_ref": str(deploy_ref),
                    "policy_candidates": ["/missing/policy.onnx", str(policy_ref)],
                    "deploy_candidates": ["/missing/sim_main.py", str(deploy_ref)],
                    "data_candidates": ["/missing/eval.json", str(report_ref)],
                }
            ],
        },
        policy_contract={
            "checkpoint_candidates": ["/also/missing/policy.onnx", str(policy_ref)],
            "deploy_config_candidates": ["/also/missing/sim_main.py", str(deploy_ref)],
            "runtime_report_candidates": ["/also/missing/eval.json", str(report_ref)],
        },
        deployment_contract={
            "preferred_profile": "unitree_sim_isaaclab",
            "ready_modes": ["sim_eval"],
            "robot_variant": "unitree_g1",
        },
        normalized_robot_asset_manifest={"unitree_robot_description": {"present": True}},
    )

    assert pack["primary_policy_ref"] == str(policy_ref)
    assert pack["primary_policy_ref_source"] == "profile.policy_candidates[1]"
    assert pack["policy_candidate_evidence_summary"]["verified_candidate_count"] == 1
    assert pack["primary_deploy_config_ref"] == str(deploy_ref)
    assert pack["primary_deploy_config_ref_source"] == "profile.deploy_candidates[1]"
    assert pack["runtime_report_candidate_evidence_summary"]["verified_candidate_count"] == 1
    assert pack["primary_runtime_report_ref"] == str(report_ref)

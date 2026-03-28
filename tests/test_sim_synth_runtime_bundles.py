from __future__ import annotations

from pathlib import Path

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
    sdk_root = tmp_path / "sdk2"
    sdk_root.mkdir()

    embodiment_context = {
        "unitree_sim_isaaclab_root": str(unitree_sim_root),
        "unitree_policy_root": str(policy_root),
        "unitree_asset_root": str(asset_root),
        "unitree_sdk2_root": str(sdk_root),
    }
    refs, runtime_bundle, launch_spec = build_backend_runtime_bundle(
        backend="isaac",
        task_id="peg_in_hole",
        policy_ref=str(policy_path),
        runtime_target_contract=describe_isaac_runtime_targets(embodiment_context),
        runtime_layout_contract=describe_isaac_runtime_layouts(embodiment_context),
        policy_contract=describe_isaac_policy_contract(embodiment_context),
        robot_asset_manifest={"unitree_usd": "/assets/g1.usd"},
        normalized_robot_asset_manifest={
            "unitree_robot_description": {"present": True, "value": "/assets/g1.usd"}
        },
        output_root=tmp_path / "bundle",
    )

    assert refs
    assert runtime_bundle["preferred_profile"] == "unitree_sim_isaaclab"
    assert "unitree_sim_isaaclab" in runtime_bundle["ready_profiles"]
    assert launch_spec["preferred_profile"] == "unitree_sim_isaaclab"
    assert "sim_main.py" in launch_spec["command"]
    assert launch_spec["policy_ready"] is True
    assert runtime_bundle["output_contract"]["profile_id"] == "unitree_sim_isaaclab"
    assert launch_spec["output_contract"]["profile_id"] == "unitree_sim_isaaclab"
    assert runtime_bundle["output_contract"]["sources"]


def test_build_holosoma_runtime_bundle_prefers_repo_profile(tmp_path: Path) -> None:
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
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
    }
    refs, runtime_bundle, launch_spec = build_backend_runtime_bundle(
        backend="holosoma",
        task_id="humanoid_wbt_g1",
        policy_ref=str(policy_path),
        runtime_target_contract=describe_holosoma_runtime_targets(embodiment_context),
        runtime_layout_contract=describe_holosoma_runtime_layouts(embodiment_context),
        policy_contract=describe_holosoma_policy_contract(embodiment_context),
        robot_asset_manifest={},
        normalized_robot_asset_manifest={},
        output_root=tmp_path / "bundle",
    )

    assert refs
    assert runtime_bundle["preferred_profile"] == "holosoma_repo"
    assert launch_spec["preferred_profile"] == "holosoma_repo"
    assert "holosoma.eval" in launch_spec["command"]
    assert launch_spec["policy_ready"] is True
    assert runtime_bundle["output_contract"]["profile_id"] == "holosoma_repo"
    assert runtime_bundle["output_contract"]["sources"]

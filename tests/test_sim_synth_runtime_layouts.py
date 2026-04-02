from __future__ import annotations

from src.world_model.sim_synth_physics.runtime_layouts import (
    describe_holosoma_policy_contract,
    describe_holosoma_runtime_layouts,
    describe_isaac_policy_contract,
    describe_isaac_runtime_layouts,
)


def test_isaac_runtime_layouts_detect_oss_repo_shapes(tmp_path) -> None:
    unitree_sim_root = tmp_path / "unitree_sim_isaaclab"
    unitree_sim_root.mkdir()
    (unitree_sim_root / "sim_main.py").write_text("", encoding="utf-8")
    (unitree_sim_root / "dds").mkdir()
    (unitree_sim_root / "action_provider").mkdir()
    logs_dir = unitree_sim_root / "logs" / "run_1"
    logs_dir.mkdir(parents=True)
    (logs_dir / "policy.onnx").write_text("x", encoding="utf-8")
    (logs_dir / "metrics.json").write_text("{}", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "policy.onnx").write_text("x", encoding="utf-8")

    contract = describe_isaac_runtime_layouts(
        {
            "unitree_sim_isaaclab_root": str(unitree_sim_root),
        }
    )
    policy_contract = describe_isaac_policy_contract(
        {
            "isaac_policy_root": str(policy_root),
        }
    )

    assert "unitree_sim_isaaclab" in contract["ready_profiles"]
    profile = next(
        profile for profile in contract["profiles"] if profile["profile_id"] == "unitree_sim_isaaclab"
    )
    assert profile["deploy_candidates"]
    assert profile["policy_candidates"]
    assert profile["data_candidates"]
    assert profile["deploy_candidate_count"] >= 1
    assert profile["policy_candidate_count"] >= 1
    assert profile["primary_deploy_candidate"].endswith("sim_main.py")
    assert profile["primary_policy_candidate"].endswith("policy.onnx")
    assert policy_contract["policy_ready"] is True
    assert policy_contract["checkpoint_candidates"]
    assert policy_contract["checkpoint_candidate_count"] >= 1
    assert policy_contract["primary_checkpoint_ref"].endswith("policy.onnx")
    assert policy_contract["runtime_report_candidates"] == []


def test_isaac_runtime_layouts_detect_lerobot_profile(tmp_path) -> None:
    lerobot_root = tmp_path / "unitree_lerobot"
    lerobot_root.mkdir()
    (lerobot_root / "examples").mkdir()
    outputs_dir = lerobot_root / "outputs" / "run_1"
    outputs_dir.mkdir(parents=True)
    (outputs_dir / "policy.onnx").write_text("x", encoding="utf-8")

    contract = describe_isaac_runtime_layouts(
        {
            "unitree_lerobot_root": str(lerobot_root),
        }
    )

    assert "unitree_lerobot" in contract["ready_profiles"]
    profile = next(
        profile for profile in contract["profiles"] if profile["profile_id"] == "unitree_lerobot"
    )
    assert profile["policy_candidates"]
    assert profile["policy_candidate_count"] >= 1
    assert profile["deploy_candidates"] == []


def test_holosoma_runtime_layouts_and_policy_contracts_detect_roots(tmp_path) -> None:
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    (motion_root / "g1_walk.npz").write_text("x", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "policy.ckpt").write_text("x", encoding="utf-8")
    (policy_root / "deploy.yaml").write_text("policy: g1", encoding="utf-8")
    retargeting_root = tmp_path / "retargeting"
    retargeting_root.mkdir()
    (retargeting_root / "g1_retargeting.yaml").write_text("{}", encoding="utf-8")

    contract = describe_holosoma_runtime_layouts(
        {
            "holosoma_root": str(holosoma_root),
            "holosoma_motion_root": str(motion_root),
            "holosoma_policy_root": str(policy_root),
            "retargeting_root": str(retargeting_root),
        }
    )
    policy_contract = describe_holosoma_policy_contract(
        {
            "holosoma_policy_root": str(policy_root),
        }
    )

    assert "holosoma_repo" in contract["ready_profiles"]
    assert "holosoma_motion_bank" in contract["ready_profiles"]
    motion_profile = next(
        profile for profile in contract["profiles"] if profile["profile_id"] == "holosoma_motion_bank"
    )
    assert motion_profile["data_candidates"]
    assert motion_profile["data_candidate_count"] >= 1
    assert policy_contract["policy_ready"] is True
    assert policy_contract["deploy_config_candidates"]
    assert policy_contract["checkpoint_candidate_count"] >= 1
    assert policy_contract["primary_checkpoint_ref"].endswith("policy.ckpt")
    assert policy_contract["primary_deploy_config_ref"].endswith("deploy.yaml")

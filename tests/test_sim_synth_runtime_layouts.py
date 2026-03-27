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
    assert policy_contract["policy_ready"] is True
    assert policy_contract["checkpoint_candidates"]


def test_holosoma_runtime_layouts_and_policy_contracts_detect_roots(tmp_path) -> None:
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "policy.ckpt").write_text("x", encoding="utf-8")
    retargeting_root = tmp_path / "retargeting"
    retargeting_root.mkdir()

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
    assert policy_contract["policy_ready"] is True

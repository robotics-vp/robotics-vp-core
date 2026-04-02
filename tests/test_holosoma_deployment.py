from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.holosoma_deployment import (
    build_holosoma_deployment_contract,
)
from src.world_model.sim_synth_physics.runtime_layouts import (
    describe_holosoma_policy_contract,
    describe_holosoma_runtime_layouts,
)
from src.world_model.sim_synth_physics.runtime_targets import describe_holosoma_runtime_targets


def test_holosoma_deployment_contract_marks_motion_training_ready(tmp_path) -> None:
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    (motion_root / "g1_walk.npz").write_text("x", encoding="utf-8")
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")

    embodiment_context = {
        "holosoma_root": str(holosoma_root),
        "holosoma_motion_root": str(motion_root),
        "motion_clip_paths": [str(motion_root / "g1_walk.npz")],
        "active_embodiments": ["unitree_g1"],
    }
    contract = build_holosoma_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=describe_holosoma_runtime_targets(embodiment_context),
        runtime_layout_contract=describe_holosoma_runtime_layouts(embodiment_context),
        policy_contract=describe_holosoma_policy_contract(embodiment_context),
    )

    assert contract["motion_train_ready"] is True
    assert "motion_train" in contract["ready_modes"]
    assert contract["preferred_profile"] in {"holosoma_repo", "holosoma_motion_bank"}


def test_holosoma_deployment_contract_flags_retarget_eval_gaps(tmp_path) -> None:
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    (motion_root / "g1_walk.npz").write_text("x", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "policy.ckpt").write_text("x", encoding="utf-8")

    embodiment_context = {
        "holosoma_root": str(holosoma_root),
        "holosoma_motion_root": str(motion_root),
        "holosoma_policy_root": str(policy_root),
        "active_embodiments": ["unitree_r1"],
    }
    contract = build_holosoma_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=describe_holosoma_runtime_targets(embodiment_context),
        runtime_layout_contract=describe_holosoma_runtime_layouts(embodiment_context),
        policy_contract=describe_holosoma_policy_contract(embodiment_context),
    )

    retarget_eval = next(
        row for row in contract["deployment_modes"] if row["mode_id"] == "retarget_eval"
    )
    assert contract["sim_launch_ready"] is True
    assert contract["retarget_eval_ready"] is False
    assert "whole_body_retargeting_contract" in retarget_eval["missing_preconditions"]


def test_holosoma_deployment_contract_ignores_install_blocked_repo_when_motion_bank_is_usable(
    tmp_path,
) -> None:
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    (motion_root / "g1_walk.npz").write_text("x", encoding="utf-8")

    embodiment_context = {
        "holosoma_root": str(holosoma_root),
        "holosoma_motion_root": str(motion_root),
        "motion_clip_paths": [str(motion_root / "g1_walk.npz")],
    }
    contract = build_holosoma_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=describe_holosoma_runtime_targets(embodiment_context),
        runtime_layout_contract=describe_holosoma_runtime_layouts(embodiment_context),
        policy_contract=describe_holosoma_policy_contract(embodiment_context),
    )

    assert contract["motion_train_ready"] is True
    assert contract["preferred_profile"] == "holosoma_motion_bank"


def test_holosoma_deployment_contract_uses_repo_derived_motion_root(tmp_path) -> None:
    holosoma_root = tmp_path / "holosoma"
    motion_root = holosoma_root / "src" / "holosoma" / "holosoma" / "data" / "motions"
    motion_root.mkdir(parents=True)
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    (holosoma_root / "scripts").mkdir()
    (motion_root / "g1_walk.npz").write_text("x", encoding="utf-8")

    embodiment_context = {
        "holosoma_root": str(holosoma_root),
    }
    runtime_target_contract = describe_holosoma_runtime_targets(embodiment_context)
    runtime_layout_contract = describe_holosoma_runtime_layouts(embodiment_context)
    contract = build_holosoma_deployment_contract(
        embodiment_context=embodiment_context,
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=describe_holosoma_policy_contract(embodiment_context),
    )

    assert "holosoma_motion_root" in contract["verified_target_ids"]
    assert contract["motion_train_ready"] is True

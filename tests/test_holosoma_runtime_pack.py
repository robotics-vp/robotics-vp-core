from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.holosoma_deployment import (
    build_holosoma_deployment_contract,
)
from src.world_model.sim_synth_physics.adapters.holosoma_runtime_pack import (
    build_holosoma_runtime_pack,
)
from src.world_model.sim_synth_physics.runtime_layouts import (
    describe_holosoma_policy_contract,
    describe_holosoma_runtime_layouts,
)
from src.world_model.sim_synth_physics.runtime_targets import describe_holosoma_runtime_targets


def test_holosoma_runtime_pack_tracks_motion_and_retargeting_surfaces(tmp_path) -> None:
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
    (policy_root / "policy.ckpt").write_text("x", encoding="utf-8")
    retargeting_root = tmp_path / "retargeting"
    retargeting_root.mkdir()
    (retargeting_root / "g1_retarget.yaml").write_text("{}", encoding="utf-8")

    embodiment_context = {
        "holosoma_root": str(holosoma_root),
        "holosoma_motion_root": str(motion_root),
        "holosoma_policy_root": str(policy_root),
        "retargeting_root": str(retargeting_root),
        "motion_clip_paths": [str(motion_clip)],
        "whole_body_retargeting": {"contract_id": "retarget_v1"},
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
    pack = build_holosoma_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        embodiment_context=embodiment_context,
    )

    assert pack["pack_status"] == "pack_ready"
    assert "motion_surface" in pack["ready_surfaces"]
    assert "retargeting_surface" in pack["ready_surfaces"]
    assert pack["preferred_profile"] == "holosoma_repo"
    assert pack["profile_install_preflight_status"] == "install_ready"
    assert pack["profile_primary_entrypoint_ref"].endswith("holosoma")


def test_holosoma_runtime_pack_allows_motion_train_without_policy(tmp_path) -> None:
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    motion_clip = motion_root / "g1_walk.npz"
    motion_clip.write_text("x", encoding="utf-8")

    embodiment_context = {
        "holosoma_root": str(holosoma_root),
        "holosoma_motion_root": str(motion_root),
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
    pack = build_holosoma_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        embodiment_context=embodiment_context,
    )

    assert pack["pack_status"] in {"pack_ready", "pack_partial"}
    assert "motion_train" in pack["ready_modes"]
    assert "policy_checkpoint" not in pack["missing_components"]
    assert pack["profile_install_preflight_status"] == "install_blocked"

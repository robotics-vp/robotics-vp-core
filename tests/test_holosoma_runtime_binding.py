from __future__ import annotations

from src.world_model.sim_synth_physics.adapters.holosoma_deployment import (
    build_holosoma_deployment_contract,
)
from src.world_model.sim_synth_physics.adapters.holosoma_runtime_binding import (
    build_holosoma_runtime_binding,
)
from src.world_model.sim_synth_physics.adapters.holosoma_runtime_pack import (
    build_holosoma_runtime_pack,
)
from src.world_model.sim_synth_physics.runtime_layouts import (
    describe_holosoma_policy_contract,
    describe_holosoma_runtime_layouts,
)
from src.world_model.sim_synth_physics.runtime_targets import describe_holosoma_runtime_targets


def test_holosoma_runtime_binding_prefers_motion_bank_when_policy_missing(tmp_path) -> None:
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
    runtime_pack = build_holosoma_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        embodiment_context=embodiment_context,
    )
    binding = build_holosoma_runtime_binding(
        task_id="humanoid_wbt_g1",
        explicit_policy_ref="",
        preferred_profile="holosoma_repo",
        launch_specs=[
            {
                "profile_id": "holosoma_repo",
                "root": str(holosoma_root),
                "command": "python -m holosoma.eval --task-id humanoid_wbt_g1",
            },
            {
                "profile_id": "holosoma_motion_bank",
                "root": str(motion_root),
                "command": "python scripts/local_holosoma_smoke.py --task-id humanoid_wbt_g1 --episodes 1",
            },
        ],
        runtime_target_contract=runtime_target_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        upstream_runtime_pack=runtime_pack,
    )

    assert binding["selected_profile"] == "holosoma_motion_bank"
    assert binding["deployment_mode"] == "motion_train"
    assert binding["binding_status"] == "binding_ready"
    assert binding["selected_motion_sources"]


def test_holosoma_runtime_binding_requires_retargeting_for_retarget_eval(tmp_path) -> None:
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    motion_clip = motion_root / "g1_walk.npz"
    motion_clip.write_text("x", encoding="utf-8")
    policy_root = tmp_path / "policies"
    policy_root.mkdir()
    (policy_root / "policy.ckpt").write_text("x", encoding="utf-8")

    embodiment_context = {
        "holosoma_root": str(holosoma_root),
        "holosoma_motion_root": str(motion_root),
        "holosoma_policy_root": str(policy_root),
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
    runtime_pack = build_holosoma_runtime_pack(
        runtime_target_contract=runtime_target_contract,
        runtime_layout_contract=runtime_layout_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        embodiment_context=embodiment_context,
    )
    binding = build_holosoma_runtime_binding(
        task_id="humanoid_wbt_g1",
        explicit_policy_ref="",
        preferred_profile="retargeting_bundle",
        launch_specs=[
            {
                "profile_id": "retargeting_bundle",
                "root": "",
                "command": "python -m holosoma.eval --task-id humanoid_wbt_g1 --retarget",
                "deployment_mode": "retarget_eval",
            }
        ],
        runtime_target_contract=runtime_target_contract,
        policy_contract=policy_contract,
        deployment_contract=deployment_contract,
        upstream_runtime_pack=runtime_pack,
    )

    assert binding["binding_status"] in {"binding_partial", "binding_blocked"}
    assert "retargeting_surface" in binding["missing_components"] or "retargeting_root" in binding["missing_components"]

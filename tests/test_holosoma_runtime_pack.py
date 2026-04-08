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
    assert pack["preferred_profile"] == "holosoma_motion_bank"
    assert pack["profile_install_preflight_status"] == "install_ready"


def test_holosoma_runtime_pack_falls_back_to_motion_bank_when_repo_install_blocked(
    tmp_path,
) -> None:
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

    assert pack["preferred_profile"] == "holosoma_motion_bank"
    assert sorted(pack["runtime_target_ids"]) == [
        "holosoma_motion_root",
        "holosoma_root",
    ]


def test_holosoma_runtime_pack_prefers_verified_local_refs(tmp_path) -> None:
    policy_ref = tmp_path / "policies" / "policy.ckpt"
    policy_ref.parent.mkdir()
    policy_ref.write_text("x", encoding="utf-8")
    report_ref = tmp_path / "outputs" / "eval.json"
    report_ref.parent.mkdir()
    report_ref.write_text("{}", encoding="utf-8")
    deploy_ref = tmp_path / "holosoma" / "holosoma"
    deploy_ref.parent.mkdir(parents=True)
    deploy_ref.write_text("", encoding="utf-8")

    pack = build_holosoma_runtime_pack(
        runtime_target_contract={
            "verified_target_ids": ["holosoma_root"],
            "runtime_target_preflight_status": "preflight_ready",
        },
        runtime_layout_contract={
            "preferred_profile_order": ["holosoma_repo"],
            "profiles": [
                {
                    "profile_id": "holosoma_repo",
                    "root": str(deploy_ref.parent.parent),
                    "root_exists": True,
                    "install_preflight_status": "install_ready",
                    "primary_entrypoint_ref": str(deploy_ref),
                    "policy_candidates": ["/missing/policy.ckpt", str(policy_ref)],
                    "deploy_candidates": ["/missing/holosoma", str(deploy_ref)],
                    "data_candidates": ["/missing/eval.json", str(report_ref)],
                }
            ],
        },
        policy_contract={
            "checkpoint_candidates": ["/also/missing/policy.ckpt", str(policy_ref)],
            "deploy_config_candidates": ["/also/missing/holosoma", str(deploy_ref)],
            "runtime_report_candidates": ["/also/missing/eval.json", str(report_ref)],
        },
        deployment_contract={
            "preferred_profile": "holosoma_repo",
            "ready_modes": ["sim_eval"],
        },
        embodiment_context={"motion_clip_paths": []},
    )

    assert pack["primary_policy_ref"] == str(policy_ref)
    assert pack["primary_policy_ref_source"] == "profile.policy_candidates[1]"
    assert pack["policy_candidate_evidence_summary"]["verified_candidate_count"] == 1
    assert pack["primary_deploy_config_ref"] == str(deploy_ref)
    assert pack["primary_deploy_config_ref_source"] == "profile.deploy_candidates[1]"
    assert pack["primary_runtime_report_ref"] == str(report_ref)
    assert pack["runtime_report_candidate_evidence_summary"]["verified_candidate_count"] == 1


def test_holosoma_runtime_pack_derives_motion_sources_from_repo_layout(tmp_path) -> None:
    holosoma_root = tmp_path / "holosoma"
    motion_root = holosoma_root / "src" / "holosoma" / "holosoma" / "data" / "motions"
    policy_root = holosoma_root / "src" / "holosoma_inference" / "holosoma_inference" / "models"
    motion_root.mkdir(parents=True)
    policy_root.mkdir(parents=True)
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    (holosoma_root / "scripts").mkdir()
    motion_clip = motion_root / "g1_walk.npz"
    motion_clip.write_text("x", encoding="utf-8")
    (policy_root / "g1_policy.onnx").write_text("x", encoding="utf-8")

    embodiment_context = {
        "holosoma_root": str(holosoma_root),
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

    assert str(motion_clip) in pack["motion_sources"]
    assert str(motion_clip) in pack["existing_motion_sources"]
    assert pack["primary_policy_ref"].endswith("g1_policy.onnx")

from __future__ import annotations

from src.world_model.sim_synth_physics.runtime_targets import (
    describe_holosoma_runtime_targets,
    describe_isaac_runtime_targets,
)


def test_isaac_runtime_targets_include_context_roots(tmp_path) -> None:
    isaaclab_root = tmp_path / "isaaclab"
    sdk_root = tmp_path / "unitree_sdk2"
    asset_root = tmp_path / "unitree_assets"
    unitree_sim_root = tmp_path / "unitree_sim_isaaclab"
    isaaclab_root.mkdir()
    sdk_root.mkdir()
    asset_root.mkdir()
    unitree_sim_root.mkdir()

    contract = describe_isaac_runtime_targets(
        {
            "isaaclab_root": str(isaaclab_root),
            "unitree_sdk2_root": str(sdk_root),
            "unitree_asset_root": str(asset_root),
            "unitree_sim_isaaclab_root": str(unitree_sim_root),
        }
    )

    assert contract["backend"] == "isaac"
    assert "isaaclab_root" in contract["ready_target_ids"]
    assert "unitree_sdk2_root" in contract["ready_target_ids"]
    assert "unitree_asset_root" in contract["ready_target_ids"]
    assert "unitree_sim_isaaclab_root" in contract["ready_target_ids"]
    assert contract["runtime_targets_ready"] is True


def test_isaac_runtime_targets_surface_install_shape_truth(tmp_path) -> None:
    sdk_root = tmp_path / "unitree_sdk2"
    asset_root = tmp_path / "unitree_assets"
    sim_root = tmp_path / "unitree_sim_isaaclab"
    sdk_root.mkdir()
    asset_root.mkdir()
    sim_root.mkdir()
    (sim_root / "sim_main.py").write_text("", encoding="utf-8")
    (sim_root / "dds").mkdir()

    contract = describe_isaac_runtime_targets(
        {
            "unitree_sdk2_root": str(sdk_root),
            "unitree_asset_root": str(asset_root),
            "unitree_sim_isaaclab_root": str(sim_root),
        }
    )

    sdk_row = next(row for row in contract["targets"] if row["target_id"] == "unitree_sdk2_root")
    asset_row = next(row for row in contract["targets"] if row["target_id"] == "unitree_asset_root")
    sim_row = next(
        row for row in contract["targets"] if row["target_id"] == "unitree_sim_isaaclab_root"
    )

    assert sdk_row["exists"] is True
    assert sdk_row["verification_status"] == "install_shape_missing"
    assert sdk_row["verified"] is False
    assert asset_row["verification_status"] == "install_shape_missing"
    assert asset_row["verified"] is False
    assert sim_row["verification_status"] == "install_shape_ready"
    assert "unitree_sim_isaaclab_root" in contract["verified_target_ids"]
    assert "unitree_sdk2_root" in contract["unverified_required_target_ids"]
    assert contract["runtime_target_preflight_status"] == "preflight_partial"


def test_isaac_runtime_targets_accept_lerobot_alias(tmp_path) -> None:
    sdk_root = tmp_path / "unitree_sdk2"
    asset_root = tmp_path / "unitree_assets"
    lerobot_root = tmp_path / "unitree_lerobot"
    sdk_root.mkdir()
    asset_root.mkdir()
    lerobot_root.mkdir()

    contract = describe_isaac_runtime_targets(
        {
            "unitree_sdk2_root": str(sdk_root),
            "unitree_asset_root": str(asset_root),
            "unitree_lerobot_root": str(lerobot_root),
        }
    )

    assert "unitree_il_lerobot_root" in contract["ready_target_ids"]
    assert "unitree_il_lerobot_root" in contract["preferred_runtime_roots"]


def test_holosoma_runtime_targets_require_motion_root(tmp_path) -> None:
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()

    contract = describe_holosoma_runtime_targets(
        {
            "holosoma_root": str(holosoma_root),
        }
    )

    assert contract["backend"] == "holosoma"
    assert contract["runtime_targets_ready"] is False
    assert "holosoma_motion_root" in contract["missing_required_target_ids"]


def test_isaac_runtime_targets_autodiscover_known_repo_roots(
    tmp_path, monkeypatch
) -> None:
    home = tmp_path / "home"
    code_root = home / "code"
    sim_root = code_root / "unitree_sim_isaaclab"
    sdk_root = code_root / "unitree_sdk2"
    asset_root = code_root / "unitree_assets"
    sim_root.mkdir(parents=True)
    sdk_root.mkdir(parents=True)
    asset_root.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    contract = describe_isaac_runtime_targets({})

    assert "unitree_sim_isaaclab_root" in contract["ready_target_ids"]
    sim_row = next(
        row for row in contract["targets"] if row["target_id"] == "unitree_sim_isaaclab_root"
    )
    assert sim_row["ref"] == str(sim_root.resolve())
    assert sim_row["source"] == "autodiscovery"


def test_holosoma_runtime_targets_autodiscover_repo_root(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    code_root = home / "code"
    holosoma_root = code_root / "holosoma"
    motion_root = code_root / "motions"
    holosoma_root.mkdir(parents=True)
    motion_root.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    contract = describe_holosoma_runtime_targets({})

    assert "holosoma_root" in contract["ready_target_ids"]
    holosoma_row = next(row for row in contract["targets"] if row["target_id"] == "holosoma_root")
    assert holosoma_row["ref"] == str(holosoma_root.resolve())
    assert holosoma_row["source"] == "autodiscovery"


def test_holosoma_runtime_targets_derive_subroots_from_repo(tmp_path) -> None:
    holosoma_root = tmp_path / "holosoma"
    (holosoma_root / "src" / "holosoma" / "holosoma" / "data" / "motions").mkdir(
        parents=True
    )
    (holosoma_root / "src" / "holosoma_inference" / "holosoma_inference" / "models").mkdir(
        parents=True
    )
    (holosoma_root / "src" / "holosoma_retargeting").mkdir(parents=True)
    (holosoma_root / "scripts").mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    (
        holosoma_root / "src" / "holosoma" / "holosoma" / "data" / "motions" / "g1_walk.npz"
    ).write_text("x", encoding="utf-8")
    (
        holosoma_root
        / "src"
        / "holosoma_inference"
        / "holosoma_inference"
        / "models"
        / "g1_policy.onnx"
    ).write_text("x", encoding="utf-8")
    (
        holosoma_root / "src" / "holosoma_retargeting" / "g1_retarget.json"
    ).write_text("{}", encoding="utf-8")

    contract = describe_holosoma_runtime_targets({"holosoma_root": str(holosoma_root)})

    motion_row = next(row for row in contract["targets"] if row["target_id"] == "holosoma_motion_root")
    policy_row = next(row for row in contract["targets"] if row["target_id"] == "holosoma_policy_root")
    retarget_row = next(row for row in contract["targets"] if row["target_id"] == "retargeting_root")

    assert motion_row["source"].endswith("_subpath")
    assert motion_row["verification_status"] == "install_shape_ready"
    assert policy_row["source"].endswith("_subpath")
    assert policy_row["verification_status"] == "install_shape_ready"
    assert retarget_row["source"].endswith("_subpath")
    assert retarget_row["verification_status"] == "install_shape_ready"
    assert "holosoma_motion_root" in contract["verified_target_ids"]


def test_holosoma_runtime_targets_surface_motion_and_retargeting_verification(tmp_path) -> None:
    holosoma_root = tmp_path / "holosoma"
    holosoma_root.mkdir()
    (holosoma_root / "README.md").write_text("holosoma", encoding="utf-8")
    (holosoma_root / "holosoma").mkdir()
    motion_root = tmp_path / "motions"
    motion_root.mkdir()
    (motion_root / "g1_walk.npz").write_text("x", encoding="utf-8")
    retargeting_root = tmp_path / "retargeting"
    retargeting_root.mkdir()
    (retargeting_root / "g1_retarget.yaml").write_text("{}", encoding="utf-8")

    contract = describe_holosoma_runtime_targets(
        {
            "holosoma_root": str(holosoma_root),
            "holosoma_motion_root": str(motion_root),
            "retargeting_root": str(retargeting_root),
        }
    )

    motion_row = next(row for row in contract["targets"] if row["target_id"] == "holosoma_motion_root")
    retarget_row = next(row for row in contract["targets"] if row["target_id"] == "retargeting_root")

    assert motion_row["verification_status"] == "install_shape_ready"
    assert motion_row["primary_marker_ref"].endswith("g1_walk.npz")
    assert retarget_row["verification_status"] == "install_shape_ready"
    assert "holosoma_motion_root" in contract["verified_target_ids"]

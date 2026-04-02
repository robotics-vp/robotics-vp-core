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

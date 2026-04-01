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

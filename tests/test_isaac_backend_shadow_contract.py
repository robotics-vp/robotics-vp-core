from __future__ import annotations

from pathlib import Path

from src.envs.physics.isaac_backend import IsaacBackend


def test_isaac_backend_shadow_loop_emits_episode_summary_and_media_refs(tmp_path: Path) -> None:
    backend = IsaacBackend(
        env_config={
            "env_name": "unitree_shadow_env",
            "task": "humanoid_balance",
            "robot": "unitree_g1_shadow",
            "output_root": str(tmp_path / "isaac_shadow"),
            "max_steps": 3,
            "action_dim": 6,
            "seed": 7,
        },
        num_envs=2,
    )

    obs = backend.reset()
    assert obs["rgb"].shape == (64, 64, 3)
    assert backend.get_current_episode_id(0)
    assert backend.get_current_episode_id(1)

    next_obs, reward, done, info = backend.step([0.1, -0.2, 0.0, 0.3, 0.0, -0.1])
    assert next_obs["action"]["joint_command"][0] == 0.1
    assert reward >= 0.0
    assert done is False
    assert info["backend_mode"] == "shadow_contract"
    assert info["media_refs"]["rgb_path"]

    summary = backend.get_episode_info()
    assert summary.episode_id == backend.get_current_episode_id(0)
    assert summary.media_refs["rgb_path"]
    assert summary.coordination_metrics["backend_mode_shadow"] == 1.0

    batch = backend.get_batch_episode_info()
    assert len(batch) == 2
    assert batch[0].episode_id
    assert batch[1].episode_id

    state = backend.get_state()
    assert state["mode"] == "shadow_contract"
    assert state["step_counts"][0] >= 1
    assert backend.get_observation_space()["joint_positions"] == [6]
    assert backend.get_action_space()["shape"] == [6]


def test_isaac_backend_reset_env_restarts_one_vector_slot(tmp_path: Path) -> None:
    backend = IsaacBackend(
        env_config={
            "env_name": "vector_shadow",
            "output_root": str(tmp_path / "vector_shadow"),
            "max_steps": 2,
            "action_dim": 4,
        },
        num_envs=2,
    )

    backend.reset()
    original_episode = backend.get_current_episode_id(1)
    backend.step([0.0, 0.0, 0.0, 0.0])
    backend.reset_env(1)

    assert backend.get_current_episode_id(1) != original_episode
    refs = backend.get_media_refs(1)
    assert refs["rgb_path"]

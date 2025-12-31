"""Tests for rollout capture helper."""
from pathlib import Path

from src.motor_backend.rollout_capture import (
    EpisodeMetadata,
    finalize_rollout_bundle,
    record_episode_rollout,
    start_rollout_capture,
)


def test_rollout_bundle_roundtrip(tmp_path: Path):
    base_dir = tmp_path / "rollouts"
    scenario_id = "scenario_1"
    start_rollout_capture(scenario_id, base_dir)

    metadata = EpisodeMetadata(
        episode_id="ep_001",
        task_id="task_a",
        robot_family="G1",
        seed=123,
        env_params={"terrain": "flat"},
    )
    rollout = record_episode_rollout(
        scenario_id=scenario_id,
        episode_idx=0,
        metadata=metadata,
        trajectory_data={"state": [0, 1, 2]},
        rgb_frames=None,
        depth_frames=None,
        metrics={"reward": 1.0},
        base_dir=base_dir,
    )
    assert rollout.trajectory_path.exists()

    bundle = finalize_rollout_bundle(scenario_id, base_dir)
    assert bundle.scenario_id == scenario_id
    assert len(bundle.episodes) == 1
    assert bundle.episodes[0].metadata.episode_id == "ep_001"

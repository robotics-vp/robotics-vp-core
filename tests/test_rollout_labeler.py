"""Tests for rollout labeler stub."""
from pathlib import Path

from src.motor_backend.datapacks import DatapackConfig, MotionClipSpec
from src.motor_backend.rollout_capture import EpisodeMetadata, EpisodeRollout, RolloutBundle
from src.vla.rollout_labeler import label_rollouts_with_vla


def test_stub_rollout_labeler_appends_tag(tmp_path: Path):
    base = DatapackConfig(
        id="dp_base",
        description="Base",
        motion_clips=[MotionClipSpec(path="data/clip.npz")],
        tags=["humanoid"],
    )
    rollout = EpisodeRollout(
        metadata=EpisodeMetadata(
            episode_id="ep1",
            task_id="task_a",
            robot_family="G1",
            seed=None,
            env_params={},
        ),
        trajectory_path=tmp_path / "trajectory.npz",
    )
    bundle = RolloutBundle(scenario_id="scenario_1", episodes=[rollout])

    labeled = label_rollouts_with_vla(bundle, base_datapack=base)
    assert labeled
    assert labeled[0].id == "dp_base_vla"
    assert "vla_stub" in labeled[0].tags
    assert "auto_labeled" in labeled[0].tags

"""Tests for rollout labeler."""
from pathlib import Path

from src.motor_backend.datapacks import DatapackConfig, MotionClipSpec
from src.motor_backend.rollout_capture import EpisodeMetadata, EpisodeRollout, RolloutBundle
from src.vla.rollout_labeler import label_rollouts_with_vla


def test_rollout_labeler_appends_tags(tmp_path: Path):
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
    assert "vla_labeled" in labeled[0].tags
    assert "auto_labeled" in labeled[0].tags


def test_rollout_labeler_stub_without_openvla(monkeypatch, tmp_path: Path):
    import src.vla.rollout_labeler as labeler

    monkeypatch.delenv("OPENVLA_ENABLE", raising=False)
    monkeypatch.delenv("VLA_ENABLE", raising=False)
    monkeypatch.setattr(labeler, "_get_openvla_controller", lambda: (_ for _ in ()).throw(AssertionError("OpenVLA not expected")))

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
    bundle = RolloutBundle(scenario_id="scenario_stub", episodes=[rollout])

    labeled = labeler.label_rollouts_with_vla(bundle, base_datapack=base)
    assert labeled
    assert "auto_labeled" in labeled[0].tags


def test_rollout_labeler_openvla_error_fallback(monkeypatch, tmp_path: Path):
    import src.vla.rollout_labeler as labeler

    monkeypatch.setenv("OPENVLA_ENABLE", "1")
    monkeypatch.setattr(labeler, "_get_openvla_controller", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

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
    bundle = RolloutBundle(scenario_id="scenario_error", episodes=[rollout])

    labeled = labeler.label_rollouts_with_vla(bundle, base_datapack=base)
    assert labeled
    assert "vla_error" in labeled[0].tags

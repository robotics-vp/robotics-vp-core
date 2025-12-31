"""Tests for rollout labeler stub."""
from src.motor_backend.datapacks import DatapackConfig, MotionClipSpec
from src.vla.rollout_labeler import StubRolloutLabeler


def test_stub_rollout_labeler_appends_tag():
    base = DatapackConfig(
        id="dp_base",
        description="Base",
        motion_clips=[MotionClipSpec(path="data/clip.npz")],
        tags=["humanoid"],
    )
    labeler = StubRolloutLabeler(tag_suffix="vla_stub")
    labeled = labeler.label_rollouts("rollouts", base_datapack=base)
    assert labeled
    assert labeled[0].id == "dp_base_vla"
    assert "vla_stub" in labeled[0].tags

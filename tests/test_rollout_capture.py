"""Tests for rollout capture helper."""
from pathlib import Path

from src.motor_backend.rollout_capture import RolloutCaptureConfig, capture_rollouts


def test_capture_rollouts_writes_manifest(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    policy_path = run_dir / "model.onnx"
    policy_path.write_text("stub")
    artifact_path = run_dir / "episode.mp4"
    artifact_path.write_text("video")

    result = capture_rollouts(
        policy_id=str(policy_path),
        scenario_id="scenario_1",
        task_id="task_a",
        datapack_ids=["dp1"],
        config=RolloutCaptureConfig(output_dir=tmp_path / "rollouts"),
    )

    assert Path(result.manifest_path).exists()
    assert result.status in {"captured", "no_artifacts_found"}
    assert any("episode.mp4" in path for path in result.artifacts)

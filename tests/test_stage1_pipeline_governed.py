import json

from scripts.run_stage1_pipeline import run_stage1_pipeline


def test_stage1_pipeline_emits_governed_sidecars(tmp_path) -> None:
    stats = run_stage1_pipeline(
        num_videos=1,
        proposals_per_video=1,
        output_dir=str(tmp_path),
    )

    assert stats["total_videos"] == 1
    governed_dir = tmp_path / "governed_video"
    assert governed_dir.exists()
    video_state_files = list(governed_dir.glob("*_video_state_v1.json"))
    assert video_state_files
    payload = json.loads(video_state_files[0].read_text())
    assert payload["version"] == "video_state_snapshot_v1"

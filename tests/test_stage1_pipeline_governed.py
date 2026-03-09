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
    reconstruction_files = list(governed_dir.glob("*_reconstruction_sidecar_v1.json"))
    counterfactual_files = list(governed_dir.glob("*_counterfactual_eval_v1.json"))
    value_target_files = list(governed_dir.glob("*_value_target_pack_v1.json"))
    governance_trace_files = list(governed_dir.glob("*_governance_trace_v1.json"))
    runtime_packet_files = list(governed_dir.glob("*_runtime_packet_v1.json"))
    assert video_state_files
    assert reconstruction_files
    assert counterfactual_files
    assert value_target_files
    assert governance_trace_files
    assert runtime_packet_files
    payload = json.loads(video_state_files[0].read_text())
    assert payload["version"] == "video_state_snapshot_v1"
    reconstruction = json.loads(reconstruction_files[0].read_text())
    assert reconstruction["version"] == "four_d_reconstruction_sidecar_v1"
    counterfactual = json.loads(counterfactual_files[0].read_text())
    assert counterfactual["version"] == "counterfactual_eval_v1"
    value_targets = json.loads(value_target_files[0].read_text())
    assert value_targets["version"] == "value_target_pack_v1"
    runtime_packet = json.loads(runtime_packet_files[0].read_text())
    assert runtime_packet["version"] == "runtime_packet_v1"

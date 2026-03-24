import json

from scripts.run_stage1_pipeline import run_stage1_pipeline
from src.regal.gen_plausibility import PlausibilityThresholds, RegalGenPlausibilityNode


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
    semantic_world_model_files = list(governed_dir.glob("*_semantic_world_model_v1.json"))
    semantic_snapshot_files = list(governed_dir.glob("*_semantic_snapshot_v1.json"))
    orchestrator_advisory_files = list(governed_dir.glob("*_orchestrator_advisory_v1.json"))
    assert video_state_files
    assert reconstruction_files
    assert counterfactual_files
    assert value_target_files
    assert governance_trace_files
    assert runtime_packet_files
    assert semantic_world_model_files
    assert semantic_snapshot_files
    assert orchestrator_advisory_files
    payload = json.loads(video_state_files[0].read_text())
    assert payload["version"] == "video_state_snapshot_v1"
    semantic_world_model = json.loads(semantic_world_model_files[0].read_text())
    assert semantic_world_model["version"] == "semantic_world_model_v1"
    assert semantic_world_model["capability_scores"]
    semantic_snapshot = json.loads(semantic_snapshot_files[0].read_text())
    assert semantic_snapshot["semantic_world_model"]["version"] == "semantic_world_model_v1"
    advisory = json.loads(orchestrator_advisory_files[0].read_text())
    assert advisory["meta_node_weights"]
    reconstruction = json.loads(reconstruction_files[0].read_text())
    assert reconstruction["version"] == "four_d_reconstruction_sidecar_v1"
    counterfactual = json.loads(counterfactual_files[0].read_text())
    assert counterfactual["version"] == "counterfactual_eval_v1"
    value_targets = json.loads(value_target_files[0].read_text())
    assert value_targets["version"] == "value_target_pack_v1"
    runtime_packet = json.loads(runtime_packet_files[0].read_text())
    assert runtime_packet["version"] == "runtime_packet_v1"
    admission_log_path = stats["proposal_admission_log"]
    admission_rows = [json.loads(line) for line in open(admission_log_path, "r", encoding="utf-8") if line.strip()]
    assert admission_rows
    assert admission_rows[0]["execution_preconditions"]["ready"] is True
    assert admission_rows[0]["future_training_signals"]["promotion_trace_complete"] is True
    datapacks = json.loads((tmp_path / "datapacks.json").read_text())
    assert datapacks[0]["episode_metrics"]["execution_preconditions"]["ready"] is True


def test_stage1_pipeline_captures_blocked_proposals_as_negative_supervision(tmp_path) -> None:
    stats = run_stage1_pipeline(
        num_videos=1,
        proposals_per_video=1,
        output_dir=str(tmp_path),
        plausibility_node=RegalGenPlausibilityNode(
            PlausibilityThresholds(min_map_first_quality=0.9, min_plausibility_score=0.95)
        ),
    )

    assert stats["blocked_proposals"] == 1
    admission_rows = [
        json.loads(line)
        for line in open(stats["proposal_admission_log"], "r", encoding="utf-8")
        if line.strip()
    ]
    assert admission_rows[0]["blocked"] is True
    assert admission_rows[0]["execution_work_order"]["decision"] == "capture_negative_supervision"

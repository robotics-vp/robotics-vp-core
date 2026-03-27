import json

from scripts.run_stage1_pipeline import run_stage1_pipeline
from src.regal.gen_plausibility import PlausibilityThresholds, RegalGenPlausibilityNode


def _stage1_manifest_with_real_scene_tracks(tmp_path):
    manifest_path = tmp_path / "stage1_manifest.json"
    payload = {
        "videos": [
            {
                "episode_id": "real_scene_tracks_demo",
                "video_path": "/tmp/demo.mp4",
                "timestamp": 1_700_000_001.0,
                "task_type": "drawer_vase",
                "instruction": "Open the drawer without hitting the vase.",
                "metadata": {
                    "duration_s": 10.0,
                    "success": True,
                    "num_frames": 4,
                    "scene_tracks_backend": "real",
                    "vision_backbone_selected": "real",
                    "teacher_runtime_backend_selected": "unavailable",
                    "scene_tracks_v1": {
                        "track_ids": ["drawer_track", "vase_track"],
                        "entity_types": [0, 0],
                        "class_ids": [0, 1],
                        "class_names": ["drawer", "vase"],
                        "poses_R": [
                            [
                                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            ],
                            [
                                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            ],
                        ],
                        "poses_t": [
                            [[0.0, 0.0, 0.0], [0.2, 0.1, 0.0]],
                            [[0.05, 0.0, 0.0], [0.2, 0.1, 0.0]],
                        ],
                        "scales": [
                            [[1.0, 1.0, 1.0], [0.8, 0.8, 0.8]],
                            [[1.0, 1.0, 1.0], [0.8, 0.8, 0.8]],
                        ],
                        "visibility": [[1.0, 1.0], [1.0, 1.0]],
                        "occlusion": [[0.0, 0.0], [0.0, 0.0]],
                        "ir_loss": [[0.0, 0.0], [0.0, 0.0]],
                        "converged": [[1.0, 1.0], [1.0, 1.0]],
                    },
                },
            }
        ]
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


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
    control_plane_context_files = list(governed_dir.glob("*_control_plane_context_v1.json"))
    assert video_state_files
    assert reconstruction_files
    assert counterfactual_files
    assert value_target_files
    assert governance_trace_files
    assert runtime_packet_files
    assert semantic_world_model_files
    assert semantic_snapshot_files
    assert orchestrator_advisory_files
    assert control_plane_context_files
    payload = json.loads(video_state_files[0].read_text())
    assert payload["version"] == "video_state_snapshot_v1"
    semantic_world_model = json.loads(semantic_world_model_files[0].read_text())
    assert semantic_world_model["version"] == "semantic_world_model_v1"
    assert semantic_world_model["capability_scores"]
    semantic_snapshot = json.loads(semantic_snapshot_files[0].read_text())
    assert semantic_snapshot["semantic_world_model"]["version"] == "semantic_world_model_v1"
    advisory = json.loads(orchestrator_advisory_files[0].read_text())
    assert advisory["meta_node_weights"]
    control_plane_context = json.loads(control_plane_context_files[0].read_text())
    assert control_plane_context["receipt_kind"] == "orchestrator_control_plane_context_v1"
    assert control_plane_context["authority_class"] == "canonical_metadata"
    assert control_plane_context["semantic_world_model_summary"]["world_model_id"]
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
    assert admission_rows[0]["benchmark_gate"]["ready"] is False
    assert admission_rows[0]["execution_work_order"]["recommended_mode"] == "shadow_stage1_datapack"
    assert admission_rows[0]["routing_source"] == "governed_video_world_model"
    datapacks = json.loads((tmp_path / "datapacks.json").read_text())
    assert datapacks[0]["episode_metrics"]["execution_preconditions"]["ready"] is True
    assert datapacks[0]["episode_metrics"]["benchmark_gate"]["ready"] is False
    assert datapacks[0]["attribution"]["tier"] == 0


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


def test_stage1_pipeline_marks_real_grounded_manifest_as_benchmark_ready(tmp_path) -> None:
    manifest_path = _stage1_manifest_with_real_scene_tracks(tmp_path)
    stats = run_stage1_pipeline(
        num_videos=1,
        proposals_per_video=1,
        output_dir=str(tmp_path),
        video_manifest=str(manifest_path),
    )

    assert stats["benchmark_ready_proposals"] == 1
    admission_rows = [
        json.loads(line)
        for line in open(stats["proposal_admission_log"], "r", encoding="utf-8")
        if line.strip()
    ]
    assert admission_rows[0]["benchmark_gate"]["ready"] is True
    assert admission_rows[0]["execution_work_order"]["recommended_mode"] == "stage1_datapack"
    datapacks = json.loads((tmp_path / "datapacks.json").read_text())
    assert datapacks[0]["episode_metrics"]["benchmark_gate"]["ready"] is True
    assert datapacks[0]["attribution"]["tier"] >= 1

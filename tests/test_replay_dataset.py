import json

import numpy as np

from scripts.run_stage1_pipeline import run_stage1_pipeline
from src.dataset_bridges.lerobot_bridge import lerobot_rows_from_replay
from src.dataset_bridges.rlds_bridge import rlds_episode_from_replay
from src.motor_backend.rollout_capture import EpisodeMetadata, record_episode_rollout, start_rollout_capture
from src.replay.dataset import ReplayDatasetBuilder, load_replay_dataset
from src.shadow_runtime.control_plane import run_shadow_control_plane


def test_replay_dataset_builds_from_shadow_run(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=42,
        episodes=2,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )

    bundle = ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(dataset_dir)

    assert (shadow_dir / "shadow_episode_traces.json").exists()
    assert (dataset_dir / "manifest.json").exists()
    assert bundle.manifest.num_episodes == 2
    assert bundle.manifest.num_steps > 0
    assert bundle.manifest.num_windows > 0

    loaded = load_replay_dataset(dataset_dir)
    assert loaded.manifest.dataset_digest == bundle.manifest.dataset_digest
    assert loaded.steps[0].pricing_tick_ref is not None
    assert loaded.episodes[0].provenance["runtime_packet_ref"] == "runtime_packets.json"
    assert loaded.episodes[0].provenance["event_spine_ref"] == "event_spine.json"
    assert loaded.episodes[0].provenance["decision_ledger_ref"] == "decision_ledger.json"
    assert loaded.steps[0].metadata["runtime_packet_id"].startswith("runtime_")
    assert loaded.steps[0].metadata["event_refs"]
    assert loaded.windows[0].metadata["decision_refs"]
    assert bundle.manifest.metadata["sources"][0]["runtime_packet_count"] == 2
    assert bundle.manifest.metadata["sources"][0]["event_count"] >= 10
    assert bundle.manifest.metadata["sources"][0]["decision_count"] >= 8


def test_replay_dataset_builds_from_workcell_episode_log(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    episode_log_path = tmp_path / "episode_log.json"
    dataset_dir = tmp_path / "replay_from_episode_log"
    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=42,
        episodes=1,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    trace_payload = json.loads((shadow_dir / "shadow_episode_traces.json").read_text())
    episode_log_path.write_text(json.dumps(trace_payload["episodes"][0]["episode_log"]), encoding="utf-8")

    bundle = ReplayDatasetBuilder().add_workcell_episode_log(episode_log_path).write(dataset_dir)

    assert bundle.manifest.num_episodes == 1
    assert bundle.manifest.num_steps > 0
    assert bundle.steps[0].task_id == "shadow_kitting"


def test_replay_dataset_builds_from_rollout_bundle_with_provenance(tmp_path):
    base_dir = tmp_path / "rollouts"
    scenario_id = "scenario_shadow"
    start_rollout_capture(scenario_id, base_dir)
    record_episode_rollout(
        scenario_id=scenario_id,
        episode_idx=0,
        metadata=EpisodeMetadata(
            episode_id="ep_rollout_001",
            task_id="shadow_kitting",
            robot_family="sim_robot",
            seed=7,
            env_params={"config": {"topology_type": "workcell_rollout"}},
        ),
        trajectory_data=[
            {"step": 0, "obs": {"state_vector": [0.0, 1.0]}, "action": {"action_vector": [0.2]}, "info": {"reward": 0.5}},
            {"step": 1, "obs": {"state_vector": [1.0, 2.0]}, "action": {"action_vector": [0.1]}, "done": True, "info": {"reward": 0.6}},
        ],
        rgb_frames=None,
        depth_frames=None,
        metrics={"reward": 1.1, "quality_score": 0.7},
        base_dir=base_dir,
    )
    episode_dir = base_dir / scenario_id / "episode_000"
    scene_tracks_path = episode_dir / "ep_rollout_001_front_scene_tracks_v1.npz"
    np.savez_compressed(
        scene_tracks_path,
        **{
            "scene_tracks_v1/track_ids": np.array(["track_1"], dtype="U16"),
            "scene_tracks_v1/summary_json": np.array(
                [json.dumps({"backend_selected": "passthrough", "training_eligible": False})],
                dtype="U256",
            ),
            "scene_tracks_v1/semantic_summary_json": np.array(
                [json.dumps({"grounding_ready": True, "semantic_density_score": 0.62})],
                dtype="U256",
            ),
        },
    )
    semantic_world_model_path = episode_dir / "ep_rollout_001_semantic_world_model_v1.json"
    semantic_world_model_path.write_text(
        json.dumps(
            {
                "world_model_id": "wm_rollout",
                "topology": {"grounded_track_object_count": 1},
                "capability_scores": {"object_memory": 0.71},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata_path = episode_dir / "metadata.json"
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_payload["scene_tracks_path"] = str(scene_tracks_path.relative_to(tmp_path))
    metadata_payload["semantic_world_model_path"] = str(semantic_world_model_path.relative_to(tmp_path))
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

    dataset_dir = tmp_path / "replay_rollout_dataset"
    bundle = ReplayDatasetBuilder().add_rollout_bundle(base_dir, scenario_id=scenario_id).write(dataset_dir)

    assert bundle.manifest.num_episodes == 1
    assert "rollout_capture_bundle_v1" in bundle.manifest.source_adapters
    assert bundle.episodes[0].provenance["source_adapter"] == "rollout_capture_bundle_v1"
    assert bundle.episodes[0].provenance["scene_tracks_ref"] == str(scene_tracks_path.resolve())
    assert bundle.episodes[0].provenance["semantic_world_model_ref"] == str(semantic_world_model_path.resolve())
    assert bundle.episodes[0].metadata["scene_tracks_non_stub"] is False
    assert bundle.episodes[0].metadata["semantic_memory_grounded"] is True
    assert bundle.episodes[0].metadata["semantic_grounding_non_heuristic"] is False
    assert bundle.manifest.metadata["schema_compatibility"][0]["compatible"] is True


def test_replay_dataset_builds_from_rollout_bundle_with_state_action_trajectory(tmp_path):
    base_dir = tmp_path / "rollouts_state_action"
    scenario_id = "scenario_state_action"
    start_rollout_capture(scenario_id, base_dir)
    record_episode_rollout(
        scenario_id=scenario_id,
        episode_idx=0,
        metadata=EpisodeMetadata(
            episode_id="ep_rollout_state_action",
            task_id="peg_in_hole",
            robot_family="workcell",
            seed=11,
            env_params={"config": {"topology_type": "workcell_rollout"}},
        ),
        trajectory_data={
            "scene_spec": {"workcell_id": "test"},
            "states": [
                {"step": 0, "joint_positions": [0.0, 0.1], "constraint_error": 0.05},
                {"step": 1, "joint_positions": [0.1, 0.2], "constraint_error": 0.02, "done": True},
            ],
            "actions": [
                {"object_id": "end_effector", "delta_position": [0.01, 0.0, -0.01]},
                {"object_id": "end_effector", "delta_position": [0.0, 0.0, -0.01]},
            ],
        },
        rgb_frames=None,
        depth_frames=None,
        metrics={"reward": 0.8},
        base_dir=base_dir,
    )

    bundle = ReplayDatasetBuilder().add_rollout_bundle(base_dir, scenario_id=scenario_id).build()

    assert bundle.manifest.num_episodes == 1
    assert bundle.manifest.num_steps == 2
    assert bundle.steps[0].step_idx == 0
    assert bundle.steps[1].done is True


def test_replay_dataset_builds_from_rehydrated_bridge_exports(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "rehydrated_dataset"
    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=42,
        episodes=1,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    source_bundle = ReplayDatasetBuilder().add_shadow_run(shadow_dir).build()
    episode = source_bundle.episodes[0]
    steps = [row for row in source_bundle.steps if row.episode_id == episode.episode_id]
    rlds_payload = rlds_episode_from_replay(episode, steps)
    lerobot_rows = lerobot_rows_from_replay(episode, steps)

    bundle = (
        ReplayDatasetBuilder()
        .add_rlds_episode(rlds_payload)
        .add_lerobot_rows(lerobot_rows)
        .write(dataset_dir)
    )

    assert bundle.manifest.num_episodes == 2
    assert bundle.manifest.metadata["execution_precondition_summary"]["ready_count"] == 2
    assert "rlds_bridge_rehydration_v1" in bundle.manifest.source_adapters
    assert "lerobot_bridge_rehydration_v1" in bundle.manifest.source_adapters
    loaded = load_replay_dataset(dataset_dir)
    assert loaded.episodes[0].metadata["execution_preconditions"]["ready"] is True


def test_replay_dataset_imports_governed_video_admission_log(tmp_path):
    stage1_dir = tmp_path / "stage1"
    dataset_dir = tmp_path / "governed_replay_dataset"
    stats = run_stage1_pipeline(
        num_videos=1,
        proposals_per_video=1,
        output_dir=str(stage1_dir),
    )

    bundle = ReplayDatasetBuilder().add_governed_video_admission_log(
        stats["proposal_admission_log"],
        run_id="governed_import_001",
    ).write(dataset_dir)

    assert bundle.manifest.num_episodes == 1
    loaded = load_replay_dataset(dataset_dir)
    episode = loaded.episodes[0]
    assert episode.provenance["runtime_packet_ref"].endswith("_runtime_packet_v1.json")
    assert episode.provenance["event_spine_ref"].endswith("_event_spine_v1.json")
    assert episode.metadata["source_execution_work_order"]["decision"] == "admit_shadow_datapack"
    assert episode.metadata["execution_preconditions"]["ready"] is True
    summary = bundle.manifest.metadata["execution_precondition_summary"]
    assert summary["satisfied_preconditions"]["signal_bool::promotion_trace_complete"] == 1
    assert summary["satisfied_preconditions"]["signal_bool::replay_roundtrip_complete"] == 1


def test_replay_dataset_imports_semantic_degraded_artifacts(tmp_path):
    artifact_path = tmp_path / "episode_test_semantic_degraded_v1.json"
    artifact_path.write_text(
        json.dumps(
            {
                "episode_id": "episode_test",
                "failure_reason": "track_ids_mismatch",
                "artifact_refs": {
                    "trajectory_path": str(tmp_path / "trajectory.npz"),
                    "teacher_trace_path": str(tmp_path / "teacher_trace_v1.json"),
                },
                "execution_preconditions": {
                    "ready": False,
                    "readiness_score": 0.0,
                    "blocking_preconditions": ["blocked::track_ids_mismatch"],
                    "satisfied_preconditions": ["artifact::trajectory_path"],
                },
                "execution_work_order": {
                    "work_order_id": "work_test",
                    "decision": "capture_negative_supervision",
                    "ready": False,
                },
                "future_training_signals": {
                    "scene_tracks_non_stub": True,
                },
                "version": "semantic_degraded_v1",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    bundle = ReplayDatasetBuilder().add_semantic_degraded_artifacts(tmp_path).build()

    assert bundle.manifest.num_episodes == 1
    episode = bundle.episodes[0]
    assert episode.provenance["teacher_trace_ref"].endswith("teacher_trace_v1.json")
    assert episode.metadata["source_execution_work_order"]["decision"] == "capture_negative_supervision"
    assert episode.metadata["execution_preconditions"]["ready"] is False
    summary = bundle.manifest.metadata["execution_precondition_summary"]
    assert summary["blocked_count"] == 1
    assert summary["satisfied_preconditions"]["signal_bool::teacher_runtime_live"] == 1
    assert summary["satisfied_preconditions"]["signal_bool::scene_tracks_non_stub"] == 1

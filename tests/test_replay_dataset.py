import json

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
    result = run_shadow_control_plane(
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

    dataset_dir = tmp_path / "replay_rollout_dataset"
    bundle = ReplayDatasetBuilder().add_rollout_bundle(base_dir, scenario_id=scenario_id).write(dataset_dir)

    assert bundle.manifest.num_episodes == 1
    assert "rollout_capture_bundle_v1" in bundle.manifest.source_adapters
    assert bundle.episodes[0].provenance["source_adapter"] == "rollout_capture_bundle_v1"
    assert bundle.manifest.metadata["schema_compatibility"][0]["compatible"] is True

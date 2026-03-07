import json

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

from __future__ import annotations

from pathlib import Path

import numpy as np


def test_scene_tracks_from_workcell_datapack(tmp_path: Path) -> None:
    from src.envs.workcell_env import WorkcellEnv
    from src.envs.workcell_env.config import WorkcellEnvConfig
    from src.motor_backend.rollout_capture import EpisodeMetadata, record_episode_rollout, start_rollout_capture
    from src.ontology.store import OntologyStore
    from src.ontology.datapack_registry import get_latest_scene_tracks_artifact
    from src.vision.scene_ir_tracker.io.scene_tracks_runner import run_scene_tracks

    config = WorkcellEnvConfig(num_parts=2, max_steps=5)
    env = WorkcellEnv(config=config, seed=123)
    env.reset(seed=123, episode_id="ep_scene_tracks")

    states = []
    actions = []
    for _ in range(3):
        action = {"object_id": "end_effector", "delta_position": (0.01, 0.0, 0.0)}
        env.step(action)
        states.append(env.physics_adapter.get_state())
        actions.append(action)

    trajectory_data = {
        "scene_spec": env.scene_spec.to_dict(),
        "states": states,
        "actions": actions,
    }

    scenario_id = "scene_tracks_test"
    base_dir = tmp_path / "rollouts"
    start_rollout_capture(scenario_id, base_dir)
    episode_meta = EpisodeMetadata(
        episode_id="ep_scene_tracks",
        task_id="workcell_task",
        robot_family="workcell",
        seed=123,
        env_params={"config": config.to_dict(), "scene_spec": env.scene_spec.to_dict()},
    )
    record_episode_rollout(
        scenario_id=scenario_id,
        episode_idx=0,
        metadata=episode_meta,
        trajectory_data=trajectory_data,
        rgb_frames=None,
        depth_frames=None,
        metrics={},
        base_dir=base_dir,
    )

    episode_dir = base_dir / scenario_id / "episode_000"
    ontology_root = tmp_path / "ontology"
    output_dir = tmp_path / "scene_tracks_out"

    result = run_scene_tracks(
        datapack_path=episode_dir,
        output_path=output_dir,
        seed=123,
        max_frames=10,
        camera="front",
        mode="vector_proxy",
        ontology_root=ontology_root,
        min_quality=0.1,
    )

    assert result.scene_tracks_path.exists()
    assert np.isfinite(result.quality.quality_score)
    assert result.quality.quality_score >= 0.1

    store = OntologyStore(root_dir=str(ontology_root))
    latest = get_latest_scene_tracks_artifact(store, datapack_id="ep_scene_tracks")
    assert latest is not None
    assert latest.get("path") == str(result.scene_tracks_path)

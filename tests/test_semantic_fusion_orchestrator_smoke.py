from __future__ import annotations

import numpy as np

from src.motor_backend.rollout_capture import EpisodeMetadata, EpisodeRollout, RolloutBundle
from src.orchestrator.semantic_fusion_runner import run_semantic_fusion_for_rollouts


def test_semantic_fusion_runner_smoke(tmp_path) -> None:
    episode_dir = tmp_path / "episode_000"
    episode_dir.mkdir(parents=True, exist_ok=True)

    track_ids = np.array(["track_a", "track_b"], dtype="U32")
    trajectory_payload = {"scene_tracks_v1": {"track_ids": track_ids}}
    trajectory_path = episode_dir / "trajectory.npz"
    np.savez_compressed(trajectory_path, trajectory=trajectory_payload)

    T, K, C = 2, 2, 2
    map_semantics = np.array(
        [
            [[0.8, 0.2], [0.3, 0.7]],
            [[0.6, 0.4], [0.4, 0.6]],
        ],
        dtype=np.float32,
    )
    map_stability = np.ones((T, K), dtype=np.float32)
    map_first_payload = {
        "map_first_supervision_v1/evidence_map_semantics": map_semantics,
        "map_first_supervision_v1/evidence_map_stability": map_stability,
        "map_first_supervision_v1/evidence_geom_residual": np.zeros((T, K), dtype=np.float32),
        "map_first_supervision_v1/evidence_occlusion": np.zeros((T, K), dtype=np.float32),
        "map_first_supervision_v1/evidence_dynamics_score": np.zeros((T, K), dtype=np.float32),
    }
    map_first_path = episode_dir / "trajectory_map_first_v1.npz"
    np.savez_compressed(map_first_path, **map_first_payload)

    vla_payload = {
        "vla_semantic_evidence_v1/version": np.array(["v1"], dtype="U8"),
        "vla_semantic_evidence_v1/class_probs": np.array(
            [
                [[0.2, 0.8], [0.7, 0.3]],
                [[0.4, 0.6], [0.5, 0.5]],
            ],
            dtype=np.float32,
        ),
        "vla_semantic_evidence_v1/track_ids": track_ids,
    }
    vla_path = episode_dir / "trajectory_vla_semantic_evidence_v1.npz"
    np.savez_compressed(vla_path, **vla_payload)

    metadata = EpisodeMetadata(
        episode_id="episode_test",
        task_id="task",
        robot_family="robot",
        seed=0,
        env_params={},
    )
    episode = EpisodeRollout(metadata=metadata, trajectory_path=trajectory_path, metrics={})
    bundle = RolloutBundle(scenario_id="scenario_test", episodes=[episode])

    summaries = run_semantic_fusion_for_rollouts(bundle, summary_path=tmp_path / "semantic_fusion_summary.jsonl")
    assert len(summaries) == 1

    fusion_path = episode_dir / "episode_test_semantic_fusion_v1.npz"
    assert fusion_path.exists()
    data = dict(np.load(fusion_path, allow_pickle=False))
    prefix = "semantic_fusion_v1/"

    assert f"{prefix}fused_class_probs" in data
    assert f"{prefix}fused_confidence" in data
    assert f"{prefix}chosen_policy_id" in data

    fused = data[f"{prefix}fused_class_probs"]
    assert fused.shape == (T, K, C)
    assert np.allclose(np.sum(fused, axis=-1), 1.0, atol=1e-4)

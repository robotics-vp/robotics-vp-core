from __future__ import annotations

from unittest import mock

import numpy as np

from src.motor_backend.rollout_capture import EpisodeMetadata, EpisodeRollout, RolloutBundle
from src.orchestrator.semantic_fusion import fuse_semantic_evidence_mvp
from src.orchestrator.semantic_fusion_runner import run_semantic_fusion_for_rollouts


def _build_rollout_bundle(tmp_path):
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
    geom_residual = np.zeros((T, K), dtype=np.float32)
    occlusion = np.zeros((T, K), dtype=np.float32)
    dynamic_evidence = np.zeros((T, K), dtype=np.float32)
    map_first_payload = {
        "map_first_supervision_v1/evidence_map_semantics": map_semantics,
        "map_first_supervision_v1/evidence_map_stability": map_stability,
        "map_first_supervision_v1/evidence_geom_residual": geom_residual,
        "map_first_supervision_v1/evidence_occlusion": occlusion,
        "map_first_supervision_v1/evidence_dynamics_score": dynamic_evidence,
    }
    map_first_path = episode_dir / "trajectory_map_first_v1.npz"
    np.savez_compressed(map_first_path, **map_first_payload)

    metadata = EpisodeMetadata(
        episode_id="episode_test",
        task_id="task",
        robot_family="robot",
        seed=0,
        env_params={},
    )
    episode = EpisodeRollout(metadata=metadata, trajectory_path=trajectory_path, metrics={})
    bundle = RolloutBundle(scenario_id="scenario_test", episodes=[episode])
    return bundle, episode_dir, {
        "map_semantics": map_semantics,
        "map_stability": map_stability,
        "geom_residual": geom_residual,
        "occlusion": occlusion,
        "dynamic_evidence": dynamic_evidence,
    }


def _expected_metrics(inputs):
    fusion_result = fuse_semantic_evidence_mvp(
        vla_class_probs=None,
        vla_confidence=None,
        map_semantics=inputs["map_semantics"],
        map_stability=inputs["map_stability"],
        geom_residual=inputs["geom_residual"],
        occlusion=inputs["occlusion"],
        dynamic_evidence=inputs["dynamic_evidence"],
        num_classes=int(inputs["map_semantics"].shape[-1]),
    )
    confidence_mean = float(np.mean(fusion_result.fused_confidence))
    disagreement_mean = float(fusion_result.diagnostics.get("disagreement_mean", 0.0)) if fusion_result.diagnostics else 0.0
    return confidence_mean, disagreement_mean


def test_emit_flag_default_on(tmp_path) -> None:
    bundle, episode_dir, _ = _build_rollout_bundle(tmp_path)

    with mock.patch("src.orchestrator.semantic_fusion_runner.np.savez_compressed") as save_mock:
        summaries = run_semantic_fusion_for_rollouts(bundle)

    assert save_mock.called
    assert len(summaries) == 1
    summary = summaries[0]
    expected_path = episode_dir / "episode_test_semantic_fusion_v1.npz"
    assert summary["semantic_fusion_path"] == str(expected_path)
    assert "semantic_fusion_keys" in summary
    assert "semantic_fusion_prefix" in summary


def test_emit_flag_disabled_no_artifacts(tmp_path) -> None:
    bundle, _, inputs = _build_rollout_bundle(tmp_path)

    with mock.patch("src.orchestrator.semantic_fusion_runner.np.savez_compressed") as save_mock, mock.patch(
        "src.orchestrator.semantic_fusion_runner._update_episode_metadata"
    ) as update_mock:
        summaries = run_semantic_fusion_for_rollouts(bundle, emit_semantic_fusion=False)

    assert not save_mock.called
    assert not update_mock.called
    assert len(summaries) == 1
    summary = summaries[0]
    assert "semantic_fusion_path" not in summary
    assert "semantic_fusion_keys" not in summary
    assert "semantic_fusion_prefix" not in summary
    expected_confidence_mean, expected_disagreement_mean = _expected_metrics(inputs)
    assert np.isclose(summary["semantic_fusion_confidence_mean"], expected_confidence_mean)
    assert np.isclose(summary["semantic_disagreement_vla_vs_map"], expected_disagreement_mean)


def test_emit_flag_summaries_identical(tmp_path) -> None:
    bundle, _, _ = _build_rollout_bundle(tmp_path)

    with mock.patch("src.orchestrator.semantic_fusion_runner.np.savez_compressed"):
        summaries_emit = run_semantic_fusion_for_rollouts(bundle)
        summaries_no_emit = run_semantic_fusion_for_rollouts(bundle, emit_semantic_fusion=False)

    assert len(summaries_emit) == 1 and len(summaries_no_emit) == 1
    for key in ("semantic_fusion_confidence_mean", "semantic_disagreement_vla_vs_map"):
        assert np.isclose(summaries_emit[0][key], summaries_no_emit[0][key])

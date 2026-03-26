import json

import numpy as np

from src.training.synthetic_branch_corpus import (
    branch_priority_multiplier,
    build_synthetic_branch_training_policy,
    load_synthetic_branch_corpus,
)


def _write_branch_corpus(tmp_path, *, with_metadata: bool = True, with_gap_labels: bool = True):
    corpus_path = tmp_path / "branches.npz"
    np.savez(
        corpus_path,
        n_branches=np.array(2),
        objective_dim=np.array(4),
        branch_0_z_sequence=np.array([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]], dtype=np.float32),
        branch_0_actions=np.array([[0.1], [0.2]], dtype=np.float32),
        branch_0_source_episode=np.array(0),
        branch_0_source_timestep=np.array(1),
        branch_0_trust_score=np.array(0.92),
        branch_0_std_ratio=np.array(1.0),
        branch_0_brick_id=np.array(0),
        branch_0_objective_vector=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        branch_0_branch_value=np.array(0.8),
        branch_1_z_sequence=np.array([[0.2, 0.3], [0.3, 0.4], [0.4, 0.5]], dtype=np.float32),
        branch_1_actions=np.array([[0.3], [0.1]], dtype=np.float32),
        branch_1_source_episode=np.array(1),
        branch_1_source_timestep=np.array(0),
        branch_1_trust_score=np.array(0.88),
        branch_1_std_ratio=np.array(0.95),
        branch_1_brick_id=np.array(1),
        branch_1_objective_vector=np.array([0.5, 0.5, 0.0, 1.0], dtype=np.float32),
        branch_1_branch_value=np.array(0.4),
    )
    if with_metadata:
        (tmp_path / "branches_metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "synthetic_branch_corpus_metadata_v1",
                    "scene_tracks_backend": "real",
                    "vision_backbone_selected": "real",
                    "semantic_grounding_mode": "non_heuristic",
                    "semantic_memory_grounded": True,
                    "future_training_signals": {"scene_tracks_non_stub": True},
                }
            ),
            encoding="utf-8",
        )
    if with_gap_labels:
        (tmp_path / "branches_gap_labels.json").write_text(
            json.dumps(
                [
                    {
                        "branch_idx": 0,
                        "coverage_gap_contribution": 0.6,
                        "economic_priority": 0.3,
                        "skill_edge": "drawer -> place",
                        "branch_value": 0.8,
                    },
                    {
                        "branch_idx": 1,
                        "coverage_gap_contribution": 0.2,
                        "economic_priority": 0.1,
                        "skill_edge": "grasp -> lift",
                        "branch_value": 0.4,
                    },
                ]
            ),
            encoding="utf-8",
        )
    return corpus_path


def test_synthetic_branch_corpus_loads_readiness_and_policy(tmp_path) -> None:
    corpus_path = _write_branch_corpus(tmp_path, with_metadata=True, with_gap_labels=True)

    corpus = load_synthetic_branch_corpus(corpus_path)
    policy = build_synthetic_branch_training_policy(
        corpus,
        requested_synth_share=0.3,
        requested_econ_weight_scale=1.0,
    )

    assert corpus.summary["branch_count"] == 2
    assert corpus.summary["semantic_gap_labeled"] is True
    assert corpus.execution_preconditions.ready is True
    assert corpus.benchmark_gate.ready is True
    assert policy["effective_synth_share_cap"] == 0.3
    assert policy["benchmark_gate_ready"] is True
    assert branch_priority_multiplier(corpus.branches[0], policy) > 1.0


def test_synthetic_branch_policy_caps_unproven_corpora(tmp_path) -> None:
    corpus_path = _write_branch_corpus(tmp_path, with_metadata=False, with_gap_labels=False)

    corpus = load_synthetic_branch_corpus(corpus_path)
    policy = build_synthetic_branch_training_policy(
        corpus,
        requested_synth_share=0.3,
        requested_econ_weight_scale=1.0,
    )

    assert corpus.summary["metadata_present"] is False
    assert corpus.summary["semantic_gap_labeled"] is False
    assert corpus.benchmark_gate.ready is False
    assert policy["effective_synth_share_cap"] <= 0.1
    assert "branch_metadata_missing" in policy["reasons"]
    assert "semantic_gap_labels_missing" in policy["reasons"]


def test_synthetic_branch_corpus_does_not_promote_passthrough_scene_tracks(tmp_path) -> None:
    corpus_path = _write_branch_corpus(tmp_path, with_metadata=True, with_gap_labels=True)
    metadata_path = tmp_path / "branches_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "synthetic_branch_corpus_metadata_v1",
                "scene_tracks_backend": "passthrough",
                "vision_backbone_selected": "real",
                "semantic_grounding_mode": "heuristic_fallback",
                "semantic_memory_grounded": True,
                "future_training_signals": {
                    "scene_tracks_non_stub": True,
                    "semantic_grounding_non_heuristic": True,
                },
            }
        ),
        encoding="utf-8",
    )

    corpus = load_synthetic_branch_corpus(corpus_path)

    assert corpus.summary["future_training_signals"]["scene_tracks_non_stub"] is False
    assert corpus.summary["future_training_signals"]["semantic_grounding_non_heuristic"] is False
    assert corpus.summary["benchmark_signals"]["scene_tracks_backend_real"] is False
    assert corpus.benchmark_gate.ready is False

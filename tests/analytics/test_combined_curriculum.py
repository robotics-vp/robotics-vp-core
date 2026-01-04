"""
Tests for combined MH × SceneIR curriculum.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analytics.combined_curriculum import (
    compute_curriculum_weights,
    compute_combined_weight,
    normalize_scores,
    sample_episodes_weighted,
    CurriculumWeights,
)


class TestNormalizeScores:
    """Tests for normalize_scores function."""

    def test_normalizes_to_01_range(self):
        """Normalized scores are in [0, 1]."""
        scores = np.array([0.1, 0.5, 0.9])
        normalized = normalize_scores(scores)
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_constant_scores_return_ones(self):
        """Constant scores return all ones."""
        scores = np.array([0.5, 0.5, 0.5])
        normalized = normalize_scores(scores)
        assert np.allclose(normalized, 1.0)


class TestComputeCombinedWeight:
    """Tests for compute_combined_weight function."""

    def test_returns_mh_only_when_scene_ir_missing(self):
        """Returns MH quality when scene_ir is None."""
        weight = compute_combined_weight(0.8, None)
        assert weight == 0.8

    def test_combines_scores_multiplicatively(self):
        """Combines scores with multiplication."""
        weight = compute_combined_weight(0.5, 0.5)
        assert np.isclose(weight, 0.25)


class TestComputeCurriculumWeights:
    """Tests for compute_curriculum_weights function."""

    def test_returns_curriculum_weights_object(self):
        """Returns CurriculumWeights instance."""
        episodes = [{"mh_quality_score": 0.8}]
        result = compute_curriculum_weights(episodes)
        assert isinstance(result, CurriculumWeights)

    def test_weights_sum_to_one(self):
        """Weights sum to 1."""
        episodes = [
            {"mh_quality_score": 0.8, "scene_ir_quality_score": 0.9},
            {"mh_quality_score": 0.5, "scene_ir_quality_score": 0.6},
        ]
        result = compute_curriculum_weights(episodes)
        assert np.isclose(result.weights.sum(), 1.0)

    def test_handles_empty_episodes(self):
        """Handles empty episode list."""
        result = compute_curriculum_weights([])
        assert len(result.weights) == 0

    def test_mh_x_scene_ir_mode_monotonic(self):
        """Higher MH+SceneIR scores give higher weights."""
        episodes = [
            {"mh_quality_score": 0.9, "scene_ir_quality_score": 0.9},  # Best
            {"mh_quality_score": 0.5, "scene_ir_quality_score": 0.5},  # Middle
            {"mh_quality_score": 0.1, "scene_ir_quality_score": 0.1},  # Worst
        ]
        result = compute_curriculum_weights(episodes, mode="mh_x_scene_ir")
        
        # Best episode should have highest weight
        assert result.weights[0] > result.weights[1]
        assert result.weights[1] > result.weights[2]

    def test_fallback_to_mh_when_scene_ir_missing(self):
        """Falls back to MH-only when scene_ir missing."""
        episodes = [
            {"mh_quality_score": 0.8},  # No scene_ir
            {"mh_quality_score": 0.5, "scene_ir_quality_score": 0.6},
        ]
        result = compute_curriculum_weights(episodes, mode="mh_x_scene_ir")
        
        assert result.missing_scene_ir_mask[0] == True
        assert result.missing_scene_ir_mask[1] == False
        # Should still have valid weights
        assert np.all(result.weights > 0)


class TestSampleEpisodesWeighted:
    """Tests for sample_episodes_weighted function."""

    def test_samples_correct_count(self):
        """Samples correct number of episodes."""
        episodes = [{"id": i} for i in range(10)]
        weights = CurriculumWeights(
            weights=np.ones(10) / 10,
            mh_scores=np.ones(10),
            scene_ir_scores=None,
            combined_scores=np.ones(10),
            missing_scene_ir_mask=np.zeros(10, dtype=bool),
        )
        
        sampled = sample_episodes_weighted(episodes, weights, n_samples=5)
        assert len(sampled) == 5

    def test_handles_empty_episodes(self):
        """Handles empty episode list."""
        weights = CurriculumWeights(
            weights=np.array([]),
            mh_scores=np.array([]),
            scene_ir_scores=None,
            combined_scores=np.array([]),
            missing_scene_ir_mask=np.array([], dtype=bool),
        )
        
        sampled = sample_episodes_weighted([], weights, n_samples=5)
        assert len(sampled) == 0

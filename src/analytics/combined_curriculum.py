"""
Combined MH × SceneIR Curriculum.

Computes combined sampling weights from Motion Hierarchy and Scene IR quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CurriculumWeights:
    """Episode weights from combined curriculum.
    
    Attributes:
        weights: (N,) array of sampling weights.
        mh_scores: (N,) Motion Hierarchy quality scores.
        scene_ir_scores: (N,) Scene IR quality scores (or None if missing).
        combined_scores: (N,) combined quality scores.
        missing_scene_ir_mask: (N,) bool, True where scene_ir was missing.
    """
    weights: np.ndarray
    mh_scores: np.ndarray
    scene_ir_scores: Optional[np.ndarray]
    combined_scores: np.ndarray
    missing_scene_ir_mask: np.ndarray


def normalize_scores(
    scores: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Normalize scores to [0, 1] range.
    
    Args:
        scores: Raw scores.
        min_val: Minimum value for clipping.
        max_val: Maximum value for clipping.
        epsilon: Small value to avoid division by zero.
    
    Returns:
        Normalized scores in [0, 1].
    """
    scores = np.clip(scores, min_val, max_val)
    score_min = scores.min()
    score_max = scores.max()
    
    if score_max - score_min < epsilon:
        return np.ones_like(scores)
    
    return (scores - score_min) / (score_max - score_min + epsilon)


def compute_combined_weight(
    mh_quality: float,
    scene_ir_quality: Optional[float],
    mh_weight: float = 0.5,
    scene_ir_weight: float = 0.5,
) -> float:
    """Compute combined weight from MH and Scene IR quality.
    
    Args:
        mh_quality: Motion Hierarchy quality score [0, 1].
        scene_ir_quality: Scene IR quality score [0, 1], or None if missing.
        mh_weight: Weight for MH component.
        scene_ir_weight: Weight for Scene IR component.
    
    Returns:
        Combined weight.
    """
    if scene_ir_quality is None:
        # Fallback to MH-only
        return float(mh_quality)
    
    # Multiplicative combination
    combined = mh_quality * scene_ir_quality
    return float(combined)


def compute_curriculum_weights(
    episodes: List[Dict[str, Any]],
    mode: str = "mh_x_scene_ir",
    w_min: float = 0.1,
    w_max: float = 1.0,
    epsilon: float = 0.01,
) -> CurriculumWeights:
    """Compute curriculum sampling weights for episodes.
    
    Args:
        episodes: List of episode metadata dicts with quality scores.
        mode: Curriculum mode ("mh_only", "scene_ir_only", "mh_x_scene_ir").
        w_min: Minimum weight (clamp floor).
        w_max: Maximum weight (clamp ceiling).
        epsilon: Small value added to avoid zero weights.
    
    Returns:
        CurriculumWeights with computed weights.
    """
    n = len(episodes)
    
    if n == 0:
        return CurriculumWeights(
            weights=np.array([]),
            mh_scores=np.array([]),
            scene_ir_scores=None,
            combined_scores=np.array([]),
            missing_scene_ir_mask=np.array([], dtype=bool),
        )
    
    # Extract scores
    mh_scores = np.zeros(n, dtype=np.float32)
    scene_ir_scores = np.zeros(n, dtype=np.float32)
    missing_scene_ir = np.zeros(n, dtype=bool)
    
    for i, ep in enumerate(episodes):
        # MH quality
        mh_scores[i] = ep.get("mh_quality_score", ep.get("motion_hierarchy_quality", 0.5))
        
        # Scene IR quality
        sir = ep.get("scene_ir_quality_score")
        if sir is None:
            sir = ep.get("scene_ir_summary", {}).get("quality_score")
        
        if sir is not None:
            scene_ir_scores[i] = float(sir)
        else:
            missing_scene_ir[i] = True
            scene_ir_scores[i] = 0.5  # Default when missing
    
    # Normalize scores
    mh_norm = normalize_scores(mh_scores)
    scene_ir_norm = normalize_scores(scene_ir_scores)
    
    # Compute combined scores based on mode
    if mode == "mh_only":
        combined = mh_norm.copy()
    elif mode == "scene_ir_only":
        combined = scene_ir_norm.copy()
        # Where scene_ir is missing, fallback to 0.5
        combined[missing_scene_ir] = 0.5
    elif mode == "mh_x_scene_ir":
        combined = mh_norm * scene_ir_norm
        # Where scene_ir is missing, use mh_only
        combined[missing_scene_ir] = mh_norm[missing_scene_ir]
    else:
        raise ValueError(f"Unknown curriculum mode: {mode}")
    
    # Clamp and add epsilon
    weights = np.clip(combined + epsilon, w_min, w_max)
    
    # Normalize weights to sum to 1
    weights = weights / weights.sum()
    
    return CurriculumWeights(
        weights=weights,
        mh_scores=mh_scores,
        scene_ir_scores=scene_ir_scores if not np.all(missing_scene_ir) else None,
        combined_scores=combined,
        missing_scene_ir_mask=missing_scene_ir,
    )


def sample_episodes_weighted(
    episodes: List[Dict[str, Any]],
    weights: CurriculumWeights,
    n_samples: int,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict[str, Any]]:
    """Sample episodes according to curriculum weights.
    
    Args:
        episodes: List of episode metadata.
        weights: Curriculum weights.
        n_samples: Number of samples to draw.
        rng: Random state for reproducibility.
    
    Returns:
        List of sampled episodes.
    """
    if len(episodes) == 0:
        return []
    
    rng = rng or np.random.RandomState()
    indices = rng.choice(len(episodes), size=n_samples, replace=True, p=weights.weights)
    return [episodes[i] for i in indices]

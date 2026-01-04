"""Curriculum weighting utilities based on motion hierarchy summaries."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def compute_episode_motion_difficulty(summary: Dict[str, Any]) -> float:
    """
    Given a motion_hierarchy_summary dict, return a scalar difficulty score.
    """
    if not isinstance(summary, dict):
        return 0.0
    return float(summary.get("structural_difficulty", 0.0))


def _normalize_weights(weights: np.ndarray, eps: float) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= eps:
        if weights.size == 0:
            return weights.astype(np.float32)
        return (np.ones_like(weights) / float(weights.size)).astype(np.float32)
    return (weights / total).astype(np.float32)


def _fill_missing(values: List[float], eps: float) -> np.ndarray:
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        mean_val = float(np.nanmean(arr))
        if not np.isfinite(mean_val) or mean_val <= eps:
            mean_val = 1.0
        arr = np.where(nan_mask, mean_val, arr)
    return arr


def compute_curriculum_weights_from_motion_hierarchy(
    summaries: List[Dict[str, Any]],
    mode: str = "focus_hard",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Given per-episode motion hierarchy summaries, return sampling weights.

    modes:
      - "focus_hard": weight ∝ difficulty
      - "focus_easy": weight ∝ 1 / (difficulty + eps)
      - "uniform_plus_hard_tail": uniform with upweighted hard tail
      - "focus_plausible": weight ∝ motion_quality_score
    """
    if not summaries:
        return np.zeros((0,), dtype=np.float32)

    if mode == "focus_plausible":
        quality_values = []
        for summary in summaries:
            if isinstance(summary, dict):
                if "motion_quality_score" in summary:
                    quality_values.append(float(summary.get("motion_quality_score", 1.0)))
                elif "plausibility_score" in summary:
                    quality_values.append(float(summary.get("plausibility_score", 1.0)))
                else:
                    quality_values.append(np.nan)
            else:
                quality_values.append(np.nan)
        d = _fill_missing(quality_values, eps)
    else:
        difficulty_values = []
        for summary in summaries:
            if isinstance(summary, dict) and "structural_difficulty" in summary:
                difficulty_values.append(float(summary.get("structural_difficulty", 0.0)))
            else:
                difficulty_values.append(np.nan)
        d = _fill_missing(difficulty_values, eps)

    d = np.clip(d, a_min=0.0, a_max=None)
    mean_val = float(np.mean(d)) if d.size else 0.0

    if mode == "focus_hard":
        weights = d / (mean_val + eps)
        weights = np.clip(weights, 0.0, 10.0)
        return _normalize_weights(weights, eps)

    if mode == "focus_easy":
        weights = 1.0 / (d + eps)
        weights = np.clip(weights, 0.0, 10.0)
        return _normalize_weights(weights, eps)

    if mode == "uniform_plus_hard_tail":
        weights = np.ones_like(d)
        q = float(np.quantile(d, 0.8))
        weights = np.where(d >= q, weights * 3.0, weights)
        return _normalize_weights(weights, eps)

    if mode == "focus_plausible":
        weights = d / (mean_val + eps)
        weights = np.clip(weights, 0.0, 10.0)
        return _normalize_weights(weights, eps)

    raise ValueError(f"Unknown curriculum mode: {mode}")

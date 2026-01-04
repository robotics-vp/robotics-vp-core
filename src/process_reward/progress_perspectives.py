"""
Progress Perspectives for Process Reward.

Computes three progress potential candidates from hop predictions:
- Phi_I: Incremental accumulation (clipped cumsum of hops).
- Phi_F: Forward anchored (init -> t).
- Phi_B: Backward anchored (t -> goal), mapped to [0, 1].

These are combined by FusionNet to produce the final Phi_star.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.process_reward.schemas import (
    ProcessRewardConfig,
    ProgressPerspectives,
    EpisodeFeatures,
)


def compute_phi_incremental(
    hops: np.ndarray,
    uncertainties: np.ndarray,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute incremental progress Phi_I.

    Phi_I[t] = clipped cumsum of hops from 0 to t-1.
    Phi_I[0] = 0 (no progress at start).

    Args:
        hops: (T-1,) array of hop predictions.
        uncertainties: (T-1,) array of hop uncertainties.
        clip_min: Minimum value for Phi.
        clip_max: Maximum value for Phi.

    Returns:
        (phi_I, conf_I) arrays of shape (T,).
    """
    T_minus_1 = len(hops)
    T = T_minus_1 + 1

    # Cumulative sum of hops
    cumsum = np.cumsum(np.concatenate([[0.0], hops]))  # (T,)

    # Normalize to [0, 1] range
    # Assuming total progress should be ~1 at goal
    max_abs = max(abs(cumsum.max()), abs(cumsum.min()), 1e-6)
    phi_I = cumsum / max_abs

    # Shift to [0, 1] if there are negative values
    phi_I = (phi_I - phi_I.min()) / (phi_I.max() - phi_I.min() + 1e-6)

    # Clip
    phi_I = np.clip(phi_I, clip_min, clip_max)

    # Confidence decreases with uncertainty
    # conf_I[t] = mean confidence of hops up to t
    if T_minus_1 > 0:
        conf_hops = np.exp(-np.abs(uncertainties))  # Convert log var to confidence
        cumconf = np.cumsum(np.concatenate([[1.0], conf_hops]))
        counts = np.arange(1, T + 1)
        conf_I = cumconf / counts
    else:
        conf_I = np.ones(T, dtype=np.float32)

    return phi_I.astype(np.float32), conf_I.astype(np.float32)


def compute_phi_forward(
    episode_features: EpisodeFeatures,
    goal_features: np.ndarray,
    hops: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute forward-anchored progress Phi_F.

    Phi_F[t] = 1 - (distance from init to t) / (distance from init to goal).
    Measures how far we've come from the initial state.

    Args:
        episode_features: Episode features.
        goal_features: Goal state features.
        hops: Optional hops (not used in distance version).
        uncertainties: Optional uncertainties for confidence.
        clip_min: Minimum value for Phi.
        clip_max: Maximum value for Phi.

    Returns:
        (phi_F, conf_F) arrays of shape (T,).
    """
    T = len(episode_features.frame_features)
    if T == 0:
        return np.array([0.0]), np.array([1.0])

    init_features = episode_features.init_features.pooled

    # Distance from init at each timestep
    dist_from_init = np.zeros(T, dtype=np.float32)
    for t in range(T):
        dist_from_init[t] = np.linalg.norm(
            episode_features.frame_features[t].pooled - init_features
        )

    # Distance from init to goal
    dist_init_goal = np.linalg.norm(goal_features - init_features)
    if dist_init_goal < 1e-6:
        # Already at goal
        return np.ones(T, dtype=np.float32), np.ones(T, dtype=np.float32)

    # Phi_F = normalized progress from init
    # At init: dist_from_init=0, so Phi_F=0
    # As we move towards goal, dist_from_init increases
    phi_F = dist_from_init / dist_init_goal

    # Clip
    phi_F = np.clip(phi_F, clip_min, clip_max)

    # Confidence based on IR quality from features
    conf_F = np.zeros(T, dtype=np.float32)
    for t in range(T):
        ir_stats = episode_features.frame_features[t].ir_stats
        pct_converged = ir_stats.get("pct_converged", 1.0)
        mean_ir = ir_stats.get("mean_ir_loss", 0.0)
        conf_F[t] = pct_converged * np.exp(-mean_ir)

    # Incorporate hop uncertainties if provided
    if uncertainties is not None and len(uncertainties) > 0:
        # Use cumulative uncertainty
        hop_conf = np.exp(-np.abs(uncertainties))
        cumconf = np.cumsum(np.concatenate([[1.0], hop_conf]))
        counts = np.arange(1, T + 1)
        avg_hop_conf = cumconf / counts
        conf_F = 0.5 * conf_F + 0.5 * avg_hop_conf

    return phi_F.astype(np.float32), conf_F.astype(np.float32)


def compute_phi_backward(
    episode_features: EpisodeFeatures,
    goal_features: np.ndarray,
    goal_frame_idx: Optional[int] = None,
    hops: Optional[np.ndarray] = None,
    uncertainties: Optional[np.ndarray] = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute backward-anchored progress Phi_B.

    Phi_B[t] = 1 - (distance from t to goal) / (distance from init to goal).
    Measures how close we are to the goal.

    When goal is unknown, falls back to Phi_F + incremental.

    Args:
        episode_features: Episode features.
        goal_features: Goal state features.
        goal_frame_idx: Optional goal frame index.
        hops: Optional hops for fallback when goal unknown.
        uncertainties: Optional uncertainties for confidence.
        clip_min: Minimum value for Phi.
        clip_max: Maximum value for Phi.

    Returns:
        (phi_B, conf_B) arrays of shape (T,).
    """
    T = len(episode_features.frame_features)
    if T == 0:
        return np.array([0.0]), np.array([1.0])

    init_features = episode_features.init_features.pooled

    # Distance from init to goal
    dist_init_goal = np.linalg.norm(goal_features - init_features)
    if dist_init_goal < 1e-6:
        # Already at goal
        return np.ones(T, dtype=np.float32), np.ones(T, dtype=np.float32)

    # Distance to goal at each timestep
    dist_to_goal = np.zeros(T, dtype=np.float32)
    for t in range(T):
        dist_to_goal[t] = np.linalg.norm(
            episode_features.frame_features[t].pooled - goal_features
        )

    # Phi_B = 1 - (dist to goal / max dist)
    # At goal: dist_to_goal=0, so Phi_B=1
    # At init: dist_to_goal=dist_init_goal, so Phi_B=0
    phi_B = 1.0 - (dist_to_goal / dist_init_goal)

    # Clip
    phi_B = np.clip(phi_B, clip_min, clip_max)

    # Confidence: higher when closer to goal and when goal is known
    conf_B = np.zeros(T, dtype=np.float32)
    goal_known_factor = 1.0 if goal_frame_idx is not None else 0.7

    for t in range(T):
        # Base confidence from visibility
        vis_stats = episode_features.frame_features[t].visibility_stats
        vis_conf = vis_stats.get("mean_visibility", 1.0)

        # Confidence increases as we approach goal
        proximity_conf = 1.0 - (dist_to_goal[t] / (dist_init_goal + 1e-6))
        proximity_conf = np.clip(proximity_conf, 0.1, 1.0)

        conf_B[t] = vis_conf * proximity_conf * goal_known_factor

    # Incorporate hop uncertainties if provided
    if uncertainties is not None and len(uncertainties) > 0:
        # Use reverse cumulative uncertainty (from goal back)
        hop_conf = np.exp(-np.abs(uncertainties))
        rev_cumconf = np.cumsum(np.concatenate([hop_conf[::-1], [1.0]]))[::-1]
        rev_counts = np.arange(T, 0, -1)
        avg_hop_conf = rev_cumconf / rev_counts
        conf_B = 0.5 * conf_B + 0.5 * avg_hop_conf

    return phi_B.astype(np.float32), conf_B.astype(np.float32)


def compute_all_perspectives(
    episode_features: EpisodeFeatures,
    hops: np.ndarray,
    uncertainties: np.ndarray,
    goal_frame_idx: Optional[int] = None,
    config: Optional[ProcessRewardConfig] = None,
) -> ProgressPerspectives:
    """Compute all three progress perspectives.

    Args:
        episode_features: Episode features.
        hops: (T-1,) array of hop predictions.
        uncertainties: (T-1,) array of hop uncertainties.
        goal_frame_idx: Optional goal frame index.
        config: Optional config for clip values.

    Returns:
        ProgressPerspectives with Phi_I, Phi_F, Phi_B and their confidences.
    """
    clip_min = config.phi_clip_min if config else 0.0
    clip_max = config.phi_clip_max if config else 1.0

    T = len(episode_features.frame_features)

    # Get goal features
    if goal_frame_idx is not None and episode_features.goal_features is not None:
        goal_features = episode_features.goal_features.pooled
    else:
        # Use last frame as proxy for goal
        goal_features = episode_features.frame_features[-1].pooled

    # Compute perspectives
    phi_I, conf_I = compute_phi_incremental(
        hops, uncertainties, clip_min=clip_min, clip_max=clip_max
    )

    phi_F, conf_F = compute_phi_forward(
        episode_features, goal_features, hops, uncertainties,
        clip_min=clip_min, clip_max=clip_max
    )

    phi_B, conf_B = compute_phi_backward(
        episode_features, goal_features, goal_frame_idx, hops, uncertainties,
        clip_min=clip_min, clip_max=clip_max
    )

    # Ensure all arrays have the same length T
    assert len(phi_I) == T, f"phi_I length {len(phi_I)} != T {T}"
    assert len(phi_F) == T, f"phi_F length {len(phi_F)} != T {T}"
    assert len(phi_B) == T, f"phi_B length {len(phi_B)} != T {T}"

    return ProgressPerspectives(
        phi_I=phi_I,
        phi_F=phi_F,
        phi_B=phi_B,
        conf_I=conf_I,
        conf_F=conf_F,
        conf_B=conf_B,
    )


def compute_perspective_disagreement(perspectives: ProgressPerspectives) -> np.ndarray:
    """Compute disagreement between perspectives at each timestep.

    Args:
        perspectives: Three progress perspectives.

    Returns:
        (T,) array of disagreement scores (max abs difference).
    """
    T = len(perspectives.phi_I)
    disagreement = np.zeros(T, dtype=np.float32)

    for t in range(T):
        values = [perspectives.phi_I[t], perspectives.phi_F[t], perspectives.phi_B[t]]
        disagreement[t] = max(values) - min(values)

    return disagreement


def compute_perspective_entropy(perspectives: ProgressPerspectives) -> np.ndarray:
    """Compute entropy of perspective confidences at each timestep.

    Lower entropy means one perspective is more confident than others.

    Args:
        perspectives: Three progress perspectives.

    Returns:
        (T,) array of entropy scores.
    """
    T = len(perspectives.phi_I)
    entropy = np.zeros(T, dtype=np.float32)

    for t in range(T):
        # Normalize confidences to sum to 1
        confs = np.array([
            perspectives.conf_I[t],
            perspectives.conf_F[t],
            perspectives.conf_B[t],
        ])
        confs = confs / (confs.sum() + 1e-8)

        # Compute entropy: -sum(p * log(p))
        confs = np.clip(confs, 1e-8, 1.0)
        entropy[t] = -np.sum(confs * np.log(confs))

    return entropy

"""Metrics helpers for Motion Hierarchy outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import torch


@dataclass
class MotionHierarchySummary:
    mean_tree_depth: float
    mean_branch_factor: float
    residual_mean: float
    residual_std: float
    structural_difficulty: float
    plausibility_score: float


def _as_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if tensor.numel() == 0:
        return None
    return tensor


def _compute_tree_stats(hierarchy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if hierarchy.ndim == 2:
        hierarchy = hierarchy.unsqueeze(0)
    if hierarchy.ndim != 3 or hierarchy.shape[1] != hierarchy.shape[2]:
        raise ValueError(f"hierarchy must be (N, N) or (B, N, N). Got {hierarchy.shape}.")

    B, N, _ = hierarchy.shape
    parent_idx = hierarchy.argmax(dim=-1)
    idx = torch.arange(N, device=hierarchy.device).unsqueeze(0).expand(B, -1)
    self_parent = (parent_idx == idx).to(dtype=hierarchy.dtype)

    child_counts = torch.zeros((B, N), device=hierarchy.device, dtype=hierarchy.dtype)
    ones = torch.ones_like(parent_idx, dtype=hierarchy.dtype)
    child_counts.scatter_add_(1, parent_idx, ones)
    branching = child_counts - self_parent

    depths = torch.zeros((B, N), device=hierarchy.device, dtype=torch.long)
    current = parent_idx.clone()
    for _ in range(N):
        is_root = current == idx
        depths = depths + (~is_root)
        current = torch.where(is_root, current, parent_idx.gather(1, current))

    mean_depth = depths.to(dtype=hierarchy.dtype).mean(dim=1)
    mean_branch = branching.mean(dim=1)
    return mean_depth, mean_branch


def _extract_residual_stats(delta_resid_stats: Dict[str, Any]) -> Tuple[float, float]:
    if not isinstance(delta_resid_stats, dict):
        return 0.0, 0.0

    mean_payload = delta_resid_stats.get("residual_mean")
    if mean_payload is None:
        mean_payload = delta_resid_stats.get("mean")
    std_payload = delta_resid_stats.get("residual_std")
    if std_payload is None:
        std_payload = delta_resid_stats.get("std")

    mean_tensor = _as_tensor(mean_payload)
    std_tensor = _as_tensor(std_payload)

    residual_mean = float(mean_tensor.mean().item()) if mean_tensor is not None else 0.0
    residual_std = float(std_tensor.mean().item()) if std_tensor is not None else 0.0
    return residual_mean, residual_std


def compute_motion_hierarchy_summary_from_stats(
    mean_tree_depth: float,
    mean_branch_factor: float,
    residual_mean: float,
    residual_std: float,
) -> MotionHierarchySummary:
    mean_tree_depth = float(mean_tree_depth)
    mean_branch_factor = float(mean_branch_factor)
    residual_mean = float(residual_mean)
    residual_std = float(residual_std)

    structural_difficulty = (
        mean_tree_depth + 0.5 * (1.0 + mean_branch_factor) + 2.0 * residual_mean
    )
    plausibility_score = float(math.exp(-residual_mean))

    return MotionHierarchySummary(
        mean_tree_depth=mean_tree_depth,
        mean_branch_factor=mean_branch_factor,
        residual_mean=residual_mean,
        residual_std=residual_std,
        structural_difficulty=structural_difficulty,
        plausibility_score=plausibility_score,
    )


def compute_motion_hierarchy_summary_from_raw(
    hierarchy: torch.Tensor,
    delta_resid_stats: Dict[str, Any],
) -> MotionHierarchySummary:
    """
    Compute a MotionHierarchySummary from raw hierarchy adjacency and residual stats.

    Args:
        hierarchy: (N, N) or (B, N, N) parent adjacency (row one-hot).
        delta_resid_stats: Dict with residual stats (mean/std per node or global).
    """
    hierarchy_tensor = _as_tensor(hierarchy)
    if hierarchy_tensor is None:
        raise ValueError("hierarchy must be a non-empty tensor or array")

    mean_depth, mean_branch = _compute_tree_stats(hierarchy_tensor)
    residual_mean, residual_std = _extract_residual_stats(delta_resid_stats)

    return compute_motion_hierarchy_summary_from_stats(
        mean_tree_depth=float(mean_depth.mean().item()),
        mean_branch_factor=float(mean_branch.mean().item()),
        residual_mean=residual_mean,
        residual_std=residual_std,
    )


def compute_motion_plausibility_flags(
    summary: MotionHierarchySummary,
    max_residual_mean: float = 1.5,
    min_plausibility_score: float = 0.1,
) -> Dict[str, Any]:
    """
    Decide if the episode is motion-plausible given MH summary.

    Returns:
        {"is_plausible": bool, "reason": str}
    """
    reasons = []
    if summary.residual_mean > max_residual_mean:
        reasons.append("residual_mean_too_high")
    if summary.plausibility_score < min_plausibility_score:
        reasons.append("plausibility_score_too_low")

    return {
        "is_plausible": not reasons,
        "reason": ",".join(reasons),
    }


def motion_quality_score(summary: MotionHierarchySummary) -> float:
    return float(summary.plausibility_score)

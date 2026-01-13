"""
Canonical RewardBreakdownV1 integration for workcell environments.

Phase 4: Standardized reward component schema for RewardIntegrity checks.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.contracts.schemas import RewardBreakdownV1


def compute_workcell_reward_breakdown(
    *,
    success: bool = False,
    progress: float = 0.0,
    time_cost: float = 0.0,
    error_count: int = 0,
    tolerance_met: bool = False,
    collision_count: int = 0,
    energy_wh: float = 0.0,
    # Task-specific
    items_picked: int = 0,
    items_placed: int = 0,
    items_total: int = 1,
    # Term weights
    sparse_success_weight: float = 1.0,
    dense_progress_weight: float = 0.0,
    time_penalty_weight: float = -0.01,
    error_penalty_weight: float = -0.1,
    tolerance_bonus_weight: float = 0.0,
    collision_penalty_weight: float = -0.05,
    energy_cost_weight: float = -0.001,
) -> RewardBreakdownV1:
    """Compute RewardBreakdownV1 from workcell task state.

    Args:
        success: Whether task completed successfully
        progress: Progress fraction [0, 1]
        time_cost: Time steps consumed
        error_count: Number of errors/constraint violations
        tolerance_met: Whether tolerance requirements met
        collision_count: Number of collisions
        energy_wh: Energy consumed in Wh
        items_picked: Items successfully picked
        items_placed: Items successfully placed
        items_total: Total items in task
        *_weight: Corresponding reward term weights

    Returns:
        RewardBreakdownV1 with standardized components
    """
    # Required components
    task_reward = sparse_success_weight if success else 0.0
    task_reward += dense_progress_weight * progress
    if tolerance_met:
        task_reward += tolerance_bonus_weight

    time_penalty = time_penalty_weight * time_cost
    energy_cost_component = energy_cost_weight * energy_wh

    # Standard optional components
    collision_penalty = collision_penalty_weight * collision_count if collision_count > 0 else None
    success_bonus = sparse_success_weight if success else None

    # Task-specific as custom
    pick_progress = items_picked / max(items_total, 1)
    place_progress = items_placed / max(items_total, 1)

    grasp_reward = 0.1 * items_picked if items_picked > 0 else None
    place_reward = 0.1 * items_placed if items_placed > 0 else None

    # Constraint violation penalty
    constraint_penalty = error_penalty_weight * error_count if error_count > 0 else None

    return RewardBreakdownV1(
        task_reward=task_reward,
        time_penalty=time_penalty,
        energy_cost=energy_cost_component,
        collision_penalty=collision_penalty,
        success_bonus=success_bonus,
        grasp_reward=grasp_reward,
        place_reward=place_reward,
        constraint_violation_penalty=constraint_penalty,
        custom_components={
            "pick_progress": pick_progress,
            "place_progress": place_progress,
        },
    )


def extract_breakdown_from_info(info: Dict[str, Any]) -> Optional[RewardBreakdownV1]:
    """Extract RewardBreakdownV1 from step info dict if present.

    Args:
        info: Step info dictionary

    Returns:
        RewardBreakdownV1 if extractable, None otherwise
    """
    if "reward_breakdown" in info:
        rb = info["reward_breakdown"]
        if isinstance(rb, RewardBreakdownV1):
            return rb
        elif isinstance(rb, dict):
            return RewardBreakdownV1(**rb)
    return None


__all__ = [
    "compute_workcell_reward_breakdown",
    "extract_breakdown_from_info",
]

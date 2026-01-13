"""
RewardBreakdownV1 implementation for dishwashing environment.

Phase 4: Standardized reward decomposition for regality compliance.
Maps dishwashing-specific metrics to canonical RewardBreakdownV1 components.
"""
from __future__ import annotations

from typing import Optional

from src.contracts.schemas import RewardBreakdownV1


def compute_dishwashing_reward_breakdown(
    # Task completion
    completed: int,
    attempts: int,
    errors: int,
    # Action parameters
    speed: float,
    care: float,
    # Economic metrics
    energy_Wh: float,
    profit: float,
    # Episode context
    mpl_rate: float,
    error_rate: float,
    # Time
    time_step_s: float = 60.0,
    # Reward weights (match reward shaping)
    throughput_weight: float = 1.0,
    error_penalty_weight: float = 2.0,
    energy_penalty_weight: float = 0.1,
) -> RewardBreakdownV1:
    """Compute RewardBreakdownV1 for a dishwashing step.

    Maps dishwashing-specific metrics to canonical components:
    - task_reward: throughput-based reward (completed items)
    - time_penalty: negative component from time cost
    - energy_cost: energy consumption penalty
    - error_penalty: penalty for errors (mapped to collision_penalty)
    - care_bonus: bonus for careful operation

    Args:
        completed: Items completed this step
        attempts: Attempts made this step
        errors: Errors this step
        speed: Action speed parameter [0,1]
        care: Action care parameter [0,1]
        energy_Wh: Energy consumed this step (Wh)
        profit: Profit this step
        mpl_rate: Current MPL rate (items/hour)
        error_rate: Current error rate
        time_step_s: Time step in seconds
        throughput_weight: Weight for throughput reward
        error_penalty_weight: Weight for error penalty
        energy_penalty_weight: Weight for energy penalty

    Returns:
        RewardBreakdownV1 with dishwashing-specific components
    """
    # Task reward: based on throughput (completed items)
    task_reward = completed * throughput_weight

    # Time penalty: proportional to time spent (encourage efficiency)
    time_penalty = -0.01 * (time_step_s / 60.0)  # Small per-minute penalty

    # Energy cost: proportional to energy consumed
    energy_cost = -energy_Wh * energy_penalty_weight

    # Error penalty: mapped to collision_penalty for consistency with workcell
    # In dishwashing, errors are analogous to collisions/failures
    collision_penalty = -errors * error_penalty_weight if errors > 0 else None

    # Care bonus: reward for careful operation (encourages quality)
    # Only applies if care > 0.5 (above baseline)
    care_bonus = care * 0.1 if care > 0.5 else None

    # Speed efficiency: small bonus for efficient speed selection
    # Optimal speed balances throughput and errors
    speed_efficiency = (speed * (1 - error_rate)) * 0.05

    # Grasp reward: map to successful completions
    grasp_reward = completed * 0.1 if completed > 0 else None

    # Place reward: not directly applicable
    place_reward = None

    # Progress reward: based on attempt rate (effort) - goes to custom_components
    progress_reward = attempts * 0.02 if attempts > 0 else 0.0

    # Build custom_components (must be Dict[str, float], no None values)
    custom = {
        "speed_efficiency": speed_efficiency,
        "profit_component": profit * 0.01,
        "progress_reward": progress_reward,
    }
    if care_bonus is not None:
        custom["care_bonus"] = care_bonus

    return RewardBreakdownV1(
        task_reward=task_reward,
        time_penalty=time_penalty,
        energy_cost=energy_cost,
        collision_penalty=collision_penalty,
        grasp_reward=grasp_reward,
        place_reward=place_reward,
        custom_components=custom,
    )


def compute_episode_reward_summary(
    total_completed: int,
    total_errors: int,
    total_energy_Wh: float,
    total_profit: float,
    episode_length: int,
    mpl_episode: float,
    error_rate_episode: float,
) -> RewardBreakdownV1:
    """Compute episode-level reward summary.

    Useful for trajectory audit aggregation.

    Args:
        total_completed: Total items completed
        total_errors: Total errors
        total_energy_Wh: Total energy consumed
        total_profit: Total profit
        episode_length: Number of steps
        mpl_episode: Episode MPL rate
        error_rate_episode: Episode error rate

    Returns:
        RewardBreakdownV1 summarizing episode rewards
    """
    return RewardBreakdownV1(
        task_reward=float(total_completed),
        time_penalty=-episode_length * 0.01,
        energy_cost=-total_energy_Wh * 0.1,
        collision_penalty=-total_errors * 2.0 if total_errors > 0 else None,
        grasp_reward=float(total_completed) * 0.1 if total_completed > 0 else None,
        place_reward=None,
        custom_components={
            "mpl_episode": mpl_episode,
            "error_rate_episode": error_rate_episode,
            "total_profit": total_profit,
        },
    )


__all__ = [
    "compute_dishwashing_reward_breakdown",
    "compute_episode_reward_summary",
]

"""Dishwashing regality extensions package.

Provides RewardBreakdownV1 and TrajectoryAuditV1 for dishwashing environment.
The main DishwashingEnv class remains in src/envs/dishwashing_env.py.
"""
from src.envs.dishwashing_regal.rewards.reward_breakdown import compute_dishwashing_reward_breakdown
from src.envs.dishwashing_regal.trajectory_audit import (
    DishwashingTrajectoryCollector,
    create_dishwashing_trajectory_audit,
)

__all__ = [
    "compute_dishwashing_reward_breakdown",
    "DishwashingTrajectoryCollector",
    "create_dishwashing_trajectory_audit",
]

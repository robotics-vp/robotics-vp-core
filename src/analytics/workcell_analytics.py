"""
Analytics integration for workcell environments.

Provides metrics computation, difficulty correlation, and
economics reporting for manufacturing tasks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.envs.workcell_env.difficulty.difficulty_features import (
    WorkcellDifficultyFeatures,
    compute_difficulty_features,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkcellEpisodeMetrics:
    """Metrics for a single workcell episode."""
    episode_id: str
    task_type: str
    success: bool
    total_reward: float
    steps: int
    time_s: float

    # Task-specific metrics
    items_completed: int = 0
    items_total: int = 0
    errors: int = 0
    tolerance_violations: int = 0

    # Economic metrics
    throughput_per_hour: float = 0.0
    implied_wage: float = 0.0
    quality_score: float = 0.0

    # Difficulty features
    difficulty: Optional[WorkcellDifficultyFeatures] = None


@dataclass
class WorkcellSuiteReport:
    """Aggregate report for workcell suite episodes."""
    num_episodes: int
    success_rate: float
    mean_reward: float
    mean_steps: float
    mean_time_s: float

    # Aggregate task metrics
    total_items_completed: int
    total_items_attempted: int
    completion_rate: float
    error_rate: float

    # Economic aggregates
    mean_throughput_per_hour: float
    mean_implied_wage: float
    mean_quality_score: float

    # Difficulty correlation
    difficulty_success_correlation: float = 0.0
    difficulty_reward_correlation: float = 0.0

    # Per-task-type breakdown
    by_task_type: Dict[str, Dict[str, float]] = field(default_factory=dict)


def compute_episode_metrics(
    episode_id: str,
    task_type: str,
    episode_data: Dict[str, Any],
    time_step_s: float = 1.0,
    price_per_unit: float = 0.30,
    human_wage: float = 18.0,
) -> WorkcellEpisodeMetrics:
    """
    Compute metrics for a single workcell episode.

    Args:
        episode_id: Episode identifier
        task_type: Type of task (kitting, assembly, etc.)
        episode_data: Dict with keys: success, reward, steps, items_completed, etc.
        time_step_s: Time per step in seconds
        price_per_unit: Price per completed item
        human_wage: Human wage for comparison

    Returns:
        WorkcellEpisodeMetrics
    """
    success = episode_data.get("success", False)
    total_reward = episode_data.get("total_reward", 0.0)
    steps = episode_data.get("steps", 0)
    time_s = steps * time_step_s

    items_completed = episode_data.get("items_completed", 0)
    items_total = episode_data.get("items_total", 0)
    errors = episode_data.get("errors", 0)
    tolerance_violations = episode_data.get("tolerance_violations", 0)

    # Compute economic metrics
    hours = time_s / 3600.0 if time_s > 0 else 1.0
    throughput_per_hour = items_completed / hours if hours > 0 else 0.0

    revenue = items_completed * price_per_unit
    damage_cost = errors * (price_per_unit * 2)  # Assume 2x cost for errors
    implied_wage = (revenue - damage_cost) / hours if hours > 0 else 0.0

    completion_rate = items_completed / max(items_total, 1)
    error_rate = errors / max(items_total, 1)
    quality_score = completion_rate * (1 - error_rate)

    # Get difficulty features if available
    difficulty = episode_data.get("difficulty_features")
    if difficulty is None and "config" in episode_data:
        from src.envs.workcell_env.config import WorkcellEnvConfig
        config = episode_data["config"]
        if isinstance(config, WorkcellEnvConfig):
            difficulty = compute_difficulty_features(config)

    return WorkcellEpisodeMetrics(
        episode_id=episode_id,
        task_type=task_type,
        success=success,
        total_reward=total_reward,
        steps=steps,
        time_s=time_s,
        items_completed=items_completed,
        items_total=items_total,
        errors=errors,
        tolerance_violations=tolerance_violations,
        throughput_per_hour=throughput_per_hour,
        implied_wage=implied_wage,
        quality_score=quality_score,
        difficulty=difficulty,
    )


def compute_suite_report(
    episode_metrics: List[WorkcellEpisodeMetrics],
) -> WorkcellSuiteReport:
    """
    Compute aggregate report from episode metrics.

    Args:
        episode_metrics: List of episode metrics

    Returns:
        WorkcellSuiteReport with aggregates and correlations
    """
    if not episode_metrics:
        return WorkcellSuiteReport(
            num_episodes=0,
            success_rate=0.0,
            mean_reward=0.0,
            mean_steps=0.0,
            mean_time_s=0.0,
            total_items_completed=0,
            total_items_attempted=0,
            completion_rate=0.0,
            error_rate=0.0,
            mean_throughput_per_hour=0.0,
            mean_implied_wage=0.0,
            mean_quality_score=0.0,
        )

    n = len(episode_metrics)

    # Basic aggregates
    success_rate = sum(1 for m in episode_metrics if m.success) / n
    mean_reward = sum(m.total_reward for m in episode_metrics) / n
    mean_steps = sum(m.steps for m in episode_metrics) / n
    mean_time_s = sum(m.time_s for m in episode_metrics) / n

    # Task aggregates
    total_completed = sum(m.items_completed for m in episode_metrics)
    total_attempted = sum(m.items_total for m in episode_metrics)
    total_errors = sum(m.errors for m in episode_metrics)

    completion_rate = total_completed / max(total_attempted, 1)
    error_rate = total_errors / max(total_attempted, 1)

    # Economic aggregates
    mean_throughput = sum(m.throughput_per_hour for m in episode_metrics) / n
    mean_wage = sum(m.implied_wage for m in episode_metrics) / n
    mean_quality = sum(m.quality_score for m in episode_metrics) / n

    # Difficulty correlations
    difficulty_success_corr = 0.0
    difficulty_reward_corr = 0.0

    metrics_with_difficulty = [m for m in episode_metrics if m.difficulty is not None]
    if len(metrics_with_difficulty) >= 3:
        difficulties = np.array([
            m.difficulty.composite_difficulty() for m in metrics_with_difficulty
        ])
        successes = np.array([float(m.success) for m in metrics_with_difficulty])
        rewards = np.array([m.total_reward for m in metrics_with_difficulty])

        if np.std(difficulties) > 0:
            difficulty_success_corr = float(np.corrcoef(difficulties, successes)[0, 1])
            difficulty_reward_corr = float(np.corrcoef(difficulties, rewards)[0, 1])

    # Per-task-type breakdown
    by_task_type: Dict[str, Dict[str, float]] = {}
    task_types = set(m.task_type for m in episode_metrics)

    for tt in task_types:
        tt_metrics = [m for m in episode_metrics if m.task_type == tt]
        tt_n = len(tt_metrics)
        by_task_type[tt] = {
            "count": float(tt_n),
            "success_rate": sum(1 for m in tt_metrics if m.success) / tt_n,
            "mean_reward": sum(m.total_reward for m in tt_metrics) / tt_n,
            "mean_quality": sum(m.quality_score for m in tt_metrics) / tt_n,
        }

    return WorkcellSuiteReport(
        num_episodes=n,
        success_rate=success_rate,
        mean_reward=mean_reward,
        mean_steps=mean_steps,
        mean_time_s=mean_time_s,
        total_items_completed=total_completed,
        total_items_attempted=total_attempted,
        completion_rate=completion_rate,
        error_rate=error_rate,
        mean_throughput_per_hour=mean_throughput,
        mean_implied_wage=mean_wage,
        mean_quality_score=mean_quality,
        difficulty_success_correlation=difficulty_success_corr,
        difficulty_reward_correlation=difficulty_reward_corr,
        by_task_type=by_task_type,
    )


def format_suite_report(report: WorkcellSuiteReport) -> str:
    """Format suite report as human-readable string."""
    lines = [
        "=" * 60,
        "WORKCELL SUITE REPORT",
        "=" * 60,
        f"Episodes: {report.num_episodes}",
        f"Success Rate: {report.success_rate:.1%}",
        f"Mean Reward: {report.mean_reward:.2f}",
        f"Mean Steps: {report.mean_steps:.1f}",
        f"Mean Time: {report.mean_time_s:.1f}s",
        "",
        "--- Task Metrics ---",
        f"Items Completed: {report.total_items_completed}/{report.total_items_attempted}",
        f"Completion Rate: {report.completion_rate:.1%}",
        f"Error Rate: {report.error_rate:.1%}",
        "",
        "--- Economic Metrics ---",
        f"Mean Throughput: {report.mean_throughput_per_hour:.1f}/hr",
        f"Mean Implied Wage: ${report.mean_implied_wage:.2f}/hr",
        f"Mean Quality Score: {report.mean_quality_score:.2f}",
        "",
        "--- Difficulty Correlations ---",
        f"Difficulty ↔ Success: {report.difficulty_success_correlation:.3f}",
        f"Difficulty ↔ Reward: {report.difficulty_reward_correlation:.3f}",
    ]

    if report.by_task_type:
        lines.append("")
        lines.append("--- By Task Type ---")
        for tt, stats in report.by_task_type.items():
            lines.append(f"  {tt}: {int(stats['count'])} eps, "
                        f"{stats['success_rate']:.0%} success, "
                        f"reward={stats['mean_reward']:.2f}")

    lines.append("=" * 60)
    return "\n".join(lines)


def log_workcell_metrics(
    episode_metrics: WorkcellEpisodeMetrics,
    logger_name: str = "workcell.metrics",
) -> None:
    """Log episode metrics to named logger."""
    log = logging.getLogger(logger_name)
    log.info(
        "Episode %s [%s]: success=%s reward=%.2f steps=%d throughput=%.1f/hr",
        episode_metrics.episode_id,
        episode_metrics.task_type,
        episode_metrics.success,
        episode_metrics.total_reward,
        episode_metrics.steps,
        episode_metrics.throughput_per_hour,
    )

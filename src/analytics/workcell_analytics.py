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
class FailureTaxonomy:
    """Breakdown of failure types for an episode."""
    collision_count: int = 0
    timeout_count: int = 0
    precision_count: int = 0
    other_count: int = 0

    @property
    def total(self) -> int:
        return self.collision_count + self.timeout_count + self.precision_count + self.other_count

    def to_dict(self) -> Dict[str, int]:
        return {
            "collision": self.collision_count,
            "timeout": self.timeout_count,
            "precision": self.precision_count,
            "other": self.other_count,
        }


@dataclass
class ManufacturingKPIs:
    """Manufacturing-specific key performance indicators."""
    cycle_time_s: float = 0.0  # Avg time per completed item
    contact_force_proxy: float = 0.0  # Proxy for gripper force usage
    scrap_proxy: float = 0.0  # Fraction of items that became scrap
    failure_taxonomy: FailureTaxonomy = field(default_factory=FailureTaxonomy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_time_s": self.cycle_time_s,
            "contact_force_proxy": self.contact_force_proxy,
            "scrap_proxy": self.scrap_proxy,
            "failure_taxonomy": self.failure_taxonomy.to_dict(),
        }


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

    # Manufacturing KPIs
    manufacturing_kpis: Optional[ManufacturingKPIs] = None

    # Difficulty features
    difficulty: Optional[WorkcellDifficultyFeatures] = None


@dataclass
class AggregateManufacturingKPIs:
    """Aggregate manufacturing KPIs across episodes."""
    mean_cycle_time_s: float = 0.0
    mean_contact_force_proxy: float = 0.0
    mean_scrap_proxy: float = 0.0
    total_failures_by_type: Dict[str, int] = field(default_factory=dict)
    failure_rate_by_type: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_cycle_time_s": self.mean_cycle_time_s,
            "mean_contact_force_proxy": self.mean_contact_force_proxy,
            "mean_scrap_proxy": self.mean_scrap_proxy,
            "total_failures_by_type": self.total_failures_by_type,
            "failure_rate_by_type": self.failure_rate_by_type,
        }


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

    # Manufacturing KPI aggregates
    manufacturing_kpis: AggregateManufacturingKPIs = field(default_factory=AggregateManufacturingKPIs)

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

    # Compute manufacturing KPIs
    manufacturing_kpis = _compute_manufacturing_kpis(
        episode_data=episode_data,
        items_completed=items_completed,
        items_total=items_total,
        time_s=time_s,
        errors=errors,
        tolerance_violations=tolerance_violations,
        success=success,
    )

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
        manufacturing_kpis=manufacturing_kpis,
        difficulty=difficulty,
    )


def _compute_manufacturing_kpis(
    episode_data: Dict[str, Any],
    items_completed: int,
    items_total: int,
    time_s: float,
    errors: int,
    tolerance_violations: int,
    success: bool,
) -> ManufacturingKPIs:
    """
    Compute manufacturing KPIs for an episode.

    Args:
        episode_data: Raw episode data dict
        items_completed: Number of items completed
        items_total: Total items attempted
        time_s: Total episode time in seconds
        errors: Number of errors
        tolerance_violations: Number of tolerance violations
        success: Whether episode was successful

    Returns:
        ManufacturingKPIs
    """
    # Cycle time: average time per completed item
    cycle_time_s = time_s / max(items_completed, 1)

    # Contact force proxy: estimate from gripper actions
    # Use gripper_close_count if available, otherwise estimate from steps
    gripper_actions = episode_data.get("gripper_close_count", 0)
    if gripper_actions == 0:
        # Estimate: assume ~2 gripper actions per completed item
        gripper_actions = max(items_completed * 2, episode_data.get("steps", 0) // 10)
    # Normalize to [0, 1] range: more gripper actions = higher force proxy
    contact_force_proxy = min(1.0, gripper_actions / max(items_total * 3, 1))

    # Scrap proxy: fraction of items that became scrap (errors + incomplete)
    items_failed = items_total - items_completed
    items_scrapped = errors + items_failed
    scrap_proxy = items_scrapped / max(items_total, 1)

    # Failure taxonomy: classify failures by type
    failure_taxonomy = _classify_failures(
        episode_data=episode_data,
        errors=errors,
        tolerance_violations=tolerance_violations,
        success=success,
        time_s=time_s,
    )

    return ManufacturingKPIs(
        cycle_time_s=cycle_time_s,
        contact_force_proxy=contact_force_proxy,
        scrap_proxy=scrap_proxy,
        failure_taxonomy=failure_taxonomy,
    )


def _classify_failures(
    episode_data: Dict[str, Any],
    errors: int,
    tolerance_violations: int,
    success: bool,
    time_s: float,
) -> FailureTaxonomy:
    """
    Classify failures into taxonomy categories.

    Categories:
        - collision: Physical collision detected
        - timeout: Episode exceeded time limit
        - precision: Tolerance/placement errors
        - other: Unclassified failures
    """
    # Extract failure info from episode_data if available
    collision_count = episode_data.get("collision_count", 0)
    timeout_count = 0
    precision_count = tolerance_violations
    other_count = 0

    # Check for timeout
    max_time = episode_data.get("max_time_s", episode_data.get("max_steps", 100) * 1.0)
    if not success and time_s >= max_time * 0.95:
        timeout_count = 1

    # Remaining errors go to other
    accounted_failures = collision_count + timeout_count + precision_count
    if errors > accounted_failures:
        other_count = errors - accounted_failures

    return FailureTaxonomy(
        collision_count=collision_count,
        timeout_count=timeout_count,
        precision_count=precision_count,
        other_count=other_count,
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

    # Manufacturing KPI aggregates
    manufacturing_kpis = _aggregate_manufacturing_kpis(episode_metrics)

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
        manufacturing_kpis=manufacturing_kpis,
        difficulty_success_correlation=difficulty_success_corr,
        difficulty_reward_correlation=difficulty_reward_corr,
        by_task_type=by_task_type,
    )


def _aggregate_manufacturing_kpis(
    episode_metrics: List[WorkcellEpisodeMetrics],
) -> AggregateManufacturingKPIs:
    """
    Aggregate manufacturing KPIs across episodes.

    Args:
        episode_metrics: List of episode metrics with manufacturing_kpis

    Returns:
        AggregateManufacturingKPIs
    """
    metrics_with_kpis = [m for m in episode_metrics if m.manufacturing_kpis is not None]

    if not metrics_with_kpis:
        return AggregateManufacturingKPIs()

    n = len(metrics_with_kpis)

    # Mean KPIs
    mean_cycle_time = sum(m.manufacturing_kpis.cycle_time_s for m in metrics_with_kpis) / n
    mean_contact_force = sum(m.manufacturing_kpis.contact_force_proxy for m in metrics_with_kpis) / n
    mean_scrap = sum(m.manufacturing_kpis.scrap_proxy for m in metrics_with_kpis) / n

    # Aggregate failure taxonomy
    total_failures = {
        "collision": sum(m.manufacturing_kpis.failure_taxonomy.collision_count for m in metrics_with_kpis),
        "timeout": sum(m.manufacturing_kpis.failure_taxonomy.timeout_count for m in metrics_with_kpis),
        "precision": sum(m.manufacturing_kpis.failure_taxonomy.precision_count for m in metrics_with_kpis),
        "other": sum(m.manufacturing_kpis.failure_taxonomy.other_count for m in metrics_with_kpis),
    }

    total_all_failures = sum(total_failures.values())
    failure_rates = {
        k: v / max(total_all_failures, 1) for k, v in total_failures.items()
    }

    return AggregateManufacturingKPIs(
        mean_cycle_time_s=mean_cycle_time,
        mean_contact_force_proxy=mean_contact_force,
        mean_scrap_proxy=mean_scrap,
        total_failures_by_type=total_failures,
        failure_rate_by_type=failure_rates,
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
        "--- Manufacturing KPIs ---",
        f"Mean Cycle Time: {report.manufacturing_kpis.mean_cycle_time_s:.1f}s",
        f"Contact Force Proxy: {report.manufacturing_kpis.mean_contact_force_proxy:.3f}",
        f"Scrap Proxy: {report.manufacturing_kpis.mean_scrap_proxy:.1%}",
        f"Failures by Type: {report.manufacturing_kpis.total_failures_by_type}",
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

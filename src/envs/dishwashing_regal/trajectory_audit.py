"""
TrajectoryAuditV1 integration for dishwashing environment.

Phase 3: Makes TrajectoryAuditV1 unavoidable in training entrypoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.contracts.schemas import TrajectoryAuditV1, RewardBreakdownV1
from src.valuation.trajectory_audit import create_trajectory_audit


class DishwashingTrajectoryCollector:
    """Collects trajectory data during episode rollout for audit.

    Usage:
        collector = DishwashingTrajectoryCollector(episode_id="ep_001")
        for step in episode:
            collector.record_step(action, obs, reward, info)
        audit = collector.build_audit()
    """

    def __init__(
        self,
        episode_id: str,
        error_rate_threshold: float = 0.12,  # SLA threshold
    ):
        self.episode_id = episode_id
        self.error_rate_threshold = error_rate_threshold

        # Trajectory data
        self._actions: List[List[float]] = []
        self._rewards: List[float] = []
        self._reward_components: Dict[str, List[float]] = {}

        # Event counters
        self._events: List[str] = []
        self._errors: int = 0
        self._completed: int = 0
        self._attempts: int = 0
        self._catastrophic_errors: int = 0

        # SLA tracking
        self._sla_violations: int = 0
        self._consecutive_zero_throughput: int = 0

        # Energy tracking
        self._energy_Wh: float = 0.0
        self._step_energies: List[float] = []

    def record_step(
        self,
        action: Any,
        obs: Dict[str, Any],
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Record a single step.

        Args:
            action: Action taken (array-like or scalar)
            obs: Observation dict
            reward: Scalar reward (unused for this env's reward structure)
            info: Step info dict with dishwashing metrics
        """
        # Extract action as list
        if hasattr(action, "tolist"):
            action_list = action.tolist()
        elif isinstance(action, (list, tuple)):
            action_list = list(action)
        elif np.isscalar(action):
            action_list = [float(action), 0.5]  # Default care
        else:
            action_list = [0.5, 0.5]
        self._actions.append(action_list)

        # Record reward (from info, not step return for dishwashing)
        step_profit = info.get("profit", 0.0)
        self._rewards.append(float(step_profit))

        # Extract reward components from info or breakdown
        if "reward_breakdown" in info:
            rb = info["reward_breakdown"]
            if isinstance(rb, RewardBreakdownV1):
                for key, val in rb.to_dict().items():
                    if key not in self._reward_components:
                        self._reward_components[key] = []
                    self._reward_components[key].append(float(val) if val is not None else 0.0)
            elif isinstance(rb, dict):
                for key, val in rb.items():
                    if key not in self._reward_components:
                        self._reward_components[key] = []
                    self._reward_components[key].append(float(val) if val is not None else 0.0)

        # Extract metrics from info
        step_errors = info.get("delta_errors", 0)
        step_completed = info.get("delta_completed", 0)
        step_attempts = info.get("delta_attempts", 0)
        step_energy = info.get("delta_energy_Wh", 0.0)

        # Update counters
        self._errors += step_errors
        self._completed += step_completed
        self._attempts += step_attempts
        self._energy_Wh += step_energy
        self._step_energies.append(step_energy)

        # Record events
        if step_errors > 0:
            for _ in range(step_errors):
                self._events.append("error")

        if step_completed > 0:
            for _ in range(step_completed):
                self._events.append("item_completed")

        # Catastrophic error detection
        if step_attempts > 0 and step_errors == step_attempts:
            self._catastrophic_errors += 1
            self._events.append("catastrophic_error")

        # SLA violation detection
        error_rate = info.get("error_rate", 0.0)
        if error_rate > self.error_rate_threshold:
            self._sla_violations += 1
            self._events.append("sla_violation")

        # Zero throughput tracking
        if step_completed == 0:
            self._consecutive_zero_throughput += 1
            if self._consecutive_zero_throughput >= 10:  # patience threshold
                self._events.append("zero_throughput_warning")
        else:
            self._consecutive_zero_throughput = 0

    def build_audit(self) -> TrajectoryAuditV1:
        """Build TrajectoryAuditV1 from collected data.

        Returns:
            TrajectoryAuditV1 with full trajectory provenance
        """
        return create_trajectory_audit(
            episode_id=self.episode_id,
            num_steps=len(self._actions),
            actions=self._actions,
            rewards=self._rewards,
            reward_components=self._reward_components if self._reward_components else None,
            events=self._events if self._events else None,
        )

    @property
    def event_counts(self) -> Dict[str, int]:
        """Get event counts for inspection."""
        from collections import Counter
        return dict(Counter(self._events))

    @property
    def summary(self) -> Dict[str, Any]:
        """Get summary metrics."""
        return {
            "completed": self._completed,
            "errors": self._errors,
            "attempts": self._attempts,
            "energy_Wh": self._energy_Wh,
            "catastrophic_errors": self._catastrophic_errors,
            "sla_violations": self._sla_violations,
        }


def create_dishwashing_trajectory_audit(
    episode_id: str,
    actions: List[List[float]],
    infos: List[Dict[str, Any]],
) -> TrajectoryAuditV1:
    """Create TrajectoryAuditV1 from episode rollout data.

    Convenience function for post-hoc audit creation.

    Args:
        episode_id: Episode identifier
        actions: List of action arrays
        infos: List of step info dicts

    Returns:
        TrajectoryAuditV1 with full provenance
    """
    collector = DishwashingTrajectoryCollector(episode_id=episode_id)

    for action, info in zip(actions, infos):
        obs = info.get("obs", {})
        reward = info.get("profit", 0.0)
        collector.record_step(action, obs, reward, info)

    return collector.build_audit()


__all__ = [
    "DishwashingTrajectoryCollector",
    "create_dishwashing_trajectory_audit",
]

"""
TrajectoryAuditV1 integration for workcell environments.

Phase 3: Makes TrajectoryAuditV1 unavoidable in training entrypoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.contracts.schemas import TrajectoryAuditV1, RewardBreakdownV1
from src.valuation.trajectory_audit import create_trajectory_audit


class WorkcellTrajectoryCollector:
    """Collects trajectory data during episode rollout for audit.

    Usage:
        collector = WorkcellTrajectoryCollector(episode_id="ep_001")
        for step in episode:
            collector.record_step(action, obs, reward, info)
        audit = collector.build_audit()
    """

    def __init__(
        self,
        episode_id: str,
        velocity_threshold: float = 10.0,
        contact_force_threshold: float = 100.0,
        penetration_threshold: float = 0.01,
    ):
        self.episode_id = episode_id
        self.velocity_threshold = velocity_threshold
        self.contact_force_threshold = contact_force_threshold
        self.penetration_threshold = penetration_threshold

        # Trajectory data
        self._actions: List[List[float]] = []
        self._rewards: List[float] = []
        self._reward_components: Dict[str, List[float]] = {}
        self._velocities: List[List[float]] = []

        # Event counters
        self._events: List[str] = []
        self._constraint_violations: int = 0
        self._collision_count: int = 0
        self._grasp_attempts: int = 0
        self._grasp_successes: int = 0
        self._place_attempts: int = 0
        self._place_successes: int = 0

        # Physics anomaly detection
        self._velocity_spikes: int = 0
        self._contact_anomalies: int = 0
        self._penetrations: List[float] = []

    def record_step(
        self,
        action: Any,
        obs: Dict[str, Any],
        reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Record a single step.

        Args:
            action: Action taken (array-like or dict)
            obs: Observation dict
            reward: Scalar reward
            info: Step info dict with task details
        """
        # Extract action as list
        if hasattr(action, "tolist"):
            action_list = action.tolist()
        elif isinstance(action, dict):
            action_list = list(action.values()) if action else [0.0]
        elif isinstance(action, (list, tuple)):
            action_list = list(action)
        else:
            action_list = [float(action)]
        self._actions.append(action_list)

        # Record reward
        self._rewards.append(float(reward))

        # Extract reward components from info or breakdown
        task_info = info.get("task", {})
        if "reward_breakdown" in info:
            rb = info["reward_breakdown"]
            if isinstance(rb, RewardBreakdownV1):
                for key, val in rb.to_dict().items():
                    if key not in self._reward_components:
                        self._reward_components[key] = []
                    self._reward_components[key].append(float(val))
            elif isinstance(rb, dict):
                for key, val in rb.items():
                    if key not in self._reward_components:
                        self._reward_components[key] = []
                    self._reward_components[key].append(float(val) if val is not None else 0.0)

        # Extract velocity from obs or info
        velocity = None
        if "ee_velocity" in obs:
            velocity = obs["ee_velocity"]
        elif "velocity" in info:
            velocity = info["velocity"]
        elif "task_state" in info and "velocity" in info["task_state"]:
            velocity = info["task_state"]["velocity"]

        if velocity is not None:
            if hasattr(velocity, "tolist"):
                velocity = velocity.tolist()
            self._velocities.append(list(velocity))

            # Check for velocity spike
            vel_mag = np.linalg.norm(velocity)
            if vel_mag > self.velocity_threshold:
                self._velocity_spikes += 1

        # Extract event counters from info
        task_state = info.get("task_state", info.get("task", {}))

        # Constraint violations
        if task_state.get("constraint_error", 0) > 0:
            self._constraint_violations += 1
            self._events.append("constraint_violation")

        # Collisions
        collision_this_step = task_state.get("collision_count", 0)
        if collision_this_step > self._collision_count:
            delta = collision_this_step - self._collision_count
            self._collision_count = collision_this_step
            for _ in range(delta):
                self._events.append("collision")

        # Contact force anomalies
        contact_force = task_state.get("contact_force_N", 0)
        if contact_force > self.contact_force_threshold:
            self._contact_anomalies += 1

        # Penetration check - track as list for create_trajectory_audit
        penetration = task_state.get("penetration", 0)
        self._penetrations.append(float(penetration))

        # Grasp/place events
        if info.get("grasp_attempt"):
            self._grasp_attempts += 1
            self._events.append("grasp_attempt")
        if info.get("grasp_success"):
            self._grasp_successes += 1
            self._events.append("grasp_success")
        if info.get("place_attempt"):
            self._place_attempts += 1
            self._events.append("place_attempt")
        if info.get("place_success"):
            self._place_successes += 1
            self._events.append("place_success")

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
            penetrations=self._penetrations if self._penetrations else None,
            velocities=self._velocities if self._velocities else None,
            velocity_threshold=self.velocity_threshold,
        )

    @property
    def event_counts(self) -> Dict[str, int]:
        """Get event counts for inspection."""
        from collections import Counter
        return dict(Counter(self._events))


def create_workcell_trajectory_audit(
    episode_id: str,
    actions: List[List[float]],
    rewards: List[float],
    infos: List[Dict[str, Any]],
    velocity_threshold: float = 10.0,
) -> TrajectoryAuditV1:
    """Create TrajectoryAuditV1 from episode rollout data.

    Convenience function for post-hoc audit creation.

    Args:
        episode_id: Episode identifier
        actions: List of action arrays
        rewards: List of scalar rewards
        infos: List of step info dicts
        velocity_threshold: Threshold for velocity spike detection

    Returns:
        TrajectoryAuditV1 with full provenance
    """
    collector = WorkcellTrajectoryCollector(
        episode_id=episode_id,
        velocity_threshold=velocity_threshold,
    )

    for action, reward, info in zip(actions, rewards, infos):
        obs = info.get("obs", {})
        collector.record_step(action, obs, reward, info)

    return collector.build_audit()


__all__ = [
    "WorkcellTrajectoryCollector",
    "create_workcell_trajectory_audit",
]

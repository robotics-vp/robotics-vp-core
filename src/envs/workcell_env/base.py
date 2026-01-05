"""
Base environment for manufacturing workcell simulations.

Defines a Gym-like interface and emits episode logs compatible with the
rollout capture schema.
"""
from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.envs.workcell_env.config import WorkcellEnvConfig
from src.envs.workcell_env.scene.scene_spec import WorkcellSceneSpec
from src.motor_backend.rollout_capture import EpisodeMetadata


@dataclass(frozen=True)
class EpisodeLog:
    """
    Episode log payload compatible with rollout capture utilities.

    Stores metadata plus a minimal trajectory and info history.
    """
    metadata: EpisodeMetadata
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    info_history: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "metadata": asdict(self.metadata),
            "trajectory": list(self.trajectory),
            "info_history": list(self.info_history),
            "metrics": dict(self.metrics),
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeLog":
        """Deserialize from dictionary."""
        metadata = data.get("metadata", {})
        if isinstance(metadata, EpisodeMetadata):
            meta = metadata
        else:
            meta = EpisodeMetadata(**metadata)
        return cls(
            metadata=meta,
            trajectory=list(data.get("trajectory", [])),
            info_history=list(data.get("info_history", [])),
            metrics=dict(data.get("metrics", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EpisodeLog":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class WorkcellEnvBase:
    """
    Base class for workcell environments with a Gym-like interface.

    Subclasses should implement _reset_impl() and _step_impl().
    """

    def __init__(
        self,
        config: WorkcellEnvConfig,
        scene_spec: WorkcellSceneSpec,
        task_id: str = "workcell_task",
        robot_family: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.scene_spec = scene_spec
        self.task_id = task_id
        self.robot_family = robot_family
        self.seed = seed

        self._rng = random.Random(seed)
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._trajectory: List[Dict[str, Any]] = []
        self._info_history: List[Dict[str, Any]] = []

    @property
    def episode_id(self) -> Optional[str]:
        """Return current episode ID, if any."""
        return self._episode_id

    @property
    def step_count(self) -> int:
        """Return number of steps taken in the current episode."""
        return self._step_count

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        robot_family: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Reset environment state.

        Returns:
            Observation dict for the initial state.
        """
        if seed is not None:
            self.seed = seed
            self._rng = random.Random(seed)
        if task_id is not None:
            self.task_id = task_id
        if robot_family is not None:
            self.robot_family = robot_family

        self._episode_id = episode_id or f"workcell_{uuid.uuid4().hex[:12]}"
        self._step_count = 0
        self._trajectory = []
        self._info_history = []

        return self._reset_impl(options or {})

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Step the environment forward by one action (Gym API compatible).

        Returns:
            obs: Observation dict
            reward: Scalar reward
            terminated: Episode ended due to success/failure
            truncated: Episode ended due to time limit
            info: Info dict with metrics/state summaries
        """
        obs, info, done = self._step_impl(action)
        self._record_transition(action, obs, info, done)
        self._step_count += 1

        # Extract reward from info, default to 0
        reward = float(info.get("reward", 0.0))

        # Determine termination type
        success = info.get("success", False)
        terminated = bool(success)
        truncated = done and not terminated

        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        """Render the environment (optional for subclasses)."""
        return None

    def close(self) -> None:
        """Clean up environment resources."""
        return None

    def get_episode_log(self, metrics: Optional[Mapping[str, float]] = None) -> EpisodeLog:
        """
        Build an episode log payload compatible with rollout capture.
        """
        metadata = EpisodeMetadata(
            episode_id=self._episode_id or f"workcell_{uuid.uuid4().hex[:12]}",
            task_id=self.task_id,
            robot_family=self.robot_family,
            seed=self.seed,
            env_params={
                "config": self.config.to_dict(),
                "scene_spec": self.scene_spec.to_dict(),
            },
        )
        return EpisodeLog(
            metadata=metadata,
            trajectory=list(self._trajectory),
            info_history=list(self._info_history),
            metrics=dict(metrics or {}),
        )

    def _reset_impl(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Subclass hook for reset logic."""
        raise NotImplementedError

    def _step_impl(self, action: Any) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Subclass hook for step logic."""
        raise NotImplementedError

    def _record_transition(
        self,
        action: Any,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        done: bool,
    ) -> None:
        """Record transition for episode logging."""
        self._trajectory.append(
            {
                "step": self._step_count,
                "action": _serialize_action(action),
                "done": bool(done),
                "obs": obs,
                "info": info,
            }
        )
        if info:
            self._info_history.append(info)


def _serialize_action(action: Any) -> Any:
    """Normalize actions into JSON-friendly containers when possible."""
    if hasattr(action, "tolist"):
        return action.tolist()
    if isinstance(action, tuple):
        return list(action)
    return action

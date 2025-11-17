"""
PyBullet Physics Backend.

Wraps existing PyBullet-based environments (drawer+vase, dishwashing_arm)
without changing their behavior.
"""

from typing import Any, Dict, Optional, Tuple, Literal

from .base_engine import PhysicsBackend
from src.envs.dishwashing_env import EpisodeInfoSummary


class PyBulletBackend(PhysicsBackend):
    """
    PyBullet physics backend wrapper.

    Wraps existing PyBullet-based environments to provide a consistent
    interface for the economic valuation pipeline.

    This is a thin wrapper that:
    - Holds a reference to the underlying env
    - Forwards reset/step/get_episode_info calls
    - Tracks info history for episode summarization
    - Does NOT change any existing behavior
    """

    def __init__(self, env, env_name: str = "drawer_vase", summarize_fn=None):
        """
        Initialize PyBullet backend wrapper.

        Args:
            env: Underlying PyBullet environment instance
                (e.g., DrawerVasePhysicsEnv, DrawerVaseArmEnv)
            env_name: Environment identifier
            summarize_fn: Optional function to summarize episode info.
                         If None, will attempt to use env-specific summarizer.
        """
        self._env = env
        self._env_name = env_name
        self._summarize_fn = summarize_fn
        self._info_history = []
        self._episode_count = 0

        # Media references for datapack integration (mirrors Isaac backend)
        self._media_refs = {}  # episode_id -> {"rgb_path": ..., "depth_path": ...}
        self._current_episode_id = None

    @property
    def engine_type(self) -> Literal["pybullet", "isaac", "ue5"]:
        """Return engine type."""
        return "pybullet"

    @property
    def env_name(self) -> str:
        """Return environment name."""
        return self._env_name

    @property
    def underlying_env(self):
        """Get reference to underlying environment."""
        return self._env

    def reset(self, initial_state: Optional[Any] = None) -> Any:
        """
        Reset the environment.

        Args:
            initial_state: Optional initial state (passed to underlying env if supported)

        Returns:
            obs: Initial observation
        """
        self._info_history = []
        self._episode_count += 1

        # Generate episode ID for datapack tracking
        import uuid
        self._current_episode_id = str(uuid.uuid4())

        # Check if env supports initial_state parameter
        if initial_state is not None and hasattr(self._env, 'reset_with_state'):
            obs, info = self._env.reset_with_state(initial_state)
        else:
            result = self._env.reset()
            # Handle both old and new gym API
            if isinstance(result, tuple):
                obs, info = result
            else:
                obs = result
                info = {}

        return obs

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Execute one simulation step.

        Args:
            action: Action to execute

        Returns:
            obs: Next observation
            reward: Step reward
            done: Episode done flag
            info: Step information dictionary
        """
        result = self._env.step(action)

        # Handle both old and new gym API
        if len(result) == 5:
            obs, reward, done, truncated, info = result
            done = done or truncated
        else:
            obs, reward, done, info = result

        # Track info for episode summarization
        self._info_history.append(info)

        return obs, reward, done, info

    def get_episode_info(self) -> EpisodeInfoSummary:
        """
        Get episode-level summary.

        Uses the environment-specific summarizer or a custom one.

        Returns:
            EpisodeInfoSummary: Episode summary for datapack creation
        """
        if self._summarize_fn is not None:
            return self._summarize_fn(self._info_history)

        # Try to use env-specific summarizer
        if self._env_name == "drawer_vase":
            try:
                from src.envs.drawer_vase_physics_env import summarize_drawer_vase_episode
                return summarize_drawer_vase_episode(self._info_history)
            except ImportError:
                pass

        if self._env_name == "dishwashing":
            try:
                from src.envs.dishwashing_env import summarize_episode_info
                return summarize_episode_info(self._info_history)
            except ImportError:
                pass

        # Fallback: create minimal summary from last info
        if not self._info_history:
            return EpisodeInfoSummary(
                termination_reason="unknown",
                mpl_episode=0.0,
                ep_episode=0.0,
                error_rate_episode=0.0,
                throughput_units_per_hour=0.0,
                energy_Wh=0.0,
                energy_Wh_per_unit=0.0,
                energy_Wh_per_hour=0.0,
                limb_energy_Wh={},
                skill_energy_Wh={},
                energy_per_limb={},
                energy_per_skill={},
                energy_per_joint={},
                energy_per_effector={},
                coordination_metrics={},
                profit=0.0,
                wage_parity=None,
            )

        last_info = self._info_history[-1]
        return EpisodeInfoSummary(
            termination_reason=last_info.get("terminated_reason", "unknown"),
            mpl_episode=last_info.get("mpl", 0.0),
            ep_episode=last_info.get("ep", 0.0),
            error_rate_episode=last_info.get("error_rate", 0.0),
            throughput_units_per_hour=last_info.get("mpl", 0.0),
            energy_Wh=last_info.get("energy_Wh", 0.0),
            energy_Wh_per_unit=last_info.get("energy_Wh_per_unit", 0.0),
            energy_Wh_per_hour=last_info.get("energy_Wh_per_hour", 0.0),
            limb_energy_Wh=last_info.get("limb_energy_Wh", {}),
            skill_energy_Wh=last_info.get("skill_energy_Wh", {}),
            energy_per_limb=last_info.get("energy_per_limb", {}),
            energy_per_skill=last_info.get("energy_per_skill", {}),
            energy_per_joint=last_info.get("energy_per_joint", {}),
            energy_per_effector=last_info.get("energy_per_effector", {}),
            coordination_metrics=last_info.get("coordination_metrics", {}),
            profit=last_info.get("profit", 0.0),
            wage_parity=last_info.get("wage_parity"),
        )

    def get_info_history(self) -> list:
        """Get complete info history for current episode."""
        return self._info_history

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self._env, 'close'):
            self._env.close()

    def get_observation_space(self) -> Any:
        """Get observation space from underlying env."""
        if hasattr(self._env, 'observation_space'):
            return self._env.observation_space
        raise NotImplementedError("Underlying env has no observation_space")

    def get_action_space(self) -> Any:
        """Get action space from underlying env."""
        if hasattr(self._env, 'action_space'):
            return self._env.action_space
        raise NotImplementedError("Underlying env has no action_space")

    def render(self, mode: str = "human") -> Optional[Any]:
        """Render using underlying env."""
        if hasattr(self._env, 'render'):
            return self._env.render(mode=mode)
        raise NotImplementedError("Underlying env has no render method")

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed."""
        if hasattr(self._env, 'seed'):
            self._env.seed(seed)

    def get_state(self) -> Any:
        """Get simulation state."""
        if hasattr(self._env, 'get_state'):
            return self._env.get_state()
        raise NotImplementedError("Underlying env has no get_state method")

    def set_state(self, state: Any) -> None:
        """Set simulation state."""
        if hasattr(self._env, 'set_state'):
            self._env.set_state(state)
        else:
            raise NotImplementedError("Underlying env has no set_state method")

    # Media references for datapack integration (API parity with Isaac backend)

    def get_media_refs(self) -> Dict[str, str]:
        """
        Get media file references (RGB, depth, etc.) for datapack integration.

        Returns:
            dict: Media references for current episode, e.g.:
                {
                    "rgb_path": "/path/to/episode_123_rgb.mp4",
                    "depth_path": "/path/to/episode_123_depth.npy",
                }
        """
        if self._current_episode_id is None:
            return {}
        return self._media_refs.get(self._current_episode_id, {})

    def set_media_refs(self, refs: Dict[str, str]) -> None:
        """
        Set media file references for current episode.

        Args:
            refs: Dictionary of media paths
        """
        if self._current_episode_id is not None:
            self._media_refs[self._current_episode_id] = refs

    def get_current_episode_id(self) -> Optional[str]:
        """
        Get current episode ID.

        Returns:
            str: Episode identifier (UUID)
        """
        return self._current_episode_id

    def get_config(self) -> Dict[str, Any]:
        """
        Get backend configuration for logging and datapack ConditionProfile.

        Returns:
            dict: Configuration including:
                - env_name
                - engine_type
                - episode_count
        """
        return {
            "env_name": self._env_name,
            "engine_type": self.engine_type,
            "episode_count": self._episode_count,
        }

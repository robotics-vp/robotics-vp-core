"""
Physics Backend Abstraction Layer.

Provides engine-agnostic interface for robotics simulation backends.
Supports PyBullet, Isaac Gym, UE5, and future engines.

This abstraction allows:
- Same training code to run on different physics engines
- Engine-specific optimizations while maintaining consistent API
- Integration with EconParams, ObjectiveProfile, and datapacks
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Literal

from src.envs.dishwashing_env import EpisodeInfoSummary


class PhysicsBackend(ABC):
    """
    Abstract base class for physics simulation backends.

    All physics engines (PyBullet, Isaac, UE5) must implement this interface
    to integrate with the economic valuation pipeline.

    The backend wraps environment-specific logic and provides:
    - Consistent observation/action interface
    - Episode summarization for datapacks
    - Energy metrics computation
    - Integration with EconParams and ObjectiveProfile
    """

    @property
    @abstractmethod
    def engine_type(self) -> Literal["pybullet", "isaac", "ue5"]:
        """
        Return the physics engine type.

        Returns:
            str: One of "pybullet", "isaac", "ue5"
        """
        pass

    @property
    @abstractmethod
    def env_name(self) -> str:
        """
        Return the environment name.

        Returns:
            str: Environment identifier (e.g., "drawer_vase", "dishwashing")
        """
        pass

    @abstractmethod
    def reset(self, initial_state: Optional[Any] = None) -> Any:
        """
        Reset the environment to initial state.

        Args:
            initial_state: Optional initial state configuration.
                          If None, use default initialization.

        Returns:
            obs: Initial observation after reset
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Execute one simulation step.

        Args:
            action: Action to execute (engine-specific format)

        Returns:
            obs: Next observation
            reward: Step reward (raw, before economic shaping)
            done: Whether episode is finished
            info: Dictionary with step information including:
                - energy_Wh: Energy consumed this step
                - success: Whether task succeeded (if applicable)
                - any engine-specific metrics
        """
        pass

    @abstractmethod
    def get_episode_info(self) -> EpisodeInfoSummary:
        """
        Get episode-level summary for datapack creation.

        Must aggregate step-level metrics into EpisodeInfoSummary:
        - MPL computation from task completions
        - Error rate from failures/damages
        - Energy productivity from energy tracking
        - Per-limb, per-skill, per-joint energy breakdown

        Returns:
            EpisodeInfoSummary: Canonical episode summary
        """
        pass

    @abstractmethod
    def get_info_history(self) -> list:
        """
        Get complete info history for the current episode.

        Returns:
            list: List of info dicts from each step
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Clean up resources.

        Should release physics engine resources, close GUI windows, etc.
        """
        pass

    def get_observation_space(self) -> Any:
        """
        Get the observation space specification.

        Returns:
            Observation space (gym.Space or similar)

        Raises:
            NotImplementedError: If not supported by backend
        """
        raise NotImplementedError("Observation space not defined for this backend")

    def get_action_space(self) -> Any:
        """
        Get the action space specification.

        Returns:
            Action space (gym.Space or similar)

        Raises:
            NotImplementedError: If not supported by backend
        """
        raise NotImplementedError("Action space not defined for this backend")

    def render(self, mode: str = "human") -> Optional[Any]:
        """
        Render the current state.

        Args:
            mode: Render mode ("human", "rgb_array", etc.)

        Returns:
            Rendered output (None for "human", array for "rgb_array")

        Raises:
            NotImplementedError: If rendering not supported
        """
        raise NotImplementedError("Rendering not supported for this backend")

    def seed(self, seed: Optional[int] = None) -> None:
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        pass  # Default: no-op

    def get_state(self) -> Any:
        """
        Get current simulation state for checkpointing.

        Returns:
            State representation (engine-specific)

        Raises:
            NotImplementedError: If state saving not supported
        """
        raise NotImplementedError("State saving not supported for this backend")

    def set_state(self, state: Any) -> None:
        """
        Restore simulation state from checkpoint.

        Args:
            state: Previously saved state

        Raises:
            NotImplementedError: If state loading not supported
        """
        raise NotImplementedError("State loading not supported for this backend")

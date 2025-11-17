from typing import Any, Dict, List, Optional, Tuple, Literal

from .base_engine import PhysicsBackend
from src.envs.dishwashing_env import EpisodeInfoSummary


class IsaacBackend(PhysicsBackend):
    """
    Stub Isaac Gym backend.
    """

    def __init__(self, env_config: Optional[Dict[str, Any]] = None, num_envs: int = 1, device: str = "cuda:0"):
        self.env_config = env_config or {}
        self._num_envs = num_envs
        self.device = device
        self._env_name = env_config.get("env_name", "isaac_default") if env_config else "isaac_default"
        self._info_history = []
        self._media_refs: Dict[Any, Dict[str, Any]] = {}  # Keyed by env_idx or episode_id
        self._episode_ids: Dict[int, str] = {}  # env_idx -> episode_id for vectorized envs
        self._state = None
        # TODO: initialize Isaac environment when available

    @property
    def engine_type(self) -> Literal["pybullet", "isaac", "ue5"]:
        return "isaac"

    @property
    def env_name(self) -> str:
        return self._env_name

    @property
    def num_envs(self) -> int:
        """Number of parallel environments (Isaac-specific for vectorized envs)."""
        return self._num_envs

    def reset(self, initial_state: Optional[Any] = None) -> Any:
        raise NotImplementedError("Isaac backend reset not implemented yet.")

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        raise NotImplementedError("Isaac backend step not implemented yet.")

    def get_episode_info(self) -> EpisodeInfoSummary:
        raise NotImplementedError("Isaac backend episode summary not implemented yet.")

    def get_info_history(self):
        """Return info history (stub)."""
        raise NotImplementedError("Isaac backend get_info_history not implemented yet.")

    def close(self) -> None:
        """Close Isaac environment (stub)."""
        raise NotImplementedError("Isaac backend close not implemented yet.")

    def get_media_refs(self, key: Optional[Any] = None) -> Dict[str, Any]:
        """Get media references for episode or env index (stub for API parity).

        Args:
            key: Either episode_id (str) or env_idx (int) for vectorized envs
        """
        if key is not None and key in self._media_refs:
            return self._media_refs[key]
        return {}

    def set_media_refs(self, key: Any, refs: Dict[str, Any]) -> None:
        """Set media references for episode or env index (stub for API parity).

        Args:
            key: Either episode_id (str) or env_idx (int) for vectorized envs
            refs: Dictionary of media references (e.g., rgb_path, depth_path)
        """
        self._media_refs[key] = refs

        # Auto-generate episode ID for integer env indices if not already set
        if isinstance(key, int) and key not in self._episode_ids:
            self._episode_ids[key] = f"ep_{key + 1:03d}"

    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed (stub)."""
        raise NotImplementedError("Isaac backend seed not implemented yet.")

    def render(self, mode: str = "rgb_array") -> Optional[Any]:
        """Render environment (stub)."""
        raise NotImplementedError("Isaac backend render not implemented yet.")

    def get_state(self) -> Any:
        """Get environment state (stub)."""
        raise NotImplementedError("Isaac backend get_state not implemented yet.")

    def set_state(self, state: Any) -> None:
        """Set environment state (stub)."""
        raise NotImplementedError("Isaac backend set_state not implemented yet.")

    def get_observation_space(self) -> Any:
        """Get observation space (stub)."""
        raise NotImplementedError("Isaac backend get_observation_space not implemented yet.")

    def get_action_space(self) -> Any:
        """Get action space (stub)."""
        raise NotImplementedError("Isaac backend get_action_space not implemented yet.")

    def get_current_episode_id(self, env_idx: Optional[int] = None) -> Optional[str]:
        """Get current episode ID (stub). Optional env_idx for vectorized envs."""
        if env_idx is not None:
            return self._episode_ids.get(env_idx, None)
        return None

    def get_config(self) -> Dict[str, Any]:
        """Get backend config (stub)."""
        # Return top-level config with env_config nested for orchestrator access
        config = {
            "env_name": self._env_name,
            "engine_type": self.engine_type,
            "num_envs": self.num_envs,
            "device": self.device,
            "env_config": self.env_config,  # Nested for task-specific params
        }
        return config

    def get_batch_episode_info(self) -> List[EpisodeInfoSummary]:
        """Get episode info for all parallel environments (stub for Isaac multi-env)."""
        raise NotImplementedError("Isaac backend get_batch_episode_info not implemented yet.")

    def reset_env(self, env_idx: int, initial_state: Optional[Any] = None) -> Any:
        """Reset a specific environment in the vectorized batch (Isaac-specific)."""
        raise NotImplementedError("Isaac backend reset_env not implemented yet.")

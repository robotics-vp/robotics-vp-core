from typing import Any, Dict, Optional, Tuple, Literal

from .base_engine import PhysicsBackend
from src.envs.dishwashing_env import EpisodeInfoSummary


class IsaacBackend(PhysicsBackend):
    """
    Stub Isaac Gym backend.
    """

    def __init__(self, env_config: Optional[Dict[str, Any]] = None, num_envs: int = 1, device: str = "cuda:0"):
        self.env_config = env_config or {}
        self.num_envs = num_envs
        self.device = device
        # TODO: initialize Isaac environment when available

    @property
    def engine_type(self) -> Literal["pybullet", "isaac", "ue5"]:
        return "isaac"

    def reset(self, initial_state: Optional[Any] = None) -> Any:
        raise NotImplementedError("Isaac backend reset not implemented yet.")

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        raise NotImplementedError("Isaac backend step not implemented yet.")

    def get_episode_info(self) -> EpisodeInfoSummary:
        raise NotImplementedError("Isaac backend episode summary not implemented yet.")

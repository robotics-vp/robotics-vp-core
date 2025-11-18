"""
PyBullet wrapper backend around DishwashingEnv (placeholder physics env).
"""
from typing import Any, Dict, Tuple, Optional

from src.physics.backends.base import PhysicsBackend
from src.envs.dishwashing_env import DishwashingEnv
from src.config.econ_params import EconParams, load_econ_params
from src.config.internal_profile import get_internal_experiment_profile
from src.vision.interfaces import VisionFrame


class PyBulletBackend(PhysicsBackend):
    def __init__(self, econ_preset: str = "toy"):
        profile = get_internal_experiment_profile("dishwashing")
        params = load_econ_params(profile, preset=econ_preset)
        self.env = DishwashingEnv(params)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            import numpy as np
            np.random.seed(seed)
        return self.env.reset()

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        obs, info, done = self.env.step(action)
        reward = float(info.get("reward", 0.0)) if isinstance(info, dict) else 0.0
        return obs, reward, bool(done), info if isinstance(info, dict) else {}

    def get_state_summary(self) -> Dict[str, Any]:
        return self.env._obs()

    @property
    def backend_name(self) -> str:
        return "pybullet"

    def build_vision_frame(self, task_id: str, episode_id: str, timestep: int) -> VisionFrame:
        """Optional helper to create a VisionFrame placeholder."""
        return VisionFrame(
            backend=self.backend_name,
            task_id=task_id,
            episode_id=episode_id,
            timestep=timestep,
            rgb_path=None,
            depth_path=None,
            segmentation_path=None,
            camera_name="default_cam",
            metadata={"state": self.get_state_summary()},
        )

"""
PyBullet wrapper backend around DishwashingEnv (placeholder physics env).
"""
from typing import Any, Dict, Tuple, Optional

from src.physics.backends.base import PhysicsBackend
from src.physics.backends.mobility import MobilityPolicy, MobilityContext
from src.envs.dishwashing_env import DishwashingEnv
from src.config.econ_params import EconParams, load_econ_params
from src.config.internal_profile import get_internal_experiment_profile
from src.vision.interfaces import VisionFrame, compute_state_digest
from src.vision.config import load_vision_config


class PyBulletBackend(PhysicsBackend):
    def __init__(self, econ_preset: str = "toy", mobility_policy: Optional[MobilityPolicy] = None):
        profile = get_internal_experiment_profile("dishwashing")
        params = load_econ_params(profile, preset=econ_preset)
        self.env = DishwashingEnv(params)
        self.mobility_policy = mobility_policy

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Deterministically reset the PyBullet environment."""
        if seed is not None:
            import numpy as np
            np.random.seed(seed)
        return self.env.reset()

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        obs, info, done = self.env.step(action)
        info_dict = info if isinstance(info, dict) else {}
        reward = float(info_dict.get("reward", 0.0)) if isinstance(info_dict, dict) else 0.0

        if self.mobility_policy:
            ctx = MobilityContext(
                task_id="unknown",
                episode_id=info_dict.get("episode_id", ""),
                env_name=self.backend_name,
                timestep=int(info_dict.get("timestep", info_dict.get("step", 0))),
                pose={"drift_mm": info_dict.get("drift_mm", 0.0)},
                contacts={"slip_rate": info_dict.get("slip_rate", 0.0)},
                target_precision_mm=float(info_dict.get("target_precision_mm", 5.0)),
                stability_margin=float(info_dict.get("stability_margin", 1.0)),
                metadata={"raw_info": info_dict},
            )
            adjustment = self.mobility_policy.compute_adjustment(ctx)
            info_dict = dict(info_dict)
            info_dict["mobility_adjustment"] = adjustment.to_dict()
        return obs, reward, bool(done), info_dict if isinstance(info_dict, dict) else {}

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Compact snapshot used by downstream logging/vision.
        Must remain JSON-safe (floats/lists/ints only).
        """
        return self.env._obs()

    @property
    def backend_name(self) -> str:
        return "pybullet"

    def build_vision_frame(self, task_id: str, episode_id: str, timestep: int) -> VisionFrame:
        """Create a canonical VisionFrame placeholder for PyBullet."""
        cfg = load_vision_config()
        state = self.get_state_summary()
        state_digest = compute_state_digest(state)
        width, height = cfg.get("input_resolution", [224, 224])
        fov = float(cfg.get("fov_deg", 90.0))
        return VisionFrame(
            backend=self.backend_name,
            backend_id=self.backend_name,
            task_id=task_id,
            episode_id=episode_id,
            timestep=timestep,
            width=int(width),
            height=int(height),
            channels=int(cfg.get("channels", 3)),
            dtype=str(cfg.get("dtype", "uint8")),
            camera_pose={"pose": "stub", "timestep": timestep},
            camera_intrinsics={
                "resolution": [int(width), int(height)],
                "fov_deg": fov,
                "principal_point": [int(width) / 2.0, int(height) / 2.0],
            },
            camera_extrinsics={"frame": "world", "translation": [0.0, 0.0, 1.0], "rotation_rpy": [0.0, 0.0, 0.0]},
            rgb_path=None,
            depth_path=None,
            segmentation_path=None,
            camera_name="default_cam",
            state_digest=state_digest,
            metadata={
                "state": state,
                "state_digest": state_digest,
                "backend_version": "pybullet_stub_v1",
            },
        )

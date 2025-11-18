"""
Isaac stub backend to lock interface; real implementation to follow.
"""
import logging
from typing import Any, Dict, Tuple

from src.physics.backends.base import PhysicsBackend
from src.physics.backends.mobility import MobilityContext
from src.vision.interfaces import VisionFrame


class IsaacStubBackend(PhysicsBackend):
    def __init__(self, mobility_policy=None):
        self._logger = logging.getLogger("IsaacStubBackend")
        self.mobility_policy = mobility_policy

    def reset(self, seed: None = None) -> Dict[str, Any]:
        self._logger.warning("Isaac stub reset called - not implemented.")
        raise NotImplementedError("Isaac backend not implemented; use pybullet backend for now.")

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._logger.warning("Isaac stub step called - not implemented.")
        if self.mobility_policy:
            ctx = MobilityContext(
                task_id="unknown",
                episode_id="",
                env_name="isaac_stub",
                timestep=0,
                pose={},
                contacts={},
                target_precision_mm=5.0,
                stability_margin=1.0,
                metadata={},
            )
            adj = self.mobility_policy.compute_adjustment(ctx)
            return {}, 0.0, True, {"mobility_adjustment": adj.to_dict()}
        raise NotImplementedError("Isaac backend not implemented; use pybullet backend for now.")

    def get_state_summary(self) -> Dict[str, Any]:
        return {"status": "isaac_stub", "notes": "placeholder state summary"}

    @property
    def backend_name(self) -> str:
        return "isaac_stub"

    def build_vision_frame(self, task_id: str, episode_id: str, timestep: int) -> VisionFrame:
        """Construct a stub VisionFrame aligning with base contract."""
        return VisionFrame(
            backend=self.backend_name,
            task_id=task_id,
            episode_id=episode_id,
            timestep=timestep,
            camera_name="isaac_stub_cam",
            metadata={"status": "stub", "message": "Isaac vision stub"},
        )

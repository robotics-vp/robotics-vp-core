"""
Isaac stub backend to lock interface; real implementation to follow.
"""
import logging
from typing import Any, Dict, Tuple

from src.physics.backends.base import PhysicsBackend
from src.vision.interfaces import VisionFrame


class IsaacStubBackend(PhysicsBackend):
    def __init__(self):
        self._logger = logging.getLogger("IsaacStubBackend")

    def reset(self, seed: None = None) -> Dict[str, Any]:
        self._logger.warning("Isaac stub reset called - not implemented.")
        raise NotImplementedError("Isaac backend not implemented; use pybullet backend for now.")

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._logger.warning("Isaac stub step called - not implemented.")
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

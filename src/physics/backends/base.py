"""
Physics backend interface for future multi-engine support.

All implementations must be deterministic for a fixed seed and JSON-safe. State
summaries and vision frames should avoid non-serializable objects.
"""
from typing import Any, Dict, Tuple, Optional

from src.vision.interfaces import VisionFrame


class PhysicsBackend:
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset environment and return the initial low-dimensional observation/state.

        Returns a JSON-safe dict containing at minimum:
        - positions/poses/velocities as scalars or lists (no tensors)
        - task identifiers needed by vision + logging
        """
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Advance simulation by one control step.

        Returns:
          obs_summary: Dict (JSON-safe) mirroring reset() structure
          reward: float
          done: bool
          info: Dict (backend-specific diagnostics, JSON-safe)
        """
        raise NotImplementedError

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Return a compact JSON-safe snapshot of the current state.

        Must include the fields required to reconstruct policy observations
        (e.g., gripper pose, object poses, contacts, energies).
        """
        raise NotImplementedError

    def build_vision_frame(self, task_id: str, episode_id: str, timestep: int) -> VisionFrame:
        """
        Construct a canonical VisionFrame for the current backend step.

        VisionFrame must populate:
        - backend (string name)
        - task_id, episode_id, timestep
        - optional rgb/depth/segmentation paths if available
        - metadata capturing camera_name and minimal state hashes
        """
        raise NotImplementedError

    @property
    def backend_name(self) -> str:
        return self.__class__.__name__

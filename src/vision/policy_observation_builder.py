"""
Canonical construction of PolicyObservation from VisionFrame + low-dim state.

Required VisionFrame fields:
- backend (str), task_id (str), episode_id (str), timestep (int)
- optional rgb/depth/segmentation paths and camera_name
- metadata must be JSON-safe

PolicyObservation enriches the VisionLatent with state_summary and mirrors the
same task/episode/timestep identifiers.
"""
from typing import Dict, Any

from src.vision.interfaces import VisionFrame, PolicyObservation
from src.policies.interfaces import VisionEncoderPolicy


class PolicyObservationBuilder:
    """Single entry point for building PolicyObservation objects."""

    def __init__(self, encoder: VisionEncoderPolicy):
        self.encoder = encoder

    def build(self, frame: VisionFrame, state_summary: Dict[str, Any]) -> PolicyObservation:
        if hasattr(self.encoder, "encode_frame"):
            latent = self.encoder.encode_frame(frame)  # type: ignore
        elif hasattr(self.encoder, "encode"):
            latent = self.encoder.encode(frame)  # type: ignore
        else:  # pragma: no cover - defensive
            raise TypeError("Encoder must implement encode(frame)")
        return PolicyObservation(
            task_id=frame.task_id,
            episode_id=frame.episode_id,
            timestep=frame.timestep,
            latent=latent,
            state_summary=state_summary,
            metadata={"backend": frame.backend},
        )

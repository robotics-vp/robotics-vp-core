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

from src.vision.interfaces import VisionFrame, PolicyObservation, VisionLatent
from src.policies.interfaces import VisionEncoderPolicy


class PolicyObservationBuilder:
    """Single entry point for building PolicyObservation objects."""

    def __init__(self, encoder: VisionEncoderPolicy):
        self.encoder = encoder

    def encode_frame(self, frame: VisionFrame) -> VisionLatent:
        """
        Centralized VisionFrame -> VisionLatent path used by all policy heads.
        """
        if hasattr(self.encoder, "encode_frame"):
            return self.encoder.encode_frame(frame)  # type: ignore[attr-defined]
        if hasattr(self.encoder, "encode"):
            return self.encoder.encode(frame)  # type: ignore[attr-defined]
        raise TypeError("Encoder must implement encode(frame)")

    def build(self, frame: VisionFrame, state_summary: Dict[str, Any]) -> PolicyObservation:
        latent = self.encode_frame(frame)
        return PolicyObservation(
            task_id=frame.task_id,
            episode_id=frame.episode_id,
            timestep=frame.timestep,
            latent=latent,
            state_summary=state_summary,
            metadata={
                "backend": frame.backend,
                "backend_id": frame.backend_id,
                "state_digest": frame.state_digest,
                "camera_intrinsics": frame.camera_intrinsics,
                "camera_extrinsics": frame.camera_extrinsics,
            },
        )

    def build_policy_features(self, frame: VisionFrame, state_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a policy-ready feature dict shared by heuristic vs neural encoders.
        """
        obs = self.build(frame, state_summary)
        return {
            "task_id": obs.task_id,
            "episode_id": obs.episode_id,
            "timestep": obs.timestep,
            "backend": frame.backend,
            "backend_id": frame.backend_id or frame.backend,
            "state_digest": frame.state_digest,
            "vision_latent": obs.latent.to_dict(),
            "state_summary": state_summary,
            "camera_intrinsics": frame.camera_intrinsics,
            "camera_extrinsics": frame.camera_extrinsics,
            "vision_metadata": frame.metadata,
        }

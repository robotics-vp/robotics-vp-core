"""
Build PolicyObservation from VisionFrame + state summary.
"""
from typing import Dict, Any

from src.vision.interfaces import VisionFrame, PolicyObservation
from src.vision.backbone_stub import VisionBackboneStub


class PolicyObservationBuilder:
    def __init__(self, backbone: VisionBackboneStub):
        self.backbone = backbone

    def build(self, frame: VisionFrame, state_summary: Dict[str, Any]) -> PolicyObservation:
        latent = self.backbone.encode_frame(frame)
        return PolicyObservation(
            task_id=frame.task_id,
            episode_id=frame.episode_id,
            timestep=frame.timestep,
            latent=latent,
            state_summary=state_summary,
            metadata={"backend": frame.backend},
        )

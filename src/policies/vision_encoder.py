"""
Heuristic VisionEncoderPolicy wrapping the deterministic VisionBackboneStub.
"""
from typing import Any, List, Sequence

from src.policies.interfaces import VisionEncoderPolicy
from src.vision.backbone_stub import VisionBackboneStub
from src.vision.interfaces import VisionFrame, VisionLatent


class HeuristicVisionEncoderPolicy(VisionEncoderPolicy):
    def __init__(self, model_name: str = "dino-stub", latent_dim: int = 16):
        self._encoder = VisionBackboneStub(model_name=model_name, latent_dim=latent_dim)
        self.mode = "stub"
        self.backbone_name = model_name

    def encode(self, frame: Any) -> VisionLatent:
        if isinstance(frame, VisionFrame):
            return self._encoder.encode_frame(frame)
        if isinstance(frame, dict):
            return self._encoder.encode_frame(VisionFrame.from_dict(frame))
        raise TypeError("VisionEncoderPolicy.encode expects VisionFrame or dict-compatible frame")

    def batch_encode(self, frames: Sequence[Any]) -> List[VisionLatent]:
        return [self.encode(f) for f in frames]

"""
Deterministic vision backbone stub (DINO placeholder).
"""
import hashlib
from typing import List

from src.vision.interfaces import VisionFrame, VisionLatent


class VisionBackboneStub:
    def __init__(self, model_name: str = "dino-stub", latent_dim: int = 16):
        self.model_name = model_name
        self.latent_dim = latent_dim

    def encode_frame(self, frame: VisionFrame) -> VisionLatent:
        digest_src = f"{frame.backend}|{frame.task_id}|{frame.episode_id}|{frame.timestep}|{frame.rgb_path}|{frame.camera_name}|{sorted(frame.metadata.items())}"
        digest = hashlib.sha256(digest_src.encode("utf-8")).digest()
        latent_vals: List[float] = []
        for i in range(self.latent_dim):
            # Deterministic float in [0,1)
            latent_vals.append(int.from_bytes(digest[i:i+2], "big") / 65535.0)
        return VisionLatent(
            backend=frame.backend,
            task_id=frame.task_id,
            episode_id=frame.episode_id,
            timestep=frame.timestep,
            latent=latent_vals,
            model_name=self.model_name,
            metadata={"source_digest": digest_hex(digest)},
        )


def digest_hex(digest: bytes) -> str:
    return digest.hex()

"""
Deterministic vision backbone stub (DINO placeholder).
"""
import hashlib
import json
from typing import List, Optional

from src.world_model.perception_grounding.provider_contracts import (
    VisionBackboneProviderContract,
)
from src.vision.interfaces import VisionFrame, VisionLatent
from src.vision.config import load_vision_config


class VisionBackboneStub:
    def __init__(self, model_name: Optional[str] = None, latent_dim: Optional[int] = None, config_path: str = ""):
        cfg = load_vision_config(config_path)
        self.model_name = model_name or cfg.get("model_name", "dino-stub")
        self.latent_dim = latent_dim or int(cfg.get("latent_dim", 16))

    def describe_provider_contract(self) -> VisionBackboneProviderContract:
        return VisionBackboneProviderContract(
            provider_id="vision_backbone_stub",
            provider_family="stub",
            availability="available",
            provider_truth_class="stub_smoke_only",
            loading_posture="disabled",
            model_name=self.model_name,
            model_version="stub",
            weights_available=True,
            gpu_required=False,
            backbone_dim=self.latent_dim,
            projection_output_dim=self.latent_dim,
            projection_head_posture="disabled",
            expected_latency_ms=1.0,
            fallback_posture="deterministic_stub",
            fallback_reason="",
            calibration_status="not_applicable",
            metadata={
                "advisory_only": True,
                "loop_authority": "provider_advisory_only",
                "successor_surface": "perception_grounding.provider_surface",
            },
        )

    def encode_frame(self, frame: VisionFrame) -> VisionLatent:
        frame_payload = frame.to_dict()
        digest_src = json.dumps(frame_payload, sort_keys=True, separators=(",", ":"))
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
            metadata={
                "source_digest": digest_hex(digest),
                "provider_truth_class": "stub_smoke_only",
                "advisory_only": True,
                "provider_contract": self.describe_provider_contract().to_dict(),
            },
        )


def digest_hex(digest: bytes) -> str:
    return digest.hex()

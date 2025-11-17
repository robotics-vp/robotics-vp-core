import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Any

import numpy as np
from PIL import Image


@dataclass
class OpenVLAConfig:
    model_name: str = "openvla/openvla-7b"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    unnorm_key: str = "bridge_orig"
    max_action_norm: float = 1.0
    # Vision backbone configuration (optional)
    use_vision_backbone: bool = False
    vision_backbone_type: str = "dummy"  # "dummy", "dino", "clip"
    vision_backbone_model: str = "facebook/dinov2-small"


class OpenVLAController:
    def __init__(self, cfg: OpenVLAConfig = OpenVLAConfig()):
        self.cfg = cfg
        self.available = False
        self.model = None
        self.processor = None

        # Vision backbone for embedding generation (optional, additive)
        self.vision_backbone = None
        self._frame_buffer: List[Any] = []  # Buffer for episode frames
        self._embedding_log: List[Dict[str, Any]] = []  # Log of embeddings

    @classmethod
    def from_config(cls, cfg_dict: Dict[str, str]):
        cfg = OpenVLAConfig(
            model_name=cfg_dict.get("model_name", "openvla/openvla-7b"),
            device=cfg_dict.get("device", "cuda:0"),
            dtype=cfg_dict.get("dtype", "bfloat16"),
            unnorm_key=cfg_dict.get("unnorm_key", "bridge_orig"),
            max_action_norm=float(cfg_dict.get("max_action_norm", 1.0)),
        )
        return cls(cfg)

    def load_model(self):
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore

            self.processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.cfg.model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=getattr(__import__("torch"), self.cfg.dtype, None),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.cfg.device)
            self.available = True
        except Exception as e:
            logging.warning(f"OpenVLA not available ({e}); falling back to dummy actions.")
            self.available = False

        # Load vision backbone if configured (optional, soft-fail)
        if self.cfg.use_vision_backbone:
            self._load_vision_backbone()

    def _load_vision_backbone(self):
        """
        Load vision backbone for embedding generation.

        This is additive infrastructure - soft-fails to DummyBackbone if dependencies unavailable.
        Does not affect VLA action prediction or training behavior.
        """
        try:
            if self.cfg.vision_backbone_type == "dino":
                from src.vla.backbones.meta_dino_backbone import MetaDINOBackbone
                self.vision_backbone = MetaDINOBackbone(
                    model_name=self.cfg.vision_backbone_model,
                    embedding_dim=384,
                )
                logging.info(f"Loaded MetaDINOBackbone: {self.vision_backbone.name}")
            else:
                # Default to DummyBackbone
                from src.vla.backbones.dummy_backbone import DummyBackbone
                self.vision_backbone = DummyBackbone(embedding_dim=384)
                logging.info(f"Loaded DummyBackbone: {self.vision_backbone.name}")
        except Exception as e:
            logging.warning(f"Vision backbone load failed ({e}); embedding generation disabled.")
            self.vision_backbone = None

    def predict_action(self, image: Image.Image, instruction: str) -> Dict[str, float]:
        if not self.available or self.model is None or self.processor is None:
            result = {
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "droll": 0.0,
                "dpitch": 0.0,
                "dyaw": 0.0,
                "gripper": 0.0,
                "vla_available": False,
                "raw_action": [0.0] * 7,
            }
        else:
            import torch

            prompt = f"In: {instruction}\nOut:"
            inputs = self.processor(prompt, image).to(self.cfg.device)
            if self.cfg.dtype == "bfloat16":
                inputs = {k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.no_grad():
                action = self.model.predict_action(**inputs, unnorm_key=self.cfg.unnorm_key, do_sample=False)
            raw = np.array(action).astype(float).tolist()
            clipped = np.clip(raw, -self.cfg.max_action_norm, self.cfg.max_action_norm)
            result = {
                "dx": float(clipped[0]),
                "dy": float(clipped[1]),
                "dz": float(clipped[2]),
                "droll": float(clipped[3]),
                "dpitch": float(clipped[4]),
                "dyaw": float(clipped[5]),
                "gripper": float(clipped[6]),
                "vla_available": True,
                "raw_action": raw,
            }

        # Optionally buffer frame for episode embedding (additive, logging only)
        if self.vision_backbone is not None:
            self._frame_buffer.append(image)
            # Log per-frame embedding if desired (for debugging/analysis)
            if len(self._frame_buffer) % 10 == 0:  # Sample every 10th frame
                try:
                    frame_emb = self.vision_backbone.encode_frame(image)
                    self._embedding_log.append({
                        "frame_idx": len(self._frame_buffer) - 1,
                        "embedding_norm": float(np.linalg.norm(frame_emb)),
                    })
                except Exception as e:
                    logging.debug(f"Frame embedding failed: {e}")

        return result

    def start_episode(self):
        """
        Reset frame buffer for new episode.

        Call this at the start of each episode to begin collecting frames
        for episode embedding computation.
        """
        self._frame_buffer = []
        self._embedding_log = []

    def end_episode(self) -> Optional[np.ndarray]:
        """
        Compute episode embedding from buffered frames.

        Call this at the end of each episode to get the pooled embedding.
        Returns None if vision backbone is not available or no frames were collected.

        This is purely for logging/analysis - does not affect training behavior.
        """
        if self.vision_backbone is None:
            return None

        if len(self._frame_buffer) == 0:
            logging.debug("No frames buffered for episode embedding.")
            return None

        try:
            # Use vision backbone to encode full sequence
            episode_embedding = self.vision_backbone.encode_sequence(self._frame_buffer)
            logging.info(
                f"Episode embedding computed: dim={len(episode_embedding)}, "
                f"frames={len(self._frame_buffer)}, norm={np.linalg.norm(episode_embedding):.4f}"
            )
            return episode_embedding
        except Exception as e:
            logging.warning(f"Episode embedding computation failed: {e}")
            return None

    def get_embedding_log(self) -> List[Dict[str, Any]]:
        """Get per-frame embedding statistics for debugging."""
        return self._embedding_log

    def has_vision_backbone(self) -> bool:
        """Check if vision backbone is available."""
        return self.vision_backbone is not None


if __name__ == "__main__":
    from PIL import Image

    controller = OpenVLAController()
    controller.load_model()
    img = Image.new("RGB", (256, 256), color="gray")
    out = controller.predict_action(img, "Open the drawer without hitting the vase.")
    print("VLA available:", out.get("vla_available"))
    print("Action:", out)

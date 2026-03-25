from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


def _normalize_backend_policy(value: Optional[str]) -> str:
    policy = str(value or "auto").strip().lower()
    if policy not in {"auto", "real", "disabled", "stub"}:
        return "auto"
    return policy


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass
class OpenVLAConfig:
    model_name: str = "openvla/openvla-7b"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    unnorm_key: str = "bridge_orig"
    max_action_norm: float = 1.0
    backend_policy: str = "auto"
    use_vision_backbone: bool = False
    vision_backbone_type: str = "dummy"  # "dummy", "dino", "clip"
    vision_backbone_model: str = "facebook/dinov2-small"
    vision_backbone_policy: str = "auto"


class OpenVLAController:
    def __init__(self, cfg: OpenVLAConfig = OpenVLAConfig()):
        self.cfg = cfg
        self.available = False
        self.model = None
        self.processor = None
        self.backend_selected = "unavailable"
        self.failure_reason: Optional[str] = None
        self.vision_backbone = None
        self.vision_backbone_selected = "disabled"
        self.vision_backbone_failure_reason: Optional[str] = None
        self._frame_buffer: List[Any] = []
        self._embedding_log: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls, cfg_dict: Dict[str, str]):
        cfg = OpenVLAConfig(
            model_name=cfg_dict.get("model_name", "openvla/openvla-7b"),
            device=cfg_dict.get("device", "cuda:0"),
            dtype=cfg_dict.get("dtype", "bfloat16"),
            unnorm_key=cfg_dict.get("unnorm_key", "bridge_orig"),
            max_action_norm=float(cfg_dict.get("max_action_norm", 1.0)),
            backend_policy=cfg_dict.get("backend_policy", "auto"),
            use_vision_backbone=_parse_bool(cfg_dict.get("use_vision_backbone", False)),
            vision_backbone_type=cfg_dict.get("vision_backbone_type", "dummy"),
            vision_backbone_model=cfg_dict.get("vision_backbone_model", "facebook/dinov2-small"),
            vision_backbone_policy=cfg_dict.get("vision_backbone_policy", "auto"),
        )
        return cls(cfg)

    def backend_status(self) -> Dict[str, Any]:
        return {
            "model_name": self.cfg.model_name,
            "device": self.cfg.device,
            "dtype": self.cfg.dtype,
            "backend_policy": _normalize_backend_policy(self.cfg.backend_policy),
            "backend_selected": self.backend_selected,
            "available": bool(self.available),
            "failure_reason": self.failure_reason,
            "vision_backbone_enabled": bool(self.cfg.use_vision_backbone),
            "vision_backbone_type": self.cfg.vision_backbone_type,
            "vision_backbone_model": self.cfg.vision_backbone_model,
            "vision_backbone_policy": _normalize_backend_policy(self.cfg.vision_backbone_policy),
            "vision_backbone_selected": self.vision_backbone_selected,
            "vision_backbone_failure_reason": self.vision_backbone_failure_reason,
            "vision_backbone_available": bool(self.vision_backbone is not None),
        }

    def load_model(self):
        policy = _normalize_backend_policy(self.cfg.backend_policy)
        self.backend_selected = "unavailable"
        self.failure_reason = None
        self.available = False
        self.model = None
        self.processor = None

        if policy == "disabled":
            self.backend_selected = "disabled"
            self.failure_reason = "backend_disabled"
        elif policy == "stub":
            self.backend_selected = "stub"
            self.failure_reason = "explicit_stub_requested"
        else:
            try:
                from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[import-not-found]

                self.processor = AutoProcessor.from_pretrained(self.cfg.model_name, trust_remote_code=True)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.cfg.model_name,
                    attn_implementation="flash_attention_2",
                    torch_dtype=getattr(__import__("torch"), self.cfg.dtype, None),
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).to(self.cfg.device)
                self.available = True
                self.backend_selected = "real"
            except Exception as exc:
                self.backend_selected = "unavailable"
                self.failure_reason = str(exc)
                logging.warning("OpenVLA unavailable (%s); backend stays unavailable.", exc)
                if policy == "real":
                    raise RuntimeError(f"OpenVLA real backend required but unavailable: {exc}") from exc

        if self.cfg.use_vision_backbone:
            self._load_vision_backbone()
        else:
            self.vision_backbone = None
            self.vision_backbone_selected = "disabled"
            self.vision_backbone_failure_reason = None

    def _load_vision_backbone(self):
        self.vision_backbone = None
        self.vision_backbone_selected = "unavailable"
        self.vision_backbone_failure_reason = None
        policy = _normalize_backend_policy(self.cfg.vision_backbone_policy)

        if policy == "disabled":
            self.vision_backbone_selected = "disabled"
            self.vision_backbone_failure_reason = "backend_disabled"
            return

        try:
            if self.cfg.vision_backbone_type == "dino":
                from src.vla.backbones.meta_dino_backbone import MetaDINOBackbone

                backbone = MetaDINOBackbone(
                    model_name=self.cfg.vision_backbone_model,
                    device=self.cfg.device,
                    backend_policy=policy,
                )
                if backbone.available or backbone.backend_selected == "stub":
                    self.vision_backbone = backbone
                self.vision_backbone_selected = backbone.backend_selected
                self.vision_backbone_failure_reason = backbone.failure_reason
                if self.vision_backbone is not None:
                    logging.info("Loaded %s", backbone.name)
                return

            if self.cfg.vision_backbone_type == "dummy":
                if policy != "stub":
                    self.vision_backbone_selected = "unavailable"
                    self.vision_backbone_failure_reason = "dummy_backbone_requires_stub_policy"
                    if policy == "real":
                        raise RuntimeError(self.vision_backbone_failure_reason)
                    return
                from src.vla.backbones.dummy_backbone import DummyBackbone

                self.vision_backbone = DummyBackbone(embedding_dim=384)
                self.vision_backbone_selected = "stub"
                self.vision_backbone_failure_reason = "explicit_stub_requested"
                logging.info("Loaded DummyBackbone stub")
                return

            self.vision_backbone_selected = "unavailable"
            self.vision_backbone_failure_reason = f"unsupported_backbone_type:{self.cfg.vision_backbone_type}"
            if policy == "real":
                raise RuntimeError(self.vision_backbone_failure_reason)
        except Exception as exc:
            self.vision_backbone = None
            self.vision_backbone_selected = "unavailable"
            self.vision_backbone_failure_reason = str(exc)
            logging.warning("Vision backbone unavailable (%s); embedding generation disabled.", exc)
            if policy == "real":
                raise

    def predict_action(self, image: Image.Image, instruction: str) -> Dict[str, Any]:
        result: Dict[str, Any]
        if self.backend_selected == "stub":
            result = {
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "droll": 0.0,
                "dpitch": 0.0,
                "dyaw": 0.0,
                "gripper": 0.0,
                "vla_available": False,
                "confidence": 0.0,
                "source": "openvla_stub",
                "fallback_mode": "explicit_stub",
                "raw_action": [0.0] * 7,
            }
        elif not self.available or self.model is None or self.processor is None:
            result = {
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "droll": 0.0,
                "dpitch": 0.0,
                "dyaw": 0.0,
                "gripper": 0.0,
                "vla_available": False,
                "confidence": 0.0,
                "source": self.cfg.model_name,
                "fallback_mode": self.failure_reason or self.backend_selected or "teacher_unavailable",
                "raw_action": [0.0] * 7,
            }
        else:
            import torch

            prompt = f"In: {instruction}\nOut:"
            inputs = self.processor(prompt, image).to(self.cfg.device)
            if self.cfg.dtype == "bfloat16":
                inputs = {
                    key: value.to(torch.bfloat16) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
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
                "confidence": 0.35,
                "source": self.cfg.model_name,
                "fallback_mode": "teacher_available",
                "raw_action": raw,
            }

        result.update(
            {
                "backend_policy": _normalize_backend_policy(self.cfg.backend_policy),
                "backend_selected": self.backend_selected,
                "failure_reason": self.failure_reason or "",
                "vision_backbone_policy": _normalize_backend_policy(self.cfg.vision_backbone_policy),
                "vision_backbone_selected": self.vision_backbone_selected,
                "vision_backbone_failure_reason": self.vision_backbone_failure_reason or "",
            }
        )

        if self.vision_backbone is not None:
            self._frame_buffer.append(image)
            if len(self._frame_buffer) % 10 == 0:
                try:
                    frame_emb = self.vision_backbone.encode_frame(image)
                    self._embedding_log.append(
                        {
                            "frame_idx": len(self._frame_buffer) - 1,
                            "embedding_norm": float(np.linalg.norm(frame_emb)),
                            "vision_backbone_selected": self.vision_backbone_selected,
                        }
                    )
                except Exception as exc:
                    logging.debug("Frame embedding failed: %s", exc)

        return result

    def start_episode(self):
        self._frame_buffer = []
        self._embedding_log = []

    def end_episode(self) -> Optional[np.ndarray]:
        if self.vision_backbone is None:
            return None
        if len(self._frame_buffer) == 0:
            logging.debug("No frames buffered for episode embedding.")
            return None

        try:
            episode_embedding = self.vision_backbone.encode_sequence(self._frame_buffer)
            logging.info(
                "Episode embedding computed: dim=%s, frames=%s, norm=%.4f",
                len(episode_embedding),
                len(self._frame_buffer),
                np.linalg.norm(episode_embedding),
            )
            return episode_embedding
        except Exception as exc:
            logging.warning("Episode embedding computation failed: %s", exc)
            return None

    def get_embedding_log(self) -> List[Dict[str, Any]]:
        return self._embedding_log

    def has_vision_backbone(self) -> bool:
        return self.vision_backbone is not None


if __name__ == "__main__":
    controller = OpenVLAController()
    controller.load_model()
    img = Image.new("RGB", (256, 256), color="gray")
    out = controller.predict_action(img, "Open the drawer without hitting the vase.")
    print("VLA available:", out.get("vla_available"))
    print("Action:", out)

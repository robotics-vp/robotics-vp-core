"""Meta DINO vision backbone with explicit backend policy."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Sequence

import numpy as np

from src.vla.backbones.dummy_backbone import DummyBackbone
from src.vla.vision_backbone import VisionBackbone, normalize_embedding


def _normalize_backend_policy(value: Optional[str]) -> str:
    policy = str(value or "auto").strip().lower()
    if policy not in {"auto", "real", "disabled", "stub"}:
        return "auto"
    return policy


class MetaDINOBackbone(VisionBackbone):
    """Meta DINO backbone that is real-or-unavailable unless stub is explicit."""

    def __init__(
        self,
        model_name: str = "facebook/dino-vitb16",
        device: str = "cuda",
        enabled: bool = True,
        backend_policy: str = "auto",
    ):
        self._model_name = model_name
        self._device = device
        self._enabled = enabled
        self._backend_policy = _normalize_backend_policy(backend_policy)
        self._available = False
        self._model: Any = None
        self._processor: Any = None
        self._fallback: Optional[DummyBackbone] = None
        self._backend_type = ""
        self._backend_selected = "unavailable"
        self._failure_reason: Optional[str] = None
        self._embedding_dim_actual = 768

        if not enabled or self._backend_policy == "disabled":
            self._backend_selected = "disabled"
            self._failure_reason = "backend_disabled"
            return

        if self._backend_policy == "stub":
            self._fallback = DummyBackbone(embedding_dim=768)
            self._backend_selected = "stub"
            self._failure_reason = "explicit_stub_requested"
            return

        self._try_load_model()

    def _try_load_model(self) -> None:
        last_error: Optional[str] = None

        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel  # type: ignore[import-not-found]

            self._model = AutoModel.from_pretrained(self._model_name)
            self._processor = AutoFeatureExtractor.from_pretrained(self._model_name)
            self._model.to(self._device)
            self._model.eval()

            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224).to(self._device)
                output = self._model(dummy)
                if hasattr(output, "last_hidden_state"):
                    self._embedding_dim_actual = int(output.last_hidden_state.shape[-1])
                else:
                    self._embedding_dim_actual = int(output.shape[-1])

            self._available = True
            self._backend_type = "transformers"
            self._backend_selected = "real"
            self._failure_reason = None
            return
        except ImportError as exc:
            last_error = f"transformers_import_error:{exc}"
            warnings.warn(f"transformers not available: {exc}")
        except Exception as exc:
            last_error = f"transformers_load_error:{exc}"
            warnings.warn(f"Failed to load {self._model_name} via transformers: {exc}")

        try:
            import torch
            import timm  # type: ignore[import-not-found]

            timm_name = self._model_name.replace("facebook/", "").replace("-", "_")
            if "dino" in timm_name.lower():
                timm_name = "vit_base_patch16_224.dino"

            self._model = timm.create_model(timm_name, pretrained=True, num_classes=0)
            self._model.to(self._device)
            self._model.eval()

            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224).to(self._device)
                output = self._model(dummy)
                self._embedding_dim_actual = int(output.shape[-1])

            self._available = True
            self._backend_type = "timm"
            self._backend_selected = "real"
            self._failure_reason = None
            return
        except ImportError as exc:
            last_error = f"timm_import_error:{exc}"
            warnings.warn(f"timm not available: {exc}")
        except Exception as exc:
            last_error = f"timm_load_error:{exc}"
            warnings.warn(f"Failed to load via timm: {exc}")

        self._available = False
        self._backend_selected = "unavailable"
        self._failure_reason = last_error or "model_unavailable"
        warnings.warn(
            f"MetaDINOBackbone: Could not load {self._model_name}. "
            "Backend remains unavailable."
        )
        if self._backend_policy == "real":
            raise RuntimeError(self._failure_reason)

    @property
    def embedding_dim(self) -> int:
        if self._available:
            return self._embedding_dim_actual
        if self._fallback is not None:
            return self._fallback.embedding_dim
        return self._embedding_dim_actual

    @property
    def available(self) -> bool:
        return self._available

    @property
    def backend_selected(self) -> str:
        return self._backend_selected

    @property
    def failure_reason(self) -> Optional[str]:
        return self._failure_reason

    @property
    def name(self) -> str:
        if self._available:
            return f"MetaDINO({self._model_name})"
        if self._backend_selected == "stub":
            return "MetaDINO(stub=DummyBackbone)"
        return f"MetaDINO({self._backend_selected})"

    def status(self) -> Dict[str, Any]:
        return {
            "model_name": self._model_name,
            "device": self._device,
            "backend_policy": self._backend_policy,
            "backend_selected": self._backend_selected,
            "available": bool(self._available),
            "backend_type": self._backend_type,
            "failure_reason": self._failure_reason,
            "embedding_dim": int(self.embedding_dim),
        }

    def _raise_unavailable(self) -> None:
        raise RuntimeError(
            "MetaDINOBackbone unavailable "
            f"(policy={self._backend_policy}, selected={self._backend_selected}, "
            f"reason={self._failure_reason or 'unknown'})"
        )

    def _preprocess_for_dino(self, image: Any) -> Any:
        import torch

        try:
            from PIL import Image as PILImage

            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                image = PILImage.fromarray(image)
        except ImportError:
            pass

        if self._processor is not None:
            inputs = self._processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].to(self._device)

        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = PILImage.fromarray(image)

        image = image.convert("RGB").resize((224, 224))
        arr = np.array(image).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self._device)

    def encode_frame(self, image: Any) -> np.ndarray:
        if not self._available:
            if self._fallback is not None:
                return self._fallback.encode_frame(image)
            self._raise_unavailable()

        import torch

        with torch.no_grad():
            inputs = self._preprocess_for_dino(image)
            output = self._model(inputs)
            if hasattr(output, "last_hidden_state"):
                embedding = output.last_hidden_state[:, 0, :].cpu().numpy()[0]
            elif hasattr(output, "pooler_output"):
                embedding = output.pooler_output.cpu().numpy()[0]
            else:
                embedding = output.cpu().numpy()[0]
        return normalize_embedding(embedding.astype(np.float32))

    def encode_sequence(self, frames: Sequence[Any]) -> np.ndarray:
        if not self._available:
            if self._fallback is not None:
                return self._fallback.encode_sequence(frames)
            self._raise_unavailable()

        if len(frames) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        embeddings = np.stack([self.encode_frame(frame) for frame in frames])
        mean_embedding = embeddings.mean(axis=0)
        return normalize_embedding(mean_embedding)

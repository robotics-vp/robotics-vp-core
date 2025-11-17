"""
Meta DINO Vision Backbone.

Wraps Meta's DINO (self-DIstillation with NO labels) vision transformer.
Soft-fails to DummyBackbone if dependencies are unavailable.
"""

import warnings
from typing import Any, Sequence, Optional

import numpy as np

from src.vla.vision_backbone import VisionBackbone, normalize_embedding
from src.vla.backbones.dummy_backbone import DummyBackbone


class MetaDINOBackbone(VisionBackbone):
    """
    Meta DINO Vision Backbone.

    Wraps DINO (or DINOv2) from Meta/Facebook for visual embeddings.
    Uses HuggingFace transformers or timm for model loading.

    If dependencies are unavailable, falls back to DummyBackbone internally.
    """

    def __init__(
        self,
        model_name: str = "facebook/dino-vitb16",
        device: str = "cuda",
        enabled: bool = True,
    ):
        """
        Initialize Meta DINO backbone.

        Args:
            model_name: HuggingFace model name or timm model name
                Options: "facebook/dino-vitb16", "facebook/dinov2-base", etc.
            device: Device to run model on ("cuda", "cpu")
            enabled: If False, skip loading and use dummy backend
        """
        self._model_name = model_name
        self._device = device
        self._enabled = enabled
        self._available = False
        self._model = None
        self._processor = None
        self._fallback = DummyBackbone(embedding_dim=768)  # Match ViT-B/16 dim

        if not enabled:
            warnings.warn("MetaDINOBackbone disabled, using DummyBackbone fallback")
            return

        # Try to load the model
        self._try_load_model()

    def _try_load_model(self):
        """Attempt to load DINO model with soft failure."""
        # Try HuggingFace transformers first
        try:
            import torch
            from transformers import AutoModel, AutoFeatureExtractor

            self._model = AutoModel.from_pretrained(self._model_name)
            self._processor = AutoFeatureExtractor.from_pretrained(self._model_name)
            self._model.to(self._device)
            self._model.eval()

            # Get actual embedding dimension
            with torch.no_grad():
                # Create dummy input to get output shape
                dummy = torch.randn(1, 3, 224, 224).to(self._device)
                output = self._model(dummy)
                if hasattr(output, "last_hidden_state"):
                    # Take CLS token
                    self._embedding_dim_actual = output.last_hidden_state.shape[-1]
                else:
                    self._embedding_dim_actual = output.shape[-1]

            self._available = True
            self._backend_type = "transformers"
            print(f"MetaDINOBackbone: Loaded {self._model_name} via transformers")
            return

        except ImportError as e:
            warnings.warn(f"transformers not available: {e}")
        except Exception as e:
            warnings.warn(f"Failed to load {self._model_name} via transformers: {e}")

        # Try timm as fallback
        try:
            import torch
            import timm

            # Map model names
            timm_name = self._model_name.replace("facebook/", "").replace("-", "_")
            if "dino" in timm_name.lower():
                timm_name = f"vit_base_patch16_224.dino"  # Common DINO model

            self._model = timm.create_model(timm_name, pretrained=True, num_classes=0)
            self._model.to(self._device)
            self._model.eval()

            # Get embedding dim
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224).to(self._device)
                output = self._model(dummy)
                self._embedding_dim_actual = output.shape[-1]

            self._available = True
            self._backend_type = "timm"
            print(f"MetaDINOBackbone: Loaded {timm_name} via timm")
            return

        except ImportError as e:
            warnings.warn(f"timm not available: {e}")
        except Exception as e:
            warnings.warn(f"Failed to load via timm: {e}")

        # All attempts failed
        warnings.warn(
            f"MetaDINOBackbone: Could not load {self._model_name}. "
            f"Falling back to DummyBackbone. "
            f"Install transformers/timm and ensure model is available."
        )
        self._available = False

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        if self._available:
            return self._embedding_dim_actual
        return self._fallback.embedding_dim

    @property
    def available(self) -> bool:
        """Check if real DINO model is available."""
        return self._available

    @property
    def name(self) -> str:
        """Return backbone name."""
        if self._available:
            return f"MetaDINO({self._model_name})"
        return f"MetaDINO(fallback=DummyBackbone)"

    def _preprocess_for_dino(self, image: Any) -> "torch.Tensor":
        """Preprocess image for DINO model."""
        import torch

        # Handle PIL Image
        try:
            from PIL import Image as PILImage

            if isinstance(image, np.ndarray):
                # Convert numpy to PIL
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                image = PILImage.fromarray(image)
        except ImportError:
            pass

        # Use processor if available (transformers)
        if self._processor is not None:
            inputs = self._processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].to(self._device)

        # Manual preprocessing for timm
        from PIL import Image as PILImage

        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = PILImage.fromarray(image)

        # Resize to 224x224
        image = image.convert("RGB").resize((224, 224))
        arr = np.array(image).astype(np.float32) / 255.0

        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        arr = (arr - mean) / std

        # To tensor: (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self._device)

    def encode_frame(self, image: Any) -> np.ndarray:
        """
        Encode a single frame into embedding.

        Args:
            image: Input image (np.ndarray or PIL.Image)

        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        if not self._available:
            return self._fallback.encode_frame(image)

        import torch

        with torch.no_grad():
            inputs = self._preprocess_for_dino(image)
            output = self._model(inputs)

            # Extract embedding
            if hasattr(output, "last_hidden_state"):
                # Transformers: take CLS token (first token)
                embedding = output.last_hidden_state[:, 0, :].cpu().numpy()[0]
            elif hasattr(output, "pooler_output"):
                embedding = output.pooler_output.cpu().numpy()[0]
            else:
                # timm: already pooled
                embedding = output.cpu().numpy()[0]

        return normalize_embedding(embedding.astype(np.float32))

    def encode_sequence(self, frames: Sequence[Any]) -> np.ndarray:
        """
        Encode sequence of frames via mean pooling.

        Args:
            frames: List of images

        Returns:
            Mean of per-frame embeddings (normalized)
        """
        if not self._available:
            return self._fallback.encode_sequence(frames)

        if len(frames) == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        embeddings = np.stack([self.encode_frame(f) for f in frames])
        mean_embedding = embeddings.mean(axis=0)
        return normalize_embedding(mean_embedding)

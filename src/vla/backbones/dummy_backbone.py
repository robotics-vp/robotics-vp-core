"""
Dummy Vision Backbone for testing and fallback.

Uses a deterministic linear projection to create stable embeddings
without requiring any ML frameworks.
"""

from typing import Any, Sequence

import numpy as np

from src.vla.vision_backbone import VisionBackbone, normalize_embedding


class DummyBackbone(VisionBackbone):
    """
    Dummy vision backbone for testing and fallback.

    - Accepts frames as numpy arrays or PIL images
    - Resizes to fixed resolution (64x64)
    - Flattens to float32
    - Uses deterministic linear projection (fixed random seed)
    - encode_sequence = mean of per-frame embeddings

    This is just to exercise the interface and wiring.
    """

    def __init__(self, embedding_dim: int = 128, seed: int = 42):
        """
        Initialize dummy backbone.

        Args:
            embedding_dim: Output embedding dimension
            seed: Random seed for deterministic projection matrix
        """
        self._embedding_dim = embedding_dim
        self._seed = seed
        self._target_size = (64, 64)

        # Create deterministic projection matrix
        rng = np.random.default_rng(seed)
        input_dim = self._target_size[0] * self._target_size[1] * 3
        self._projection = rng.standard_normal((input_dim, embedding_dim)).astype(np.float32)

        # Normalize projection matrix rows for stability
        norms = np.linalg.norm(self._projection, axis=0, keepdims=True)
        self._projection = self._projection / (norms + 1e-8)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self._embedding_dim

    def _preprocess_image(self, image: Any) -> np.ndarray:
        """
        Preprocess image to fixed size float32 array.

        Args:
            image: np.ndarray or PIL.Image

        Returns:
            Flattened float32 array of shape (64*64*3,)
        """
        # Handle PIL Image
        try:
            from PIL import Image as PILImage

            if isinstance(image, PILImage.Image):
                image = np.array(image.convert("RGB").resize(self._target_size))
        except ImportError:
            pass

        # Ensure numpy array
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray or PIL.Image, got {type(image)}")

        # Handle grayscale
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        # Resize if needed (simple nearest-neighbor)
        if image.shape[:2] != self._target_size:
            image = self._resize_image(image)

        # Ensure 3 channels
        if image.shape[-1] != 3:
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[:, :, :3]  # Drop alpha
            else:
                raise ValueError(f"Unexpected number of channels: {image.shape[-1]}")

        # Normalize to [0, 1] and flatten
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        return image.flatten()

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Simple nearest-neighbor resize."""
        h, w = image.shape[:2]
        th, tw = self._target_size

        # Compute scaling
        y_indices = np.floor(np.arange(th) * h / th).astype(int)
        x_indices = np.floor(np.arange(tw) * w / tw).astype(int)

        # Resample
        resized = image[y_indices][:, x_indices]
        return resized

    def encode_frame(self, image: Any) -> np.ndarray:
        """
        Encode a single frame into embedding.

        Args:
            image: Input image (np.ndarray or PIL.Image)

        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        flat = self._preprocess_image(image)
        embedding = flat @ self._projection
        return normalize_embedding(embedding)

    def encode_sequence(self, frames: Sequence[Any]) -> np.ndarray:
        """
        Encode sequence of frames via mean pooling.

        Args:
            frames: List of images

        Returns:
            Mean of per-frame embeddings
        """
        if len(frames) == 0:
            return np.zeros(self._embedding_dim, dtype=np.float32)

        embeddings = np.stack([self.encode_frame(f) for f in frames])
        mean_embedding = embeddings.mean(axis=0)
        return normalize_embedding(mean_embedding)

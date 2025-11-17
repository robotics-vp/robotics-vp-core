"""
Vision Backbone Abstraction.

Provides a unified interface for vision encoders (DINO, CLIP, etc.)
that can feed embeddings into:
- VLA / OpenVLA
- Datapacks (episode_embedding)
- Valuation (novelty/regime features)

This is additive infrastructure - no changes to Phase B math, rewards, or RL loops.
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np


class VisionBackbone(ABC):
    """
    Abstract base class for vision backbones.

    Implementations should encode images into fixed-size embeddings
    that can be used for:
    - Novelty detection (embedding distance in datapack valuation)
    - Regime classification (clustering episodes by visual features)
    - VLA action grounding (visual context for action prediction)
    """

    @abstractmethod
    def encode_frame(self, image: Any) -> np.ndarray:
        """
        Encode a single frame into a 1D embedding vector.

        Args:
            image: Input image, can be:
                - np.ndarray of shape (H, W, 3) or (H, W) for grayscale
                - PIL.Image.Image

        Returns:
            np.ndarray of shape (embedding_dim,) - 1D embedding vector

        Note:
            The embedding should be normalized and stable across runs
            for the same input (deterministic).
        """
        pass

    @abstractmethod
    def encode_sequence(self, frames: Sequence[Any]) -> np.ndarray:
        """
        Encode a sequence of frames into a pooled embedding.

        Args:
            frames: Sequence of images (list of np.ndarray or PIL.Image)

        Returns:
            np.ndarray of shape (embedding_dim,) - pooled embedding

        Note:
            Default implementation is mean pooling over per-frame embeddings.
            Subclasses may override with temporal models (transformers, RNNs).
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vector."""
        pass

    @property
    def name(self) -> str:
        """Return the name of the backbone for logging."""
        return self.__class__.__name__


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    L2 normalize an embedding vector.

    Args:
        embedding: 1D numpy array

    Returns:
        Normalized embedding (unit norm)
    """
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

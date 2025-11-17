from typing import Any
import numpy as np


class DinoEncoder:
    """Thin wrapper around DINO/vision backbone semantic features."""

    def encode(self, features: Any) -> np.ndarray:
        return np.array(features, dtype=float)

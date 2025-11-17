from typing import Any
import numpy as np


class OpenVLAEncoder:
    """Thin wrapper around OpenVLA affordance outputs."""

    def encode(self, features: Any) -> np.ndarray:
        # Placeholder: assume features already a numpy array
        return np.array(features, dtype=float)

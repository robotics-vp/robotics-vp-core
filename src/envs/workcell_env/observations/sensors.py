"""
Sensor stubs for workcell environments.
"""
from __future__ import annotations

from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy optional for stubs
    np = None


class _BaseSensor:
    """Base class for sensor stubs producing placeholder frames."""

    def __init__(
        self,
        *,
        width: int = 64,
        height: int = 64,
        channels: int = 3,
        dtype: str = "uint8",
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.dtype = dtype

    def capture(self) -> Any:
        """Return a placeholder frame filled with zeros."""
        if np is not None:
            return np.zeros((self.height, self.width, self.channels), dtype=self.dtype)
        return [
            [[0 for _ in range(self.channels)] for _ in range(self.width)]
            for _ in range(self.height)
        ]


class RGBSensor(_BaseSensor):
    """Stub RGB sensor returning zero-valued images."""

    def __init__(self, *, width: int = 64, height: int = 64) -> None:
        super().__init__(width=width, height=height, channels=3, dtype="uint8")


class DepthSensor(_BaseSensor):
    """Stub depth sensor returning zero-valued depth maps."""

    def __init__(self, *, width: int = 64, height: int = 64) -> None:
        super().__init__(width=width, height=height, channels=1, dtype="float32")


class SegmentationSensor(_BaseSensor):
    """Stub segmentation sensor returning zero-valued labels."""

    def __init__(self, *, width: int = 64, height: int = 64) -> None:
        super().__init__(width=width, height=height, channels=1, dtype="int32")


__all__ = ["RGBSensor", "DepthSensor", "SegmentationSensor"]

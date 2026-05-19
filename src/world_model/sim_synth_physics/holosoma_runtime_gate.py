"""Holosoma runtime availability gates.

Importability is enough for local deploy-path smoke tests, but it is not enough
to claim that the full Holosoma simulator/runtime can execute WM episodes.
"""

from __future__ import annotations

import importlib.util
import os


HOLOSOMA_RUNTIME_ENABLE_ENV = "ROBOTICS_VP_ENABLE_HOLOSOMA_RUNTIME"


def holosoma_importable() -> bool:
    try:
        return importlib.util.find_spec("holosoma") is not None
    except Exception:
        return False


def holosoma_runtime_enabled() -> bool:
    enabled = str(os.environ.get(HOLOSOMA_RUNTIME_ENABLE_ENV, "") or "").strip().lower()
    return enabled in {"1", "true", "yes", "on"} and holosoma_importable()


__all__ = [
    "HOLOSOMA_RUNTIME_ENABLE_ENV",
    "holosoma_importable",
    "holosoma_runtime_enabled",
]

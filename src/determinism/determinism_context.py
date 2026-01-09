"""Determinism context for reproducible training and evaluation.

Records all seeds, versions, and device info for provenance.
Wraps existing determinism utilities with richer context.
"""
from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.utils.config_digest import sha256_json


@dataclass
class DeterminismContext:
    """Context capturing all determinism-relevant state.

    Records seeds, versions, and device info for reproducibility.
    """

    # Seeds
    seed_python: int = 0
    seed_numpy: int = 0
    seed_torch: int = 0
    seed_env: Optional[int] = None

    # Versions
    python_version: str = ""
    numpy_version: str = ""
    torch_version: str = ""

    # Device
    device: str = "cpu"
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None

    # Flags
    deterministic_algorithms: bool = False
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = False

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "seeds": {
                "python": self.seed_python,
                "numpy": self.seed_numpy,
                "torch": self.seed_torch,
                "env": self.seed_env,
            },
            "versions": {
                "python": self.python_version,
                "numpy": self.numpy_version,
                "torch": self.torch_version,
            },
            "device": {
                "device": self.device,
                "cuda_available": self.cuda_available,
                "cuda_version": self.cuda_version,
                "cudnn_version": self.cudnn_version,
            },
            "flags": {
                "deterministic_algorithms": self.deterministic_algorithms,
                "cudnn_deterministic": self.cudnn_deterministic,
                "cudnn_benchmark": self.cudnn_benchmark,
            },
            "created_at": self.created_at,
        }

    def sha256(self) -> str:
        """Compute SHA-256 of context."""
        return sha256_json(self.to_dict())

    def seed_bundle(self) -> Dict[str, int]:
        """Get seed bundle for manifest."""
        bundle = {
            "python": self.seed_python,
            "numpy": self.seed_numpy,
            "torch": self.seed_torch,
        }
        if self.seed_env is not None:
            bundle["env"] = self.seed_env
        return bundle


# Global context
_current_context: Optional[DeterminismContext] = None


def set_determinism(
    seed: int = 42,
    seed_env: Optional[int] = None,
    strict: bool = False,
) -> DeterminismContext:
    """Set deterministic behavior and record context.

    Args:
        seed: Base seed for all RNGs
        seed_env: Optional separate seed for environment
        strict: If True, enable torch deterministic algorithms

    Returns:
        DeterminismContext with recorded state
    """
    global _current_context

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic flags
    deterministic_algorithms = False
    if strict:
        try:
            torch.use_deterministic_algorithms(True)
            deterministic_algorithms = True
        except Exception:
            pass

    cudnn_deterministic = False
    cudnn_benchmark = False
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        cudnn_deterministic = True

    # Build context
    cuda_version = None
    cudnn_version = None
    if torch.cuda.is_available():
        try:
            cuda_version = torch.version.cuda
            cudnn_version = str(torch.backends.cudnn.version())
        except Exception:
            pass

    context = DeterminismContext(
        seed_python=seed,
        seed_numpy=seed,
        seed_torch=seed,
        seed_env=seed_env,
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        torch_version=torch.__version__,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cuda_available=torch.cuda.is_available(),
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        deterministic_algorithms=deterministic_algorithms,
        cudnn_deterministic=cudnn_deterministic,
        cudnn_benchmark=cudnn_benchmark,
    )

    _current_context = context
    return context


def get_context() -> Optional[DeterminismContext]:
    """Get current determinism context."""
    return _current_context


def get_context_summary() -> Dict[str, Any]:
    """Get summary of current determinism context."""
    if _current_context is None:
        return {"status": "not_set"}
    return _current_context.to_dict()


def require_context() -> DeterminismContext:
    """Get context, raising if not set."""
    if _current_context is None:
        raise RuntimeError(
            "Determinism context not set. Call set_determinism() first."
        )
    return _current_context


__all__ = [
    "DeterminismContext",
    "set_determinism",
    "get_context",
    "get_context_summary",
    "require_context",
]

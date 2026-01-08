"""Determinism helpers for demos and evaluations."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def configure_determinism(seed: Optional[int] = None, strict: bool = False) -> None:
    """Configure deterministic behavior for numpy/random/torch."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if strict:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def maybe_enable_determinism_from_env(default_seed: int = 0) -> Optional[int]:
    """Enable deterministic mode when VPE_DETERMINISTIC is set."""
    flag = os.getenv("VPE_DETERMINISTIC", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return None

    seed_env = os.getenv("VPE_DETERMINISTIC_SEED")
    seed = default_seed if seed_env is None else int(seed_env)
    strict_flag = os.getenv("VPE_DETERMINISTIC_STRICT", "").strip().lower()
    strict = strict_flag in {"1", "true", "yes", "on"}
    configure_determinism(seed=seed, strict=strict)
    return seed

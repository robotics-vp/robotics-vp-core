"""
Workcell environment configuration helpers.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.envs.workcell_env.config import PRESETS, WorkcellEnvConfig


def load_workcell_env_config(
    data: Optional[Dict[str, Any]] = None,
    preset: Optional[str] = None,
) -> WorkcellEnvConfig:
    """
    Load workcell environment config from dict and/or preset.

    Args:
        data: Optional dict with config values (overrides preset)
        preset: Optional preset name ("assembly_bench_simple", etc.)

    Returns:
        WorkcellEnvConfig instance
    """
    if preset and preset in PRESETS:
        base = PRESETS[preset]
        if not data:
            return base
        merged = base.to_dict()
        merged.update(data)
        return WorkcellEnvConfig.from_dict(merged)

    if data:
        return WorkcellEnvConfig.from_dict(data)

    return WorkcellEnvConfig()


__all__ = [
    "WorkcellEnvConfig",
    "load_workcell_env_config",
]

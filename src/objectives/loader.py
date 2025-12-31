from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from src.objectives.economic_objective import EconomicObjectiveSpec


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_objective_spec(path: str | Path) -> EconomicObjectiveSpec:
    if not path:
        return EconomicObjectiveSpec()
    payload = yaml.safe_load(Path(path).read_text())
    if payload is None:
        return EconomicObjectiveSpec()
    if not isinstance(payload, Mapping):
        raise ValueError(f"Objective config must be a mapping, got {type(payload).__name__}")
    return EconomicObjectiveSpec(
        mpl_weight=_safe_float(payload.get("mpl_weight", 1.0), 1.0),
        energy_weight=_safe_float(payload.get("energy_weight", 0.0), 0.0),
        error_weight=_safe_float(payload.get("error_weight", 0.0), 0.0),
        novelty_weight=_safe_float(payload.get("novelty_weight", 0.0), 0.0),
        risk_weight=_safe_float(payload.get("risk_weight", 0.0), 0.0),
        extra_weights=payload.get("extra_weights"),
    )

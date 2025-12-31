from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.config.objective_profile import ObjectiveVector


@dataclass(frozen=True)
class EconomicObjectiveSpec:
    """Economic objective weights for reward overlays and reporting."""

    mpl_weight: float = 1.0
    energy_weight: float = 0.0
    error_weight: float = 0.0
    novelty_weight: float = 0.0
    risk_weight: float = 0.0
    extra_weights: Mapping[str, float] | None = None

    @classmethod
    def from_objective_vector(
        cls,
        objective_vector: ObjectiveVector,
        extra_weights: Mapping[str, float] | None = None,
    ) -> "EconomicObjectiveSpec":
        return cls(
            mpl_weight=objective_vector.w_mpl,
            energy_weight=objective_vector.w_energy,
            error_weight=objective_vector.w_error,
            novelty_weight=objective_vector.w_novelty,
            risk_weight=objective_vector.w_safety,
            extra_weights=extra_weights,
        )


@dataclass(frozen=True)
class CompiledRewardOverlay:
    reward_scales: Mapping[str, float]


def compile_economic_overlay(obj: EconomicObjectiveSpec) -> CompiledRewardOverlay:
    scales: dict[str, float] = {}

    if obj.mpl_weight != 0.0:
        scales["mpl_per_hour"] = obj.mpl_weight
    if obj.energy_weight != 0.0:
        scales["energy_kwh"] = -abs(obj.energy_weight)
    if obj.error_weight != 0.0:
        scales["error_rate"] = -abs(obj.error_weight)
    if obj.novelty_weight != 0.0:
        scales["novelty_score"] = obj.novelty_weight
    if obj.risk_weight != 0.0:
        scales["risk_score"] = -abs(obj.risk_weight)

    if obj.extra_weights:
        for key, value in obj.extra_weights.items():
            scales[str(key)] = float(value)

    return CompiledRewardOverlay(reward_scales=scales)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_spec_from_mapping(payload: Mapping[str, Any]) -> EconomicObjectiveSpec:
    if "objective_vector" in payload:
        vector = payload.get("objective_vector") or []
        obj_vec = ObjectiveVector.from_list(list(vector))
        return EconomicObjectiveSpec.from_objective_vector(obj_vec, payload.get("extra_weights"))
    if "objective_preset" in payload:
        preset = str(payload.get("objective_preset"))
        obj_vec = ObjectiveVector.from_preset(preset)
        return EconomicObjectiveSpec.from_objective_vector(obj_vec, payload.get("extra_weights"))

    return EconomicObjectiveSpec(
        mpl_weight=_safe_float(payload.get("mpl_weight", payload.get("mpl", 1.0))),
        energy_weight=_safe_float(payload.get("energy_weight", payload.get("energy", 0.0))),
        error_weight=_safe_float(payload.get("error_weight", payload.get("error", 0.0))),
        novelty_weight=_safe_float(payload.get("novelty_weight", payload.get("novelty", 0.0))),
        risk_weight=_safe_float(payload.get("risk_weight", payload.get("risk", 0.0))),
        extra_weights=payload.get("extra_weights"),
    )


def load_economic_objective_spec(path: Path) -> EconomicObjectiveSpec:
    """Load an objective spec from YAML or JSON."""
    if not path or not path.exists():
        return EconomicObjectiveSpec()
    payload = yaml.safe_load(path.read_text())
    if payload is None:
        return EconomicObjectiveSpec()
    if not isinstance(payload, Mapping):
        raise ValueError(f"Objective config at {path} must be a mapping, got {type(payload).__name__}")
    return _build_spec_from_mapping(payload)

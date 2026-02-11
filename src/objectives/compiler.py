"""Objective tensor scalarization compiler."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.objectives.profile import ObjectiveProfile
from src.objectives.tensor import ObjectiveTensor


@dataclass
class ConstraintFlag:
    axis: str
    flag: str
    threshold: float
    observed: float

    def to_dict(self) -> Dict[str, float | str]:
        return {
            "axis": self.axis,
            "flag": self.flag,
            "threshold": float(self.threshold),
            "observed": float(self.observed),
        }


class ObjectiveCompiler:
    """Compile objective tensors into scalar rewards under an ObjectiveProfile."""

    def __init__(self, profile: ObjectiveProfile):
        self.profile = profile

    def _direction(self, axis: str) -> float:
        default_maximize = axis not in {"error", "energy", "damage_cost", "constraint_penalty"}
        return 1.0 if self.profile.maximize_axis(axis, default=default_maximize) else -1.0

    def _mean_vector(self, objective_tensor: ObjectiveTensor) -> Tuple[np.ndarray, List[str]]:
        axes = list(objective_tensor.schema.axes)
        vec = objective_tensor.mean_vector(normalize=True)
        return vec.astype(np.float64), axes

    def constraint_flags(self, objective_tensor: ObjectiveTensor) -> List[Dict[str, float | str]]:
        vec, axes = self._mean_vector(objective_tensor)
        axis_to_value = {axis: float(vec[i]) for i, axis in enumerate(axes)}
        flags: List[ConstraintFlag] = []

        for axis, spec in self.profile.constraints.items():
            if axis not in axis_to_value:
                continue
            observed = axis_to_value[axis]
            if "min" in spec and observed < float(spec["min"]):
                flags.append(
                    ConstraintFlag(
                        axis=axis,
                        flag="below_min",
                        threshold=float(spec["min"]),
                        observed=observed,
                    )
                )
            if "max" in spec and observed > float(spec["max"]):
                flags.append(
                    ConstraintFlag(
                        axis=axis,
                        flag="above_max",
                        threshold=float(spec["max"]),
                        observed=observed,
                    )
                )
        return [f.to_dict() for f in flags]

    def scalarize(self, objective_tensor: ObjectiveTensor) -> float:
        vec, axes = self._mean_vector(objective_tensor)
        axis_to_value = {axis: float(vec[i]) for i, axis in enumerate(axes)}
        directioned = {
            axis: self._direction(axis) * axis_to_value[axis]
            for axis in axes
        }

        mode = self.profile.scalarizer.lower().strip()
        if mode == "weighted_sum":
            return self._weighted_sum(directioned)
        if mode == "constrained":
            return self._constrained(directioned, objective_tensor)
        if mode == "lexicographic":
            return self._lexicographic(directioned, axes)
        if mode in {"chebyshev", "chebyshev_e"}:
            return self._chebyshev(directioned)
        if mode in {"epsilon", "epsilon_constraint"}:
            return self._epsilon(directioned, objective_tensor)
        if mode == "product":
            return self._product(directioned, axes)

        # Safe fallback
        return self._weighted_sum(directioned)

    def _weighted_sum(self, directioned: Dict[str, float]) -> float:
        if not directioned:
            return 0.0
        total = 0.0
        for axis, value in directioned.items():
            total += self.profile.weight_for(axis, default=1.0) * float(value)
        return float(total)

    def _constrained(self, directioned: Dict[str, float], objective_tensor: ObjectiveTensor) -> float:
        base = self._weighted_sum(directioned)
        flags = self.constraint_flags(objective_tensor)
        if not flags:
            return base
        penalty = 0.0
        for flag in flags:
            observed = float(flag["observed"])
            threshold = float(flag["threshold"])
            penalty += abs(observed - threshold)
        return float(base - self.profile.penalty_weight * penalty)

    def _lexicographic(self, directioned: Dict[str, float], axes: List[str]) -> float:
        order = self.profile.lexicographic_order or axes
        # Encode lexicographic tuple into scalar with decreasing magnitudes.
        score = 0.0
        scale = 1.0
        for axis in reversed(order):
            score += scale * float(directioned.get(axis, 0.0))
            scale *= 1e3
        return float(score)

    def _chebyshev(self, directioned: Dict[str, float]) -> float:
        if not directioned:
            return 0.0
        distances = []
        for axis, value in directioned.items():
            target = float(self.profile.chebyshev_target.get(axis, 1.0))
            weight = max(1e-6, self.profile.weight_for(axis, default=1.0))
            distances.append(weight * abs(target - float(value)))
        return float(-max(distances))

    def _epsilon(self, directioned: Dict[str, float], objective_tensor: ObjectiveTensor) -> float:
        for axis, min_value in self.profile.epsilon.items():
            observed = directioned.get(axis, None)
            if observed is None:
                continue
            if observed < float(min_value):
                # Hard reject while keeping finite scalar.
                return float(-1e6 + observed)
        # Then optimize primary weighted objective.
        return self._weighted_sum(directioned)

    def _product(self, directioned: Dict[str, float], axes: List[str]) -> float:
        if not axes:
            return 0.0
        prod = 1.0
        for axis in axes:
            value = max(1e-6, float(directioned.get(axis, 0.0)))
            prod *= value
        return float(prod)

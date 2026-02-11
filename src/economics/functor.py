"""Objective->Econ mapping layer (ObjectiveEconFunctor)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from src.economics.econ_tensor import EconTensor, EconTensorSchema
from src.objectives.tensor import ObjectiveTensor


@dataclass
class ObjectiveEconFunctor:
    """Map objective outcomes + constraints into econ deltas."""

    base_price_per_unit: float = 1.0
    uncertainty_discount_scale: float = 0.5
    constraint_penalty_scale: float = 1.0

    def map(
        self,
        objective_tensor: ObjectiveTensor,
        *,
        constraint_flags: Optional[Iterable[Mapping[str, Any]]] = None,
        uncertainty: float = 0.0,
        context: Optional[Mapping[str, Any]] = None,
    ) -> EconTensor:
        vec = objective_tensor.mean_vector(normalize=True)
        axis = {name: float(vec[i]) for i, name in enumerate(objective_tensor.schema.axes)}

        throughput = axis.get("throughput", 0.0)
        error = axis.get("error", 0.0)
        safety = axis.get("safety", 0.0)
        energy = axis.get("energy", 0.0)

        violations = list(constraint_flags or [])
        violation_count = float(len(violations))
        uncertainty = max(0.0, min(1.0, float(uncertainty)))

        marginal_frontier_gain = max(0.0, throughput + safety - error - 0.5 * energy)
        value_earned = self.base_price_per_unit * max(0.0, throughput - error)
        price_tick = self.base_price_per_unit * (1.0 + 0.5 * marginal_frontier_gain)
        constraint_penalty = self.constraint_penalty_scale * violation_count
        uncertainty_discount = self.uncertainty_discount_scale * uncertainty

        values = np.asarray(
            [
                value_earned,
                price_tick,
                marginal_frontier_gain,
                constraint_penalty,
                uncertainty_discount,
            ],
            dtype=np.float32,
        )
        econ_context = {
            "source": "objective_econ_functor",
            "objective_context": dict(objective_tensor.context),
            "violation_count": int(violation_count),
            "uncertainty": uncertainty,
        }
        if context:
            econ_context.update(dict(context))

        return EconTensor(
            values=values,
            schema=EconTensorSchema(),
            context=econ_context,
        )

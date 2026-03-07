"""Loss wiring for contract-aware critics and structured replay targets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from src.rl.contract_aware_critic import CriticOutput


@dataclass(frozen=True)
class ContractAwareLossWeights:
    scalar: float = 1.0
    objective: float = 0.5
    econ: float = 0.5
    consistency: float = 0.1
    confidence: float = 0.05


def _confidence_target(errors: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
    return torch.exp(-torch.clamp(errors.abs() / max(scale, 1e-6), min=0.0, max=10.0))


def contract_aware_losses(
    *,
    outputs: CriticOutput,
    scalar_targets: torch.Tensor,
    objective_targets: Optional[torch.Tensor] = None,
    econ_targets: Optional[torch.Tensor] = None,
    weights: ContractAwareLossWeights | None = None,
) -> Dict[str, torch.Tensor]:
    use_weights = weights or ContractAwareLossWeights()
    scalar_loss = F.mse_loss(outputs.compiled_scalar, scalar_targets)
    losses: Dict[str, torch.Tensor] = {"scalar_loss": scalar_loss}
    total = use_weights.scalar * scalar_loss

    if objective_targets is not None:
        objective_loss = F.mse_loss(outputs.objective_vector, objective_targets)
        objective_conf_target = _confidence_target(
            torch.mean(torch.abs(outputs.objective_vector.detach() - objective_targets.detach()), dim=-1),
            scale=1.0,
        )
        objective_conf_loss = F.mse_loss(outputs.objective_confidence, objective_conf_target)
        losses["objective_loss"] = objective_loss
        losses["objective_confidence_loss"] = objective_conf_loss
        total = total + use_weights.objective * objective_loss + use_weights.confidence * objective_conf_loss

    if econ_targets is not None:
        econ_loss = F.mse_loss(outputs.econ_vector, econ_targets)
        econ_conf_target = _confidence_target(
            torch.mean(torch.abs(outputs.econ_vector.detach() - econ_targets.detach()), dim=-1),
            scale=1.0,
        )
        econ_conf_loss = F.mse_loss(outputs.econ_confidence, econ_conf_target)
        losses["econ_loss"] = econ_loss
        losses["econ_confidence_loss"] = econ_conf_loss
        total = total + use_weights.econ * econ_loss + use_weights.confidence * econ_conf_loss

    consistency_loss = F.mse_loss(outputs.compiled_scalar, outputs.compiled_scalar_baseline.detach())
    scalar_conf_target = _confidence_target((outputs.compiled_scalar.detach() - scalar_targets.detach()).abs(), scale=1.0)
    scalar_conf_loss = F.mse_loss(outputs.scalar_confidence, scalar_conf_target)
    losses["consistency_loss"] = consistency_loss
    losses["scalar_confidence_loss"] = scalar_conf_loss
    total = total + use_weights.consistency * consistency_loss + use_weights.confidence * scalar_conf_loss
    losses["total_loss"] = total
    return losses

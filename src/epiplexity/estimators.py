"""Epiplexity estimators (prequential / requential)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProbeModelConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    lr: float = 1e-3


class EpiplexityEstimator(ABC):
    """Base interface for epiplexity estimators."""

    @abstractmethod
    def fit_and_score(
        self,
        tokens: torch.Tensor,
        steps: int,
        batch_size: int,
        seed: int,
    ) -> Tuple[float, float, List[float]]:
        """Fit a probe model and return (S_T_proxy, H_T_proxy, loss_curve)."""
        raise NotImplementedError


class _ProbeModel(nn.Module):
    def __init__(self, input_dim: int, config: ProbeModelConfig) -> None:
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(max(1, config.num_layers)):
            layers.append(nn.Linear(dim, config.hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            dim = config.hidden_dim
        layers.append(nn.Linear(dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrequentialAUCLossEstimator(EpiplexityEstimator):
    """Prequential estimator using area-under-loss-curve proxy."""

    def __init__(self, config: ProbeModelConfig | None = None) -> None:
        self.config = config or ProbeModelConfig()

    def fit_and_score(
        self,
        tokens: torch.Tensor,
        steps: int,
        batch_size: int,
        seed: int,
    ) -> Tuple[float, float, List[float]]:
        if tokens.dim() != 3:
            raise ValueError("tokens must be [N, T, D]")

        tokens = tokens.detach()
        torch.manual_seed(seed)
        N, T, D = tokens.shape
        if T < 2:
            return 0.0, 0.0, []

        device = tokens.device
        x = tokens[:, :-1, :].reshape(-1, D)
        y = tokens[:, 1:, :].reshape(-1, D)
        num_samples = x.shape[0]

        model = _ProbeModel(D, self.config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

        losses: List[float] = []
        rng = torch.Generator().manual_seed(seed)
        for _ in range(max(1, steps)):
            idx = torch.randint(0, num_samples, (min(batch_size, num_samples),), generator=rng)
            pred = model(x[idx])
            loss = F.mse_loss(pred, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        preq_code_length = float(sum(losses))
        initial_loss = float(losses[0]) if losses else 0.0
        s_t_proxy = max(0.0, initial_loss * len(losses) - preq_code_length)
        h_t_proxy = float(losses[-1]) if losses else 0.0
        return s_t_proxy, h_t_proxy, losses


class RequentialEstimator(EpiplexityEstimator):
    """Stub requential estimator (teacher-student) for future extension."""

    def fit_and_score(
        self,
        tokens: torch.Tensor,
        steps: int,
        batch_size: int,
        seed: int,
    ) -> Tuple[float, float, List[float]]:
        _ = tokens
        _ = steps
        _ = batch_size
        _ = seed
        return 0.0, 0.0, []


__all__ = [
    "ProbeModelConfig",
    "EpiplexityEstimator",
    "PrequentialAUCLossEstimator",
    "RequentialEstimator",
]

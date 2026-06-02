"""Epiplexity estimators (prequential / requential)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from abc import ABC, abstractmethod
import hashlib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProbeModelConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    lr: float = 1e-3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hidden_dim": int(self.hidden_dim),
            "num_layers": int(self.num_layers),
            "dropout": float(self.dropout),
            "lr": float(self.lr),
        }


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

    def estimator_id(self) -> str:
        return self.__class__.__name__

    def config_payload(self) -> Dict[str, Any]:
        return {}

    def config_sha(self) -> str:
        payload = json.dumps(self.config_payload(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def estimate_flops(
        self,
        tokens: torch.Tensor,
        steps: int,
        batch_size: int,
    ) -> float:
        _ = tokens
        _ = steps
        _ = batch_size
        return 0.0


class _ProbeModel(nn.Module):
    def __init__(self, input_dim: int, config: ProbeModelConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
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

    def config_payload(self) -> Dict[str, Any]:
        return {
            "estimator_id": self.estimator_id(),
            "probe_model": self.config.to_dict(),
        }

    def estimate_flops(
        self,
        tokens: torch.Tensor,
        steps: int,
        batch_size: int,
    ) -> float:
        if tokens.dim() != 3:
            return 0.0
        _, T, D = tokens.shape
        num_samples = max(0, int(tokens.shape[0]) * max(0, int(T) - 1))
        if num_samples <= 0:
            return 0.0
        effective_batch = max(1, min(int(batch_size), num_samples))
        return float(max(1, steps)) * float(effective_batch) * _probe_training_flops(D, self.config)

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
    """Online requential estimator using evaluate-then-update ordering."""

    def __init__(self, config: ProbeModelConfig | None = None) -> None:
        self.config = config or ProbeModelConfig()

    def config_payload(self) -> Dict[str, Any]:
        return {
            "estimator_id": self.estimator_id(),
            "probe_model": self.config.to_dict(),
        }

    def estimate_flops(
        self,
        tokens: torch.Tensor,
        steps: int,
        batch_size: int,
    ) -> float:
        if tokens.dim() != 3:
            return 0.0
        _, T, D = tokens.shape
        num_samples = max(0, int(tokens.shape[0]) * max(0, int(T) - 1))
        if num_samples <= 0:
            return 0.0
        effective_batch = max(1, min(int(batch_size), num_samples))
        return float(max(1, steps)) * float(effective_batch) * _probe_training_flops(D, self.config)

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
            batch_x = x[idx]
            batch_y = y[idx]

            with torch.no_grad():
                pred_before = model(batch_x)
                prequential_loss = F.mse_loss(pred_before, batch_y)
            losses.append(float(prequential_loss.detach().cpu().item()))

            pred_after = model(batch_x)
            train_loss = F.mse_loss(pred_after, batch_y)
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            optimizer.step()

        requential_code_length = float(sum(losses))
        initial_loss = float(losses[0]) if losses else 0.0
        s_t_proxy = max(0.0, initial_loss * len(losses) - requential_code_length)
        h_t_proxy = float(losses[-1]) if losses else 0.0
        return s_t_proxy, h_t_proxy, losses


def _probe_training_flops(input_dim: int, config: ProbeModelConfig) -> float:
    layer_dims: List[Tuple[int, int]] = []
    dim = int(input_dim)
    hidden_dim = int(config.hidden_dim)
    for _ in range(max(1, int(config.num_layers))):
        layer_dims.append((dim, hidden_dim))
        dim = hidden_dim
    layer_dims.append((dim, int(input_dim)))
    forward_flops = float(sum(2 * inp * out for inp, out in layer_dims))
    # Approximate training compute as forward + backward + optimizer update.
    return max(1.0, 3.0 * forward_flops)


__all__ = [
    "ProbeModelConfig",
    "EpiplexityEstimator",
    "PrequentialAUCLossEstimator",
    "RequentialEstimator",
]

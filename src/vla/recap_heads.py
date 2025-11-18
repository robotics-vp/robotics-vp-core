"""
RECAP-style head interfaces (minimal torch modules, deterministic-friendly).
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import bisect

import torch
from torch import nn


@dataclass
class AdvantageConditioningConfig:
    advantage_bins: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_bin(self, advantage: float) -> int:
        idx = bisect.bisect_right(self.advantage_bins, advantage)
        num_bins = len(self.advantage_bins) + 1
        return max(0, min(idx, num_bins - 1))


class AdvantageConditioningHead(nn.Module):
    """
    Tiny MLP that predicts a binned advantage class.

    The binning is deterministic: bins are sorted thresholds, with the final bin
    covering values above the largest threshold and the first bin covering
    values below the smallest threshold.
    """

    def __init__(self, config: AdvantageConditioningConfig, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.config = config
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, self.num_bins)

    @property
    def num_bins(self) -> int:
        return len(self.config.advantage_bins) + 1

    def compute_bin(self, advantage: float) -> int:
        """Return bin index for a scalar advantage, clamped into valid range."""
        return self.config.compute_bin(advantage)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.hidden(features))


@dataclass
class DistributionalValueConfig:
    metrics: List[str]
    num_atoms: int
    value_supports: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributionalValueHead(nn.Module):
    """
    Predicts per-metric discrete value distributions.
    """

    def __init__(self, config: DistributionalValueConfig, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.config = config
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_logits = nn.Linear(hidden_dim, len(self.config.metrics) * self.config.num_atoms)

    def init_output_structure(self) -> Dict[str, Any]:
        """Initialize empty distributional outputs."""
        return {metric: [0.0] * self.config.num_atoms for metric in self.config.metrics}

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Tensor of shape [batch, num_metrics * num_atoms] ordered by metrics in config.metrics.
        """
        return self.value_logits(self.hidden(features))

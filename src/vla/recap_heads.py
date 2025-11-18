"""
RECAP-style head interfaces (lightweight, dependency-free).
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List
import bisect


@dataclass
class AdvantageConditioningConfig:
    advantage_bins: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvantageConditioningHead:
    def __init__(self, config: AdvantageConditioningConfig):
        self.config = config

    def compute_bin(self, advantage: float) -> int:
        """Return bin index for a scalar advantage."""
        return bisect.bisect_right(self.config.advantage_bins, advantage) - 1


@dataclass
class DistributionalValueConfig:
    metrics: List[str]
    num_atoms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributionalValueHead:
    def __init__(self, config: DistributionalValueConfig):
        self.config = config

    def init_output_structure(self) -> Dict[str, Any]:
        """Initialize empty distributional outputs."""
        return {metric: [0.0] * self.config.num_atoms for metric in self.config.metrics}


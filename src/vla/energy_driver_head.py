"""
Semantic energy driver head for attribution (weakly supervised).

This is not a physics model; it explains *why* energy was spent in semantic terms
so datapacks can carry interpretable tags alongside physics energy metrics.
"""
import torch
import torch.nn as nn
from typing import List, Dict


ENERGY_DRIVERS = [
    "long_reach",
    "high_friction",
    "cautious_fragility",
    "replanning_occlusion",
    "high_speed_execution",
]


class EnergyDriverHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int = 128, n_drivers: int = len(ENERGY_DRIVERS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_drivers),
        )

    def forward(self, x):
        """Return sigmoid scores for each semantic energy driver."""
        return torch.sigmoid(self.net(x))


def heuristic_energy_tags(episode_summary: Dict, thresholds: Dict[str, float] | None = None) -> List[str]:
    """
    Weak heuristics mapping physics metrics to semantic driver tags.
    """
    thresholds = thresholds or {}
    tags = []
    energy = episode_summary.get("energy", {})
    limbed = energy.get("limb_energy_Wh", {}) or {}
    total = sum(limbed.values()) if limbed else energy.get("total_Wh", 0.0)

    shoulder_frac = limbed.get("shoulder", 0.0) / max(total, 1e-6) if total > 0 else 0.0
    wrist_frac = limbed.get("wrist", 0.0) / max(total, 1e-6) if total > 0 else 0.0

    if shoulder_frac > thresholds.get("long_reach_frac", 0.5):
        tags.append("long_reach")
    if wrist_frac > thresholds.get("high_friction_frac", 0.4):
        tags.append("high_friction")
    if energy.get("total_Wh", 0.0) > thresholds.get("cautious_energy", 0.5) and wrist_frac > 0.2:
        tags.append("cautious_fragility")

    return tags

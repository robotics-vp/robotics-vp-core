"""
Objective and Economic Context Profiles.

Defines declarative structures for:
- ObjectiveVector: Multi-objective weights (MPL vs error vs energy vs safety)
- EconContext: Economic context (wage, energy price, market, customer segment)

These are wired into DataPackMeta for tracking which objectives were used,
but do NOT affect Phase B reward shaping yet - just logging and schema.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


@dataclass
class ObjectiveVector:
    """
    Multi-objective weights for programmable economic objectives.

    These weights determine the relative importance of different metrics
    in the meta-objective J. Higher weight = more importance.

    This is the "north star" the system optimizes toward.
    """
    w_mpl: float = 1.0        # Weight on marginal product of labor
    w_error: float = 1.0      # Weight on error rate (penalty)
    w_energy: float = 1.0     # Weight on energy efficiency
    w_safety: float = 1.0     # Weight on safety / fragility avoidance
    w_novelty: float = 0.0    # Weight on novelty / exploration (optional)

    def to_list(self) -> List[float]:
        """Convert to list for tensor operations."""
        return [self.w_mpl, self.w_error, self.w_energy, self.w_safety, self.w_novelty]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_list(cls, weights: List[float]) -> "ObjectiveVector":
        """Create from list of weights."""
        if len(weights) >= 5:
            return cls(
                w_mpl=weights[0],
                w_error=weights[1],
                w_energy=weights[2],
                w_safety=weights[3],
                w_novelty=weights[4],
            )
        elif len(weights) >= 4:
            return cls(
                w_mpl=weights[0],
                w_error=weights[1],
                w_energy=weights[2],
                w_safety=weights[3],
            )
        else:
            raise ValueError(f"Need at least 4 weights, got {len(weights)}")

    @classmethod
    def from_preset(cls, preset: str) -> "ObjectiveVector":
        """
        Create from preset name.

        Presets:
        - throughput: Focus on MPL, less on energy
        - safety: Prioritize safety and error reduction
        - energy_saver: Focus on energy efficiency
        - balanced: Equal weights
        - throughput_focused: Customer wants max throughput
        - premium_safety: Customer prioritizes zero damage
        """
        presets = {
            "throughput": cls(w_mpl=2.0, w_error=1.0, w_energy=0.5, w_safety=1.0),
            "safety": cls(w_mpl=0.5, w_error=2.0, w_energy=1.0, w_safety=3.0),
            "energy_saver": cls(w_mpl=1.0, w_error=1.0, w_energy=3.0, w_safety=1.0),
            "balanced": cls(w_mpl=1.0, w_error=1.0, w_energy=1.0, w_safety=1.0),
            "throughput_focused": cls(w_mpl=3.0, w_error=0.5, w_energy=0.3, w_safety=0.5),
            "premium_safety": cls(w_mpl=1.0, w_error=3.0, w_energy=1.0, w_safety=5.0),
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        return presets[preset]

    def normalize(self) -> "ObjectiveVector":
        """Return normalized copy (weights sum to 1)."""
        total = self.w_mpl + self.w_error + self.w_energy + self.w_safety + self.w_novelty
        if total == 0:
            total = 1.0
        return ObjectiveVector(
            w_mpl=self.w_mpl / total,
            w_error=self.w_error / total,
            w_energy=self.w_energy / total,
            w_safety=self.w_safety / total,
            w_novelty=self.w_novelty / total,
        )


@dataclass
class EconContext:
    """
    Economic context for adaptive econ parameters.

    Captures market conditions, customer segment, and human baselines.
    This context influences how econ parameters should be adjusted.
    """
    # Human baselines
    wage_human: float = 18.0                  # $/hour (national average)
    energy_price_kWh: float = 0.12            # $/kWh

    # Market segmentation
    market_region: str = "US"                 # "US", "EU", "APAC", etc.
    task_family: str = "dishwashing"          # "dishwashing", "drawer_vase", "bricklaying"
    customer_segment: str = "balanced"        # "premium_safety", "throughput_focused", "energy_saver"

    # Human performance baselines
    baseline_mpl_human: float = 60.0          # units/hour
    baseline_error_human: float = 0.05        # error rate (fraction)

    # Optional: time-varying context
    time_of_day: Optional[str] = None         # "peak", "off_peak"
    season: Optional[str] = None              # "summer", "winter" (for energy pricing)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EconContext":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_profile(cls, profile: dict) -> "EconContext":
        """
        Create from internal experiment profile dict.

        Falls back to defaults for missing fields.
        """
        return cls(
            wage_human=profile.get("wage_human", 18.0),
            energy_price_kWh=profile.get("energy_price_kWh", 0.12),
            market_region=profile.get("market_region", "US"),
            task_family=profile.get("task_family", "dishwashing"),
            customer_segment=profile.get("customer_segment", "balanced"),
            baseline_mpl_human=profile.get("baseline_mpl_human", profile.get("mpl_human", 60.0)),
            baseline_error_human=profile.get("baseline_error_human", 0.05),
            time_of_day=profile.get("time_of_day"),
            season=profile.get("season"),
        )


@dataclass
class RewardWeights:
    """
    Learned reward weights output by EconObjectiveNet.

    These are the actual mixing coefficients used in compute_econ_reward.
    Produced by a DL model from ObjectiveVector + EconContext.
    """
    alpha_mpl: float = 1.0        # Weight on ΔMPL term
    alpha_ep: float = 1.0         # Weight on ΔEP (energy productivity) term
    alpha_error: float = 1.0      # Weight on error rate term
    alpha_energy: float = 1.0     # Weight on energy consumption term
    alpha_safety: float = 1.0     # Weight on safety violation term

    def to_list(self) -> List[float]:
        """Convert to list."""
        return [self.alpha_mpl, self.alpha_ep, self.alpha_error, self.alpha_energy, self.alpha_safety]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_list(cls, alphas: List[float]) -> "RewardWeights":
        """Create from list."""
        if len(alphas) < 5:
            raise ValueError(f"Need 5 alphas, got {len(alphas)}")
        return cls(
            alpha_mpl=alphas[0],
            alpha_ep=alphas[1],
            alpha_error=alphas[2],
            alpha_energy=alphas[3],
            alpha_safety=alphas[4],
        )

    def normalize(self) -> "RewardWeights":
        """Return normalized copy (weights sum to 1)."""
        total = sum(self.to_list())
        if total == 0:
            total = 1.0
        return RewardWeights(
            alpha_mpl=self.alpha_mpl / total,
            alpha_ep=self.alpha_ep / total,
            alpha_error=self.alpha_error / total,
            alpha_energy=self.alpha_energy / total,
            alpha_safety=self.alpha_safety / total,
        )


# Default preset mapping for advisory scripts / analysis tooling
OBJECTIVE_PRESETS: Dict[str, ObjectiveVector] = {
    "throughput": ObjectiveVector.from_preset("throughput"),
    "energy_saver": ObjectiveVector.from_preset("energy_saver"),
    "balanced": ObjectiveVector.from_preset("balanced"),
    "safety_first": ObjectiveVector.from_preset("safety"),
    "throughput_focused": ObjectiveVector.from_preset("throughput_focused"),
    "premium_safety": ObjectiveVector.from_preset("premium_safety"),
}


def get_objective_presets() -> Dict[str, ObjectiveVector]:
    """Return a copy of named objective presets for analysis/reporting."""
    return OBJECTIVE_PRESETS.copy()

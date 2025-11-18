"""
Economic Domain Adapter (Phase G).

Handles calibration of economic vectors between simulation (PyBullet, Isaac)
and real-world labor economics. Acts as the "Exchange Rate" mechanism.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.ontology.models import EconVector


@dataclass
class EconDomainAdapterConfig:
    source_domain: str  # e.g. "pybullet"
    target_domain: str = "platform_econ"
    scaling: Dict[str, float] = field(default_factory=dict)
    offsets: Dict[str, float] = field(default_factory=dict)
    version: str = "v0.1_identity"


class EconDomainAdapter:
    """
    Deterministically maps raw EconVectors to calibrated platform economics.
    """

    def __init__(self, config: Optional[EconDomainAdapterConfig] = None):
        self.config = config or EconDomainAdapterConfig(source_domain="pybullet")

    def map_vector(self, econ: EconVector) -> EconVector:
        """
        Apply scaling and offsets to calibrate the vector.
        Returns a NEW EconVector instance (does not mutate input).
        """
        # Create a copy
        calibrated = EconVector(
            episode_id=econ.episode_id,
            mpl_units_per_hour=self._calibrate(econ.mpl_units_per_hour, "mpl_units_per_hour"),
            wage_parity=self._calibrate(econ.wage_parity, "wage_parity"),
            energy_cost=self._calibrate(econ.energy_cost, "energy_cost"),
            damage_cost=self._calibrate(econ.damage_cost, "damage_cost"),
            novelty_delta=self._calibrate(econ.novelty_delta, "novelty_delta"),
            reward_scalar_sum=econ.reward_scalar_sum, # Reward scalar is usually preserved
            mobility_penalty=self._calibrate(econ.mobility_penalty, "mobility_penalty"),
            precision_bonus=self._calibrate(econ.precision_bonus, "precision_bonus"),
            stability_risk_score=self._calibrate(econ.stability_risk_score, "stability_risk_score"),
            components=econ.components.copy(), # Deep copy if needed, but dict copy is fine for now
            metadata=econ.metadata.copy(),
            source_domain=self.config.source_domain,
            calibration_version=self.config.version,
        )
        
        # Mark as calibrated in metadata
        calibrated.metadata["is_calibrated"] = True
        calibrated.metadata["raw_source_domain"] = econ.source_domain
        
        return calibrated

    def _calibrate(self, value: float, component: str) -> float:
        scale = self.config.scaling.get(component, 1.0)
        offset = self.config.offsets.get(component, 0.0)
        return value * scale + offset

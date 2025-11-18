"""
Economic Domain Adapter (Phase G).

Handles calibration of economic vectors between simulation (PyBullet, Isaac)
and real-world labor economics. Acts as the "Exchange Rate" mechanism.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import yaml

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

    def __init__(self, config: Optional[EconDomainAdapterConfig] = None, domain_name: str = "default", config_path: Optional[str] = None):
        self.domain_name = domain_name
        self.config = config or self._load_config_from_yaml(domain_name, config_path)

    def _load_config_from_yaml(self, domain_name: str, config_path: Optional[str]) -> EconDomainAdapterConfig:
        cfg_path = Path(config_path) if config_path else Path(__file__).resolve().parents[2] / "config" / "econ_domains.yaml"
        if not cfg_path.exists():
            return EconDomainAdapterConfig(source_domain=domain_name or "pybullet")
        try:
            with cfg_path.open("r") as f:
                payload = yaml.safe_load(f) or {}
        except Exception:
            payload = {}

        profile = payload.get(domain_name) or payload.get("default") or {}

        def _safe_float(val, default: float = 1.0) -> float:
            try:
                return float(val)
            except Exception:
                return default
        scaling = {}
        if isinstance(profile, dict):
            scaling["mpl_units_per_hour"] = _safe_float(profile.get("mpl_scale"), 1.0)
            scaling["energy_cost"] = _safe_float(profile.get("energy_scale"), 1.0)
            scaling["damage_cost"] = _safe_float(profile.get("damage_scale"), 1.0)
            offsets = profile.get("offset", {}) if isinstance(profile.get("offset"), dict) else {}
        else:
            offsets = {}
        return EconDomainAdapterConfig(
            source_domain=profile.get("source_domain", domain_name or "pybullet") if isinstance(profile, dict) else (domain_name or "pybullet"),
            target_domain=profile.get("target_domain", "platform_econ") if isinstance(profile, dict) else "platform_econ",
            scaling=scaling,
            offsets=offsets,
            version=f"profile:{domain_name}",
        )

    def map_vector(self, econ: EconVector) -> EconVector:
        """
        Apply scaling and offsets to calibrate the vector.
        Returns a NEW EconVector instance (does not mutate input).
        """
        raw_components = (econ.components or {}).copy()
        calibrated_components = self._calibrate_components(raw_components)

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
            components=raw_components,
            metadata=econ.metadata.copy(),
            source_domain=self.config.source_domain,
            calibration_version=self.config.version,
        )
        
        # Mark as calibrated in metadata
        calibrated.metadata["is_calibrated"] = True
        calibrated.metadata["raw_source_domain"] = econ.source_domain
        calibrated.metadata["econ_domain_name"] = self.domain_name
        calibrated.metadata["calibration_target_domain"] = self.config.target_domain
        calibrated.metadata["raw_components"] = raw_components
        calibrated.metadata["calibrated_components"] = calibrated_components
        calibrated.metadata["raw_econ_snapshot"] = {
            "mpl_units_per_hour": econ.mpl_units_per_hour,
            "wage_parity": econ.wage_parity,
            "energy_cost": econ.energy_cost,
            "damage_cost": econ.damage_cost,
            "novelty_delta": econ.novelty_delta,
            "reward_scalar_sum": econ.reward_scalar_sum,
            "mobility_penalty": econ.mobility_penalty,
            "precision_bonus": econ.precision_bonus,
            "stability_risk_score": econ.stability_risk_score,
        }
        
        return calibrated

    def _calibrate(self, value: float, component: str) -> float:
        scale = self.config.scaling.get(component, 1.0)
        offset = self.config.offsets.get(component, 0.0)
        return value * scale + offset

    def _calibrate_components(self, components: Dict[str, float]) -> Dict[str, float]:
        calibrated = {}
        for k, v in components.items():
            calibrated[k] = self._calibrate(v, k)
        return calibrated

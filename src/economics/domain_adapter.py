"""
Economic Domain Adapter (Phase G).

Handles calibration of economic vectors between simulation (PyBullet, Isaac)
and real-world labor economics. Acts as the "Exchange Rate" mechanism.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from src.ontology.models import EconVector


@dataclass
class EconDomainAdapterConfig:
    source_domain: str  # e.g. "pybullet"
    target_domain: str = "platform_econ"
    scaling: Dict[str, float] = field(default_factory=dict)
    offsets: Dict[str, float] = field(default_factory=dict)
    version: str = "v0.1_identity"


logger = logging.getLogger(__name__)


class EconDomainAdapter:
    """
    Deterministically maps raw EconVectors to calibrated platform economics.
    """

    def __init__(
        self,
        config: Optional[EconDomainAdapterConfig] = None,
        domain_name: str = "default",
        config_path: Optional[str] = None,
    ):
        self.domain_name = domain_name
        self.config_path = config_path
        self.config = config or self._load_config_from_yaml(domain_name, config_path)

    def _load_config_from_yaml(self, domain_name: str, config_path: Optional[str]) -> EconDomainAdapterConfig:
        cfg_path = Path(config_path) if config_path else Path(__file__).resolve().parents[2] / "config" / "econ_domains.yaml"
        if not cfg_path.exists():
            logger.warning("EconDomainAdapter config not found at %s; using identity calibration", cfg_path)
            return EconDomainAdapterConfig(source_domain=domain_name or "pybullet")

        try:
            with cfg_path.open("r") as f:
                payload = yaml.safe_load(f) or {}
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to load econ domain config from %s: %s", cfg_path, exc)
            payload = {}

        profile = payload.get(domain_name)
        if profile is None:
            logger.warning("EconDomainAdapter: domain '%s' not found in %s; falling back to identity/default profile", domain_name, cfg_path)
            profile = payload.get("default", {}) or {}

        if not isinstance(profile, dict):
            profile = {}

        scaling, offsets = self._extract_calibration(profile)
        return EconDomainAdapterConfig(
            source_domain=profile.get("source_domain", domain_name or "pybullet"),
            target_domain=profile.get("target_domain", "platform_econ"),
            scaling=scaling,
            offsets=offsets,
            version=str(profile.get("version") or f"profile:{domain_name or 'default'}"),
        )

    def _extract_calibration(self, profile: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Normalize YAML profile into scale/offset maps.

        Supports new keys (scale_mpl, bias_energy_wh, etc.) and legacy names
        (mpl_scale, energy_scale, damage_scale) for backward compatibility.
        """
        scaling: Dict[str, float] = {}
        offsets: Dict[str, float] = {}

        def _first(keys, default=None):
            for key in keys:
                if key in profile:
                    return profile[key]
            return default

        def _safe_float(val, default: float) -> float:
            try:
                return float(val)
            except Exception:
                return default

        component_key_map = {
            "mpl_units_per_hour": (["scale_mpl", "mpl_scale", "scale_mpl_units_per_hour"], ["bias_mpl", "mpl_bias"]),
            "energy_cost": (["scale_energy_wh", "energy_scale", "scale_energy_cost"], ["bias_energy_wh", "energy_bias"]),
            "damage_cost": (["scale_damage", "damage_scale"], ["bias_damage", "damage_bias"]),
            "wage_parity": (["scale_wage_parity"], ["bias_wage_parity"]),
            "novelty_delta": (["scale_novelty_delta"], ["bias_novelty_delta"]),
            "mobility_penalty": (["scale_mobility_penalty"], ["bias_mobility_penalty"]),
            "precision_bonus": (["scale_precision_bonus"], ["bias_precision_bonus"]),
            "stability_risk_score": (["scale_stability_risk_score"], ["bias_stability_risk_score"]),
            "reward_scalar_sum": (["scale_reward_scalar_sum"], ["bias_reward_scalar_sum"]),
        }

        for component, (scale_keys, bias_keys) in component_key_map.items():
            scaling_val = _safe_float(_first(scale_keys, 1.0), 1.0)
            offset_val = _safe_float(_first(bias_keys, 0.0), 0.0)
            if scaling_val != 1.0:
                scaling[component] = scaling_val
            if offset_val != 0.0:
                offsets[component] = offset_val

        # Capture any additional explicit scale_*/bias_* entries for components/components dict
        for key, val in profile.items():
            if key.startswith("scale_"):
                comp = key[len("scale_") :]
                scaling.setdefault(comp, _safe_float(val, 1.0))
            elif key.startswith("bias_"):
                comp = key[len("bias_") :]
                offsets.setdefault(comp, _safe_float(val, 0.0))

        # Legacy offset dict (offset.energy_cost style)
        if isinstance(profile.get("offset"), dict):
            for comp, val in profile["offset"].items():
                offsets.setdefault(comp, _safe_float(val, 0.0))

        return scaling, offsets

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
            reward_scalar_sum=self._calibrate(econ.reward_scalar_sum, "reward_scalar_sum"),
            mobility_penalty=self._calibrate(econ.mobility_penalty, "mobility_penalty"),
            precision_bonus=self._calibrate(econ.precision_bonus, "precision_bonus"),
            stability_risk_score=self._calibrate(econ.stability_risk_score, "stability_risk_score"),
            components=raw_components,
            metadata=(econ.metadata or {}).copy(),
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
        calibrated.metadata["calibrated_econ_snapshot"] = {
            "mpl_units_per_hour": calibrated.mpl_units_per_hour,
            "wage_parity": calibrated.wage_parity,
            "energy_cost": calibrated.energy_cost,
            "damage_cost": calibrated.damage_cost,
            "novelty_delta": calibrated.novelty_delta,
            "reward_scalar_sum": calibrated.reward_scalar_sum,
            "mobility_penalty": calibrated.mobility_penalty,
            "precision_bonus": calibrated.precision_bonus,
            "stability_risk_score": calibrated.stability_risk_score,
        }
        calibrated.metadata["calibration_profile"] = {
            "scaling": dict(self.config.scaling),
            "offsets": dict(self.config.offsets),
            "version": self.config.version,
            "domain": self.domain_name,
            "config_path": str(self.config_path or Path(__file__).resolve().parents[2] / "config" / "econ_domains.yaml"),
        }
        
        return calibrated

    def _calibrate(self, value: float, component: str) -> float:
        scale, offset = self._resolve_calibration(component)
        try:
            return float(value) * scale + offset
        except Exception:
            return 0.0

    def _calibrate_components(self, components: Dict[str, float]) -> Dict[str, float]:
        calibrated = {}
        for k, v in components.items():
            calibrated[k] = self._calibrate(v, k)
        return calibrated

    def _resolve_calibration(self, component: str) -> Tuple[float, float]:
        """
        Resolve scale/offset with sensible aliases to keep domains simple.
        """
        scale = self.config.scaling.get(component)
        offset = self.config.offsets.get(component)

        # Alias common econ component names to base fields
        if scale is None:
            if "energy" in component:
                scale = self.config.scaling.get("energy_cost")
            elif "damage" in component or "collision" in component:
                scale = self.config.scaling.get("damage_cost")
            elif "mpl" in component:
                scale = self.config.scaling.get("mpl_units_per_hour")
        if offset is None:
            if "energy" in component:
                offset = self.config.offsets.get("energy_cost")
            elif "damage" in component or "collision" in component:
                offset = self.config.offsets.get("damage_cost")
            elif "mpl" in component:
                offset = self.config.offsets.get("mpl_units_per_hour")

        return float(scale if scale is not None else 1.0), float(offset if offset is not None else 0.0)

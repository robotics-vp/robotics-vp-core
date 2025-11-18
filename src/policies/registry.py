"""
Policy registry for Phase G policy abstractions.

Loads config/policies.yaml and instantiates the heuristic implementations.
Default behavior remains unchanged; "neural"/"stub" values are ignored for
now and routed to the heuristic versions to preserve current outputs.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.policies.data_valuation import HeuristicDataValuationPolicy
from src.policies.pricing import HeuristicPricingPolicy
from src.policies.safety_risk import HeuristicSafetyRiskPolicy
from src.policies.energy_cost import HeuristicEnergyCostPolicy
from src.policies.episode_quality import HeuristicEpisodeQualityPolicy
from src.policies.orchestrator_policy import HeuristicOrchestratorPolicy
from src.policies.sampler_weights import HeuristicSamplerWeightPolicy
from src.policies.meta_advisor import HeuristicMetaAdvisorPolicy
from src.policies.vision_encoder import HeuristicVisionEncoderPolicy
from src.policies.vision_encoder import HeuristicVisionEncoderPolicy
from src.policies.reward_model_heuristic import HeuristicRewardModelPolicy
from src.policies.datapack_auditor import HeuristicDatapackAuditor


DEFAULT_POLICY_CONFIG: Dict[str, str] = {
    "data_valuation": "heuristic",
    "pricing": "heuristic",
    "safety_risk": "heuristic",
    "energy_cost": "heuristic",
    "episode_quality": "heuristic",
    "sampler_weights": "heuristic",
    "orchestrator": "heuristic",
    "meta_advisor": "heuristic",
    "vision_encoder": "heuristic",
    "reward_model": "heuristic",
    "datapack_auditor": "heuristic",
}


@dataclass
class PolicyBundle:
    data_valuation: Any
    pricing: Any
    safety_risk: Any
    energy_cost: Any
    episode_quality: Any
    sampler_weights: Any
    orchestrator: Any
    meta_advisor: Any
    vision_encoder: Any
    reward_model: Any
    datapack_auditor: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_config(path: Optional[str] = None) -> Dict[str, str]:
    cfg_path = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "policies.yaml"
    if not cfg_path.exists():
        return dict(DEFAULT_POLICY_CONFIG)
    try:
        with cfg_path.open("r") as f:
            loaded = yaml.safe_load(f) or {}
        merged = dict(DEFAULT_POLICY_CONFIG)
        merged.update({k: str(v) for k, v in loaded.items()})
        return merged
    except Exception:
        return dict(DEFAULT_POLICY_CONFIG)


def _select_policy(kind: str, heuristic_cls):
    """
    Select the policy implementation. For Phase G initial drop, all values
    route to heuristic implementations to preserve behavior.
    """
    _ = kind  # Reserved for future neural/stub routing
    return heuristic_cls()


def build_all_policies(config_path: Optional[str] = None) -> PolicyBundle:
    """
    Instantiate all policies according to config/policies.yaml.
    """
    cfg = _load_config(config_path)
    return PolicyBundle(
        data_valuation=_select_policy(cfg.get("data_valuation", "heuristic"), HeuristicDataValuationPolicy),
        pricing=_select_policy(cfg.get("pricing", "heuristic"), HeuristicPricingPolicy),
        safety_risk=_select_policy(cfg.get("safety_risk", "heuristic"), HeuristicSafetyRiskPolicy),
        energy_cost=_select_policy(cfg.get("energy_cost", "heuristic"), HeuristicEnergyCostPolicy),
        episode_quality=_select_policy(cfg.get("episode_quality", "heuristic"), HeuristicEpisodeQualityPolicy),
        sampler_weights=_select_policy(cfg.get("sampler_weights", "heuristic"), HeuristicSamplerWeightPolicy),
        orchestrator=_select_policy(cfg.get("orchestrator", "heuristic"), HeuristicOrchestratorPolicy),
        meta_advisor=_select_policy(cfg.get("meta_advisor", "heuristic"), HeuristicMetaAdvisorPolicy),
        vision_encoder=_select_policy(cfg.get("vision_encoder", "heuristic"), HeuristicVisionEncoderPolicy),
        reward_model=_select_policy(cfg.get("reward_model", "heuristic"), HeuristicRewardModelPolicy),
        datapack_auditor=_select_policy(cfg.get("datapack_auditor", "heuristic"), HeuristicDatapackAuditor),
    )

"""
Policy registry for Phase G policy abstractions.

Loads config/policies.yaml and instantiates the heuristic implementations.
Default behavior remains unchanged; "neural"/"stub" values are ignored for
now and routed to the heuristic versions to preserve current outputs.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.analytics.econ_correlator import load_trust_matrix
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


@dataclass
class RegisteredPolicy:
    policy_id: str
    type: str  # "monolithic" | "hydra"
    trunk: Any = None
    heads: Optional[Dict[str, Any]] = None
    model: Any = None
    backend_support: List[str] = field(default_factory=list)
    default_skill_mode: Optional[str] = None


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


def _select_policy(kind: str, heuristic_cls, *args, **kwargs):
    """
    Select the policy implementation. For Phase G initial drop, all values
    route to heuristic implementations to preserve behavior.
    """
    _ = kind  # Reserved for future neural/stub routing
    return heuristic_cls(*args, **kwargs)


def build_all_policies(config_path: Optional[str] = None) -> PolicyBundle:
    """
    Instantiate all policies according to config/policies.yaml.
    """
    cfg = _load_config(config_path)
    trust_matrix = load_trust_matrix()
    return PolicyBundle(
        data_valuation=_select_policy(cfg.get("data_valuation", "heuristic"), HeuristicDataValuationPolicy, trust_matrix),
        pricing=_select_policy(cfg.get("pricing", "heuristic"), HeuristicPricingPolicy),
        safety_risk=_select_policy(cfg.get("safety_risk", "heuristic"), HeuristicSafetyRiskPolicy, trust_matrix),
        energy_cost=_select_policy(cfg.get("energy_cost", "heuristic"), HeuristicEnergyCostPolicy),
        episode_quality=_select_policy(cfg.get("episode_quality", "heuristic"), HeuristicEpisodeQualityPolicy),
        sampler_weights=_select_policy(cfg.get("sampler_weights", "heuristic"), HeuristicSamplerWeightPolicy, trust_matrix),
        orchestrator=_select_policy(cfg.get("orchestrator", "heuristic"), HeuristicOrchestratorPolicy),
        meta_advisor=_select_policy(cfg.get("meta_advisor", "heuristic"), HeuristicMetaAdvisorPolicy),
        vision_encoder=_select_policy(cfg.get("vision_encoder", "heuristic"), HeuristicVisionEncoderPolicy),
        reward_model=_select_policy(cfg.get("reward_model", "heuristic"), HeuristicRewardModelPolicy),
        datapack_auditor=_select_policy(cfg.get("datapack_auditor", "heuristic"), HeuristicDatapackAuditor),
    )


class PolicyRegistry:
    """
    Lightweight registry for Hydra/monolithic policies (RL/VLA-facing).
    Backwards-compatible: existing heuristic bundle remains unchanged.
    """

    def __init__(self, register_test_policy: bool = True) -> None:
        self.policies: Dict[str, RegisteredPolicy] = {}
        if register_test_policy:
            try:
                self._register_hydra_test_policy()
            except Exception:
                # Best-effort registration; failures should not block callers
                pass

    def register_hydra_policy(
        self,
        policy_id: str,
        trunk: Any,
        heads: Dict[str, Any],
        backend_support: Optional[List[str]] = None,
        default_skill_mode: Optional[str] = None,
    ) -> None:
        self.policies[policy_id] = RegisteredPolicy(
            policy_id=policy_id,
            type="hydra",
            trunk=trunk,
            heads=heads,
            backend_support=backend_support or [],
            default_skill_mode=default_skill_mode,
        )

    def register_monolithic_policy(
        self,
        policy_id: str,
        model: Any,
        backend_support: Optional[List[str]] = None,
    ) -> None:
        self.policies[policy_id] = RegisteredPolicy(
            policy_id=policy_id,
            type="monolithic",
            model=model,
            backend_support=backend_support or [],
        )

    def get_policy(self, policy_id: str, skill_mode: Optional[str] = None) -> Any:
        if policy_id not in self.policies:
            raise KeyError(f"Policy {policy_id} not registered")
        entry = self.policies[policy_id]
        if entry.type == "hydra":
            if skill_mode is None:
                raise ValueError(f"Hydra policy {policy_id} requires skill_mode")
            from src.rl.hydra_heads import HydraPolicy  # Local import to avoid circulars

            head = (entry.heads or {}).get(skill_mode)
            if head is None:
                head = (entry.heads or {}).get(entry.default_skill_mode)
            if head is None:
                raise KeyError(f"No head found for skill_mode={skill_mode}")
            return HydraPolicy(entry.trunk, head)
        return entry.model

    def list_policies(self, backend_id: Optional[str] = None) -> List[str]:
        if backend_id is None:
            return list(self.policies.keys())
        return [
            pid
            for pid, entry in self.policies.items()
            if not entry.backend_support or backend_id in entry.backend_support
        ]

    def _register_hydra_test_policy(self) -> None:
        """
        Minimal deterministic Hydra policy used for smoke tests.
        """
        import torch.nn as nn
        from src.rl.trunk_net import TrunkNet

        cfg = _load_config()
        hidden_dim = 16
        condition_dim = 16
        try:
            condition_hidden = int(cfg.get("condition_film_hidden_dim", hidden_dim))
        except Exception:
            condition_hidden = hidden_dim
        try:
            condition_context_dim = int(cfg.get("condition_context_dim", hidden_dim))
        except Exception:
            condition_context_dim = hidden_dim
        use_policy_condition = str(cfg.get("use_condition_vector_for_policy", "false")).lower() in ("1", "true", "yes")
        condition_fusion_mode = str(cfg.get("condition_fusion_mode", "film"))
        trunk = TrunkNet(
            vision_dim=4,
            state_dim=4,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            use_condition_film=False,
            use_condition_vector=True,
            use_condition_vector_for_policy=use_policy_condition,
            condition_fusion_mode=condition_fusion_mode,
            condition_film_hidden_dim=condition_hidden,
            condition_context_dim=condition_context_dim,
        )

        class _StubHead(nn.Module):
            def __init__(self, head_bias: float) -> None:
                super().__init__()
                self.linear = nn.Linear(hidden_dim, 2)
                nn.init.constant_(self.linear.weight, head_bias)
                nn.init.constant_(self.linear.bias, head_bias)

            def forward(self, trunk_features, condition=None):
                return self.linear(trunk_features)

        heads = {
            "default": _StubHead(0.0),
            "frontier_exploration": _StubHead(0.1),
            "recovery_heavy": _StubHead(-0.1),
        }
        self.register_hydra_policy(
            policy_id="hydra_test_policy",
            trunk=trunk,
            heads=heads,
            backend_support=["any"],
            default_skill_mode="default",
        )

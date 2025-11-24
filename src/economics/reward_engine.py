"""
RewardEngine: advisory wrapper to decompose rewards and compute EconVectors.

Does NOT alter scalar rewards used by SAC/PPO; it only mirrors existing reward
math into logged components and episode-level EconVector aggregation.
"""
from dataclasses import asdict
from typing import Any, Dict, List, Tuple
from datetime import datetime

from src.ontology.models import Task, Robot, Episode, EpisodeEvent, EconVector
from src.policies.registry import build_all_policies
from src.economics.domain_adapter import EconDomainAdapter, EconDomainAdapterConfig


class RewardEngine:
    def __init__(
        self,
        task: Task,
        robot: Robot,
        config: Dict[str, Any],
        policies=None,
        econ_domain_name: str = "default",
    ):
        self.task = task
        self.robot = robot
        self.config = config or {}
        self.policies = policies or build_all_policies()

        # Initialize domain adapter from YAML profile with optional overrides
        domain_name = self.config.get("econ_domain_name", econ_domain_name or "default")
        self.adapter = EconDomainAdapter(
            domain_name=domain_name,
            config_path=self.config.get("econ_domain_config_path"),
        )
        # Preserve legacy inline overrides without changing behavior
        if isinstance(self.adapter.config.scaling, dict):
            self.adapter.config.scaling.update(self.config.get("econ_scaling", {}) or {})
        else:
            self.adapter.config.scaling = self.config.get("econ_scaling", {}) or {}
        if isinstance(self.adapter.config.offsets, dict):
            self.adapter.config.offsets.update(self.config.get("econ_offsets", {}) or {})
        else:
            self.adapter.config.offsets = self.config.get("econ_offsets", {}) or {}
        if self.config.get("source_domain"):
            self.adapter.config.source_domain = self.config["source_domain"]

    def step_reward(
        self,
        raw_env_reward: float,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Decompose raw_env_reward into components without changing the scalar.
        """
        components: Dict[str, float] = {}
        # Pull known components if present
        for key in ("mpl_component", "ep_component", "error_penalty", "energy_penalty", "safety_bonus", "novelty_bonus"):
            if key in info:
                components[key] = float(info[key])
        components["scalar_reward"] = float(raw_env_reward)
        return float(raw_env_reward), components

    def compute_econ_vector(
        self,
        episode: Episode,
        events: List[EpisodeEvent],
    ) -> EconVector:
        """
        Aggregate econ signals from events. This mirrors existing fields but does
        not change training rewards.
        """
        reward_scalar_sum = sum(e.reward_scalar for e in events)
        mpl_units_per_hour = max(e.reward_components.get("mpl_component", 0.0) for e in events) if events else 0.0
        wage_parity = self._safe_float(self.config.get("wage_parity_stub"), 1.0)
        energy_cost = sum(e.reward_components.get("energy_penalty", 0.0) for e in events)
        damage_cost = sum(e.reward_components.get("collision_penalty", 0.0) for e in events)
        try:
            if self.policies:
                energy_feats = self.policies.energy_cost.build_features(events)
                energy_eval = self.policies.energy_cost.evaluate(energy_feats)
                energy_cost = self._safe_float(energy_eval.get("energy_cost", energy_cost), energy_cost)
                safety_feats = self.policies.safety_risk.build_features(events)
                safety_eval = self.policies.safety_risk.evaluate(safety_feats)
                damage_cost = self._safe_float(safety_eval.get("damage_estimate", damage_cost), damage_cost)
        except Exception:
            # Preserve existing behavior on any policy failure
            pass
        novelty_delta = max(e.reward_components.get("novelty_bonus", 0.0) for e in events) if events else 0.0
        components_agg: Dict[str, float] = {}
        mobility_penalty = 0.0
        precision_bonus = 0.0
        stability_risk_score = 0.0
        stability_vals = []
        recovery_events = 0
        for e in events:
            for k, v in e.reward_components.items():
                components_agg[k] = components_agg.get(k, 0.0) + self._safe_float(v)
            md = getattr(e, "metadata", {}) or {}
            mobility = md.get("mobility_adjustment", {}) if isinstance(md, dict) else {}
            recovery_required = mobility.get("recovery_required") if isinstance(mobility, dict) else None
            stability_margin = mobility.get("metadata", {}).get("stability_margin") if isinstance(mobility, dict) else None
            precision_gate = mobility.get("precision_gate_passed") if isinstance(mobility, dict) else None
            if recovery_required:
                mobility_penalty += 1.0
                recovery_events += 1
            if stability_margin is not None:
                stability_vals.append(self._safe_float(stability_margin))
            if precision_gate is False:
                mobility_penalty += 0.5
            if precision_gate is True and mobility.get("metadata", {}).get("drift_mm") is not None:
                precision_bonus += max(0.0, 1.0 - self._safe_float(mobility.get("metadata", {}).get("drift_mm") / 10.0))
        if stability_vals:
            stability_risk_score = 1.0 - min(1.0, sum(stability_vals) / len(stability_vals))

        raw_econ = EconVector(
            episode_id=episode.episode_id,
            mpl_units_per_hour=mpl_units_per_hour,
            wage_parity=wage_parity,
            energy_cost=energy_cost,
            damage_cost=damage_cost,
            novelty_delta=novelty_delta,
            reward_scalar_sum=reward_scalar_sum,
            mobility_penalty=mobility_penalty,
            precision_bonus=precision_bonus,
            stability_risk_score=stability_risk_score,
            components=components_agg,
            metadata={
                "task_id": episode.task_id,
                "robot_id": episode.robot_id,
                "computed_at": datetime.utcnow().isoformat(),
                "recovery_events": recovery_events,
            },
            source_domain=self.adapter.config.source_domain,
        )
        
        # Apply calibration
        calibrated_econ = self.adapter.map_vector(raw_econ)
        try:
            calibrated_econ.metadata.setdefault("raw_econ_vector", asdict(raw_econ))
        except Exception:
            calibrated_econ.metadata.setdefault("raw_econ_vector", {})
        return calibrated_econ

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

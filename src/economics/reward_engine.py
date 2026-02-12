"""
RewardEngine: advisory wrapper to decompose rewards and compute EconVectors.

Does NOT alter scalar rewards used by SAC/PPO; it only mirrors existing reward
math into logged components and episode-level EconVector aggregation.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

from src.ontology.models import Task, Robot, Episode, EpisodeEvent, EconVector
from src.policies.registry import build_all_policies
from src.economics.domain_adapter import EconDomainAdapter
from src.objectives.compiler import ObjectiveCompiler
from src.objectives.profile import ObjectiveProfile
from src.objectives.tensor import ObjectiveTensor, objective_tensor_from_axes


class RewardEngine:
    def __init__(
        self,
        task: Task,
        robot: Robot,
        config: Dict[str, Any],
        policies=None,
        econ_domain_name: str = "default",
        objective_profile: Optional[ObjectiveProfile] = None,
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
        configured_profile = self.config.get("objective_profile")
        if objective_profile is None and configured_profile:
            if isinstance(configured_profile, ObjectiveProfile):
                objective_profile = configured_profile
            else:
                objective_profile = ObjectiveProfile.from_dict(configured_profile)
        self.objective_profile = objective_profile
        self.objective_compiler = (
            ObjectiveCompiler(objective_profile) if objective_profile else None
        )

    def step_reward(
        self,
        raw_env_reward: float,
        info: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Decompose raw_env_reward into components without changing the scalar.
        """
        components: Dict[str, Any] = {}
        # Pull known components if present
        for key in (
            "mpl_component",
            "ep_component",
            "error_penalty",
            "energy_penalty",
            "safety_bonus",
            "novelty_bonus",
        ):
            if key in info:
                components[key] = float(info[key])
        components["scalar_reward"] = float(raw_env_reward)
        if self.objective_compiler is None:
            return float(raw_env_reward), components

        objective_tensor = self.compute_objective_tensor(
            episode_metrics=info or {},
            policy_metrics=None,
            map_first_metrics=info or {},
        )
        scalar_reward = self.objective_compiler.scalarize(objective_tensor)
        flags = self.objective_compiler.constraint_flags(objective_tensor)
        components["scalar_reward_legacy"] = float(raw_env_reward)
        components["scalar_reward_objective"] = float(scalar_reward)
        components["objective_tensor_v1"] = objective_tensor.to_dict()
        components["objective_constraint_flags"] = flags
        return float(scalar_reward), components

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
        mpl_units_per_hour = (
            max(e.reward_components.get("mpl_component", 0.0) for e in events) if events else 0.0
        )
        wage_parity = self._safe_float(self.config.get("wage_parity_stub"), 1.0)
        energy_cost = sum(e.reward_components.get("energy_penalty", 0.0) for e in events)
        damage_cost = sum(e.reward_components.get("collision_penalty", 0.0) for e in events)
        try:
            if self.policies:
                energy_feats = self.policies.energy_cost.build_features(events)
                energy_eval = self.policies.energy_cost.evaluate(energy_feats)
                energy_cost = self._safe_float(
                    energy_eval.get("energy_cost", energy_cost), energy_cost
                )
                safety_feats = self.policies.safety_risk.build_features(events)
                safety_eval = self.policies.safety_risk.evaluate(safety_feats)
                damage_cost = self._safe_float(
                    safety_eval.get("damage_estimate", damage_cost), damage_cost
                )
        except Exception:
            # Preserve existing behavior on any policy failure
            pass
        novelty_delta = (
            max(e.reward_components.get("novelty_bonus", 0.0) for e in events) if events else 0.0
        )
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
            recovery_required = (
                mobility.get("recovery_required") if isinstance(mobility, dict) else None
            )
            stability_margin = (
                mobility.get("metadata", {}).get("stability_margin")
                if isinstance(mobility, dict)
                else None
            )
            precision_gate = (
                mobility.get("precision_gate_passed") if isinstance(mobility, dict) else None
            )
            if recovery_required:
                mobility_penalty += 1.0
                recovery_events += 1
            if stability_margin is not None:
                stability_vals.append(self._safe_float(stability_margin))
            if precision_gate is False:
                mobility_penalty += 0.5
            if precision_gate is True and mobility.get("metadata", {}).get("drift_mm") is not None:
                precision_bonus += max(
                    0.0, 1.0 - self._safe_float(mobility.get("metadata", {}).get("drift_mm") / 10.0)
                )
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

    def compute_objective_tensor(
        self,
        episode_metrics: Dict[str, Any],
        policy_metrics: Optional[Dict[str, Any]] = None,
        map_first_metrics: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ObjectiveTensor:
        """Build ObjectiveTensor from episode/policy/map-first metric slices."""
        policy_metrics = policy_metrics or {}
        map_first_metrics = map_first_metrics or {}
        throughput = self._safe_float(
            episode_metrics.get(
                "mpl_component",
                episode_metrics.get("mpl_t", episode_metrics.get("throughput", 0.0)),
            )
        )
        error = self._safe_float(
            episode_metrics.get(
                "delta_errors",
                episode_metrics.get("error_penalty", episode_metrics.get("error_rate", 0.0)),
            )
        )
        error = abs(error)
        energy = self._safe_float(
            episode_metrics.get(
                "energy_penalty",
                episode_metrics.get("ep_t", episode_metrics.get("energy_Wh_per_unit", 0.0)),
            )
        )
        energy = abs(energy)

        map_quality = map_first_metrics.get("map_first_quality_score")
        if map_quality is None and isinstance(map_first_metrics.get("map_first_summary"), dict):
            map_quality = map_first_metrics.get("map_first_summary", {}).get(
                "map_first_quality_score"
            )
        map_quality = self._safe_float(map_quality, 1.0)

        safety_bonus = self._safe_float(episode_metrics.get("safety_bonus", 0.0))
        safety = max(0.0, 1.0 - min(1.0, error)) + max(0.0, safety_bonus) + max(0.0, map_quality)

        tensor = objective_tensor_from_axes(
            {
                "throughput": throughput,
                "error": error,
                "safety": safety,
                "energy": energy,
            },
            context={
                "task_id": self.task.task_id,
                "robot_id": self.robot.robot_id,
                **(context or {}),
            },
            provenance={
                "source": "reward_engine",
                "episode_metrics_keys": sorted(list((episode_metrics or {}).keys()))[:64],
                "policy_metrics_keys": sorted(list((policy_metrics or {}).keys()))[:64],
            },
        )
        return tensor

    def compute_objective_tensor_from_events(
        self,
        episode: Episode,
        events: List[EpisodeEvent],
        policy_metrics: Optional[Dict[str, Any]] = None,
    ) -> ObjectiveTensor:
        """Aggregate event stream into a single ObjectiveTensor artifact."""
        if not events:
            return self.compute_objective_tensor(
                episode_metrics={},
                policy_metrics=policy_metrics,
                context={"episode_id": episode.episode_id},
            )
        mpl_vals = []
        error_vals = []
        energy_vals = []
        safety_vals = []
        map_first_vals = []
        for event in events:
            comps = event.reward_components or {}
            mpl_vals.append(self._safe_float(comps.get("mpl_component", comps.get("mpl_t", 0.0))))
            error_vals.append(
                abs(self._safe_float(comps.get("error_penalty", comps.get("delta_errors", 0.0))))
            )
            energy_vals.append(
                abs(self._safe_float(comps.get("energy_penalty", comps.get("ep_t", 0.0))))
            )
            safety_vals.append(self._safe_float(comps.get("safety_bonus", 0.0)))
            map_first_vals.append(
                self._safe_float(
                    (event.metadata or {}).get(
                        "map_first_quality_score",
                        ((event.metadata or {}).get("map_first_summary") or {}).get(
                            "map_first_quality_score", 1.0
                        ),
                    ),
                    1.0,
                )
            )
        episode_metrics = {
            "mpl_component": float(np.mean(mpl_vals)) if mpl_vals else 0.0,
            "delta_errors": float(np.mean(error_vals)) if error_vals else 0.0,
            "energy_penalty": float(np.mean(energy_vals)) if energy_vals else 0.0,
            "safety_bonus": float(np.mean(safety_vals)) if safety_vals else 0.0,
            "map_first_quality_score": float(np.mean(map_first_vals)) if map_first_vals else 1.0,
        }
        return self.compute_objective_tensor(
            episode_metrics=episode_metrics,
            policy_metrics=policy_metrics,
            map_first_metrics=episode_metrics,
            context={
                "episode_id": episode.episode_id,
                "task_id": episode.task_id,
                "robot_id": episode.robot_id,
            },
        )

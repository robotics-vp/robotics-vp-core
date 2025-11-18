"""
RewardEngine: advisory wrapper to decompose rewards and compute EconVectors.

Does NOT alter scalar rewards used by SAC/PPO; it only mirrors existing reward
math into logged components and episode-level EconVector aggregation.
"""
from typing import Any, Dict, List, Tuple
from datetime import datetime

from src.ontology.models import Task, Robot, Episode, EpisodeEvent, EconVector


class RewardEngine:
    def __init__(self, task: Task, robot: Robot, config: Dict[str, Any]):
        self.task = task
        self.robot = robot
        self.config = config or {}

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
        novelty_delta = max(e.reward_components.get("novelty_bonus", 0.0) for e in events) if events else 0.0
        components_agg: Dict[str, float] = {}
        for e in events:
            for k, v in e.reward_components.items():
                components_agg[k] = components_agg.get(k, 0.0) + self._safe_float(v)

        return EconVector(
            episode_id=episode.episode_id,
            mpl_units_per_hour=mpl_units_per_hour,
            wage_parity=wage_parity,
            energy_cost=energy_cost,
            damage_cost=damage_cost,
            novelty_delta=novelty_delta,
            reward_scalar_sum=reward_scalar_sum,
            components=components_agg,
            metadata={
                "task_id": episode.task_id,
                "robot_id": episode.robot_id,
                "computed_at": datetime.utcnow().isoformat(),
            },
        )

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

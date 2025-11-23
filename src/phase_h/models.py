"""
Phase H data models: Skill, ExplorationBudget, SkillReturns.

Per PHASE_H_ECONOMIC_LEARNER_DESIGN.md.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from src.utils.json_safe import to_json_safe


class SkillStatus(Enum):
    """Skill lifecycle states."""
    EXPLORATION = "exploration"  # Early phase: high exploration budget
    TRAINING = "training"        # Active training: balanced exploration/exploitation
    MATURE = "mature"            # High performance: harvest returns
    DEPRECATED = "deprecated"    # Replaced by better skill variant


@dataclass
class Skill:
    """
    A skill is a trainable capability with economic attributes.
    """
    skill_id: str  # "drawer_open_v2", "dish_place_precision", etc.
    display_name: str
    description: str

    # Economic Stats
    mpl_baseline: float  # Initial MPL (units/hr)
    mpl_current: float   # Current MPL after training
    mpl_target: float    # Target MPL (economic goal)

    # Cost Stats
    training_cost_usd: float  # Cumulative training cost
    data_cost_per_episode: float  # Cost per additional training episode

    # Risk/Quality Stats
    success_rate: float  # Fraction of successful executions
    failure_rate: float  # Fraction of failures
    recovery_rate: float  # Fraction of failures recovered
    fragility_score: float  # 1.0 - failure_rate

    # Exploration Stats
    ood_exposure: float  # Fraction of training data that was OOD
    novelty_tier_avg: float  # Average novelty tier of training data

    # Metadata
    training_episodes: int
    last_updated: str  # ISO timestamp
    status: str = "training"  # Default to training

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dict."""
        return to_json_safe({
            "skill_id": self.skill_id,
            "display_name": self.display_name,
            "description": self.description,
            "mpl_baseline": self.mpl_baseline,
            "mpl_current": self.mpl_current,
            "mpl_target": self.mpl_target,
            "training_cost_usd": self.training_cost_usd,
            "data_cost_per_episode": self.data_cost_per_episode,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "recovery_rate": self.recovery_rate,
            "fragility_score": self.fragility_score,
            "ood_exposure": self.ood_exposure,
            "novelty_tier_avg": self.novelty_tier_avg,
            "training_episodes": self.training_episodes,
            "last_updated": self.last_updated,
            "status": self.status,
        })


@dataclass
class ExplorationBudget:
    """Exploration budget for a skill."""
    skill_id: str
    budget_usd: float  # Total budget for this skill
    spent_usd: float   # Cumulative spend
    remaining_usd: float  # budget_usd - spent_usd

    # Allocation breakdown
    data_collection_pct: float  # % for new data collection
    compute_training_pct: float  # % for model training
    human_supervision_pct: float  # % for expert demonstrations

    # Derived allocations
    max_episodes: int  # remaining_usd / data_cost_per_episode

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dict."""
        return to_json_safe({
            "skill_id": self.skill_id,
            "budget_usd": self.budget_usd,
            "spent_usd": self.spent_usd,
            "remaining_usd": self.remaining_usd,
            "data_collection_pct": self.data_collection_pct,
            "compute_training_pct": self.compute_training_pct,
            "human_supervision_pct": self.human_supervision_pct,
            "max_episodes": self.max_episodes,
        })


@dataclass
class SkillReturns:
    """Measured returns from a skill investment."""
    skill_id: str

    # Productivity Returns
    delta_mpl: float  # MPL improvement (units/hr)
    delta_mpl_pct: float  # Percentage improvement

    # Efficiency Returns
    delta_energy_wh: float  # Energy savings (Wh)
    delta_time_sec: float  # Time savings (sec/task)

    # Quality Returns
    delta_damage: float  # Damage cost reduction ($)
    delta_success_rate: float  # Success rate improvement

    # Exploration Returns
    delta_novelty_coverage: float  # New state-action coverage
    unique_failure_modes_discovered: int

    # Financial Returns
    roi_pct: float  # (returns - costs) / costs * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-safe dict."""
        return to_json_safe({
            "skill_id": self.skill_id,
            "delta_mpl": self.delta_mpl,
            "delta_mpl_pct": self.delta_mpl_pct,
            "delta_energy_wh": self.delta_energy_wh,
            "delta_time_sec": self.delta_time_sec,
            "delta_damage": self.delta_damage,
            "delta_success_rate": self.delta_success_rate,
            "delta_novelty_coverage": self.delta_novelty_coverage,
            "unique_failure_modes_discovered": self.unique_failure_modes_discovered,
            "roi_pct": self.roi_pct,
        })


def update_skill_status(skill: Skill) -> str:
    """
    Update skill status based on performance.

    Rules:
    - If success_rate > 0.95 AND mpl_current >= mpl_target → MATURE
    - If success_rate < 0.6 → EXPLORATION
    - Else → TRAINING
    """
    if skill.success_rate > 0.95 and skill.mpl_current >= skill.mpl_target:
        return SkillStatus.MATURE.value

    if skill.success_rate < 0.6:
        return SkillStatus.EXPLORATION.value

    return SkillStatus.TRAINING.value

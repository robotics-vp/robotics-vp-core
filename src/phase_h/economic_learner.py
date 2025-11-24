"""
Economic Learner: dynamic skill portfolio management with ROI-based budget allocation.

Per PHASE_H_ECONOMIC_LEARNER_DESIGN.md.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.phase_h.models import Skill, SkillStatus, ExplorationBudget, SkillReturns, update_skill_status


def allocate_exploration_budget(
    skill: Skill,
    total_budget_usd: float,
    mpl_gap: float,
) -> ExplorationBudget:
    """
    Allocate exploration budget for a skill.

    Logic:
    - Larger MPL gap → larger budget
    - Lower success rate → more budget for data
    - Higher novelty exposure → more budget for diverse data
    """
    # Base allocation proportional to MPL gap
    base_budget = max(0, mpl_gap) * 200.0  # $200 per unit MPL gap

    # Quality penalty: low success → need more training
    quality_penalty = 1.0 - skill.success_rate
    adjusted_budget = base_budget * (1.0 + quality_penalty)

    # Novelty bonus: high OOD exposure → more valuable
    novelty_bonus = skill.ood_exposure * 0.5
    final_budget = adjusted_budget * (1.0 + novelty_bonus)

    # Cap at total available budget
    final_budget = min(final_budget, total_budget_usd)

    # Allocate breakdown based on skill status
    if skill.status == SkillStatus.EXPLORATION.value:
        # Heavy data collection
        data_pct = 0.6
        compute_pct = 0.3
        human_pct = 0.1
    elif skill.status == SkillStatus.TRAINING.value:
        # Balanced
        data_pct = 0.4
        compute_pct = 0.5
        human_pct = 0.1
    else:  # MATURE or DEPRECATED
        # Minimal budget (maintenance only)
        data_pct = 0.2
        compute_pct = 0.7
        human_pct = 0.1

    max_episodes = int(final_budget * data_pct / skill.data_cost_per_episode) if skill.data_cost_per_episode > 0 else 0

    return ExplorationBudget(
        skill_id=skill.skill_id,
        budget_usd=final_budget,
        spent_usd=0.0,
        remaining_usd=final_budget,
        data_collection_pct=data_pct,
        compute_training_pct=compute_pct,
        human_supervision_pct=human_pct,
        max_episodes=max_episodes,
    )


def compute_skill_roi(
    skill: Skill,
    returns: SkillReturns,
    wage_target_usd_per_hr: float = 18.0,
    price_per_unit: float = 0.30,
    hours_deployed: int = 1000,
    energy_price_per_kwh: float = 0.12,
    previous_roi: Optional[float] = None,
    alpha: float = 0.3,
) -> float:
    """
    Compute return on investment for a skill.

    ROI = (Value Created - Training Cost) / Training Cost * 100

    Value Created = ΔMP product value + energy savings + damage savings
    """
    # Productivity value: ΔMP * price_per_unit * hours_deployed
    productivity_value = returns.delta_mpl * price_per_unit * hours_deployed

    # Efficiency savings
    energy_savings = (abs(returns.delta_energy_wh) / 1000.0) * energy_price_per_kwh * hours_deployed

    # Quality savings
    damage_savings = abs(returns.delta_damage) * hours_deployed

    # Total value created
    total_value = productivity_value + energy_savings + damage_savings

    # ROI
    if skill.training_cost_usd > 0:
        roi_pct = (total_value - skill.training_cost_usd) / skill.training_cost_usd * 100.0
    else:
        roi_pct = 0.0

    if previous_roi is not None:
        roi_pct = (alpha * roi_pct) + ((1 - alpha) * float(previous_roi))

    return roi_pct


class EconomicLearner:
    """
    Manages skill portfolio and budget allocation.

    Runs periodic learner cycles to:
    1. Measure returns
    2. Compute ROI
    3. Reallocate budgets
    4. Update skill statuses
    5. Generate reports
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize economic learner.

        Args:
            config: Dict with:
                - total_exploration_budget: Total USD budget
                - reallocation_period_episodes: Episodes between reallocations
                - price_per_unit: Price per output unit
                - hours_deployed: Deployment horizon for ROI calc
        """
        self.total_budget_usd = float(config.get("total_exploration_budget", 10000.0))
        self.reallocation_period = int(config.get("reallocation_period_episodes", 1000))
        self.price_per_unit = float(config.get("price_per_unit", 0.30))
        self.hours_deployed = int(config.get("hours_deployed", 1000))

        self.skills: Dict[str, Skill] = {}
        self.budgets: Dict[str, ExplorationBudget] = {}
        self.returns_history: List[SkillReturns] = []
        self.roi_history: Dict[str, float] = {}

    def add_skill(self, skill: Skill):
        """Add a skill to the portfolio."""
        self.skills[skill.skill_id] = skill

        # Allocate initial budget
        mpl_gap = max(0, skill.mpl_target - skill.mpl_current)
        self.budgets[skill.skill_id] = allocate_exploration_budget(
            skill, self.total_budget_usd, mpl_gap
        )

    def run_cycle(self, episode_count: int) -> Optional[Dict[str, Any]]:
        """
        Run one learner cycle.

        Steps:
        1. Measure returns for all skills
        2. Compute ROI
        3. Reallocate budgets
        4. Update skill statuses
        5. Generate market state report

        Returns:
            Dict with cycle summary, or None if not time for cycle
        """
        if episode_count % self.reallocation_period != 0:
            return None  # Not time for cycle yet

        # 1. Measure returns (stub: would integrate with actual training metrics)
        returns_by_skill = self._measure_all_returns()

        # 2. Compute ROI
        roi_by_skill = {}
        for skill_id, skill in self.skills.items():
            if skill_id in returns_by_skill:
                roi_by_skill[skill_id] = returns_by_skill[skill_id].roi_pct

        # 3. Reallocate budgets
        self._reallocate_budgets(roi_by_skill)

        # 4. Update statuses
        for skill_id, skill in self.skills.items():
            skill.status = update_skill_status(skill)

        # 5. Generate report
        summary = {
            "episode_count": episode_count,
            "total_budget_usd": self.total_budget_usd,
            "skill_count": len(self.skills),
            "roi_by_skill": roi_by_skill,
            "budgets": {sid: b.to_dict() for sid, b in self.budgets.items()},
        }

        return summary

    def _measure_all_returns(self) -> Dict[str, SkillReturns]:
        """
        Measure returns for all skills.

        In real implementation, would integrate with:
        - SIMA-2 quality signals
        - Ontology MPL metrics
        - RECAP scores
        """
        returns_by_skill = {}

        for skill_id, skill in self.skills.items():
            # Stub: compute delta from baseline
            delta_mpl = skill.mpl_current - skill.mpl_baseline
            delta_mpl_pct = (delta_mpl / skill.mpl_baseline * 100.0) if skill.mpl_baseline > 0 else 0.0

            returns = SkillReturns(
                skill_id=skill_id,
                delta_mpl=delta_mpl,
                delta_mpl_pct=delta_mpl_pct,
                delta_energy_wh=0.0,  # Stub
                delta_time_sec=0.0,  # Stub
                delta_damage=0.0,  # Stub
                delta_success_rate=skill.success_rate - 0.5,  # Stub: assume baseline 0.5
                delta_novelty_coverage=skill.ood_exposure,
                unique_failure_modes_discovered=0,  # Stub
                roi_pct=0.0,  # Will be computed
            )

            # Compute ROI for this skill (EMA smoothed)
            previous_roi = self.roi_history.get(skill_id)
            returns.roi_pct = compute_skill_roi(
                skill,
                returns,
                price_per_unit=self.price_per_unit,
                hours_deployed=self.hours_deployed,
                previous_roi=previous_roi,
            )
            self.roi_history[skill_id] = returns.roi_pct

            returns_by_skill[skill_id] = returns
            self.returns_history.append(returns)

        return returns_by_skill

    def _reallocate_budgets(self, roi_by_skill: Dict[str, float]):
        """
        Reallocate exploration budget based on ROI.

        Strategy:
        - High ROI skills: Increase budget by 20% (max)
        - Low ROI skills: Decrease budget by 20% (max)
        - Negative ROI skills: Consider deprecation

        Bounds: ±20% per cycle (Phase H requirement).
        """
        MAX_INCREASE = 1.2
        MAX_DECREASE = 0.8

        planned_budgets: Dict[str, ExplorationBudget] = {}

        for skill_id, roi in roi_by_skill.items():
            if skill_id not in self.skills or skill_id not in self.budgets:
                continue

            skill = self.skills[skill_id]
            current_budget = self.budgets[skill_id].budget_usd

            if roi > 50.0:
                # High ROI: invest more
                new_budget = current_budget * MAX_INCREASE
            elif roi > 0.0:
                # Positive ROI: maintain
                new_budget = current_budget
            else:
                # Negative ROI: reduce or deprecate
                new_budget = current_budget * MAX_DECREASE

                if roi < -50.0:
                    skill.status = SkillStatus.DEPRECATED.value

            mpl_gap = max(0, skill.mpl_target - skill.mpl_current)
            planned_budgets[skill_id] = allocate_exploration_budget(skill, new_budget, mpl_gap)

        # Enforce global exploration budget cap (20% of total budget)
        cap = 0.2 * self.total_budget_usd
        exploration_total = sum(
            b.budget_usd
            for sid, b in planned_budgets.items()
            if self.skills.get(sid) and self.skills[sid].status == SkillStatus.EXPLORATION.value
        )
        if exploration_total > cap and exploration_total > 0:
            scale = cap / exploration_total
            for skill_id, budget in list(planned_budgets.items()):
                skill = self.skills.get(skill_id)
                if skill is None or skill.status != SkillStatus.EXPLORATION.value:
                    continue
                scaled_budget_usd = budget.budget_usd * scale
                scaled_remaining = min(budget.remaining_usd, scaled_budget_usd)
                max_episodes = (
                    int(scaled_budget_usd * budget.data_collection_pct / skill.data_cost_per_episode)
                    if skill.data_cost_per_episode > 0
                    else 0
                )
                planned_budgets[skill_id] = ExplorationBudget(
                    skill_id=budget.skill_id,
                    budget_usd=scaled_budget_usd,
                    spent_usd=min(budget.spent_usd, scaled_budget_usd),
                    remaining_usd=scaled_remaining,
                    data_collection_pct=budget.data_collection_pct,
                    compute_training_pct=budget.compute_training_pct,
                    human_supervision_pct=budget.human_supervision_pct,
                    max_episodes=max_episodes,
                )

        for skill_id, budget in planned_budgets.items():
            self.budgets[skill_id] = budget

    def save_artifacts(self, output_dir: Path):
        """
        Save Phase H artifacts to disk.

        Generates:
        - skill_market_state.json
        - exploration_budget.json
        - skill_returns.json
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Skill market state
        market_state = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_budget_usd": self.total_budget_usd,
            "allocated_usd": sum(b.budget_usd for b in self.budgets.values()),
            "skills": {sid: skill.to_dict() for sid, skill in self.skills.items()},
        }

        with open(output_dir / "skill_market_state.json", "w") as f:
            json.dump(market_state, f, indent=2)

        # Exploration budget
        budget_state = {
            "total_budget_usd": self.total_budget_usd,
            "reallocation_period_episodes": self.reallocation_period,
            "budgets_by_skill": {sid: budget.to_dict() for sid, budget in self.budgets.items()},
        }

        with open(output_dir / "exploration_budget.json", "w") as f:
            json.dump(budget_state, f, indent=2)

        # Skill returns (latest)
        if self.returns_history:
            returns_state = {
                "returns": [r.to_dict() for r in self.returns_history[-len(self.skills):]],
            }

            with open(output_dir / "skill_returns.json", "w") as f:
                json.dump(returns_state, f, indent=2)

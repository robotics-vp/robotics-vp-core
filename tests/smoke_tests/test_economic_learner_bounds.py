import unittest

from src.phase_h.economic_learner import EconomicLearner, compute_skill_roi
from src.phase_h.models import Skill, SkillStatus, ExplorationBudget, SkillReturns


def _make_skill(skill_id: str, status: str) -> Skill:
    return Skill(
        skill_id=skill_id,
        display_name=f"Skill {skill_id}",
        description="Test skill",
        mpl_baseline=10.0,
        mpl_current=12.0,
        mpl_target=20.0,
        training_cost_usd=100.0,
        data_cost_per_episode=1.0,
        success_rate=0.8,
        failure_rate=0.1,
        recovery_rate=0.1,
        fragility_score=0.1,
        ood_exposure=0.1,
        novelty_tier_avg=1.0,
        training_episodes=100,
        last_updated="2025-01-01T00:00:00Z",
        status=status,
    )


class EconomicLearnerBoundsTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "total_exploration_budget": 10000.0,
            "reallocation_period_episodes": 10,
            "price_per_unit": 0.30,
            "hours_deployed": 1000,
        }

    def test_roi_ema_applied(self):
        """EMA should smooth ROI with alpha=0.3."""
        skill = _make_skill("roi_skill", SkillStatus.TRAINING.value)
        returns = SkillReturns(
            skill_id=skill.skill_id,
            delta_mpl=0.0,
            delta_mpl_pct=0.0,
            delta_energy_wh=0.0,
            delta_time_sec=0.0,
            delta_damage=0.1,  # Yields total_value ~= training_cost for ROI ~= 0
            delta_success_rate=0.0,
            delta_novelty_coverage=0.0,
            unique_failure_modes_discovered=0,
            roi_pct=0.0,
        )

        roi = compute_skill_roi(skill, returns=returns, previous_roi=100.0, alpha=0.3)
        self.assertAlmostEqual(roi, 70.0, places=1)

    def test_per_skill_budget_bounds(self):
        """Per-skill budget change stays within ±20%."""
        learner = EconomicLearner(self.config)
        skill = _make_skill("bound_skill", SkillStatus.TRAINING.value)
        learner.add_skill(skill)
        learner.budgets[skill.skill_id] = ExplorationBudget(
            skill_id=skill.skill_id,
            budget_usd=1000.0,
            spent_usd=0.0,
            remaining_usd=1000.0,
            data_collection_pct=0.4,
            compute_training_pct=0.5,
            human_supervision_pct=0.1,
            max_episodes=100,
        )

        learner._reallocate_budgets({skill.skill_id: 100.0})
        self.assertAlmostEqual(learner.budgets[skill.skill_id].budget_usd, 1200.0, places=2)

        learner._reallocate_budgets({skill.skill_id: -100.0})
        self.assertAlmostEqual(learner.budgets[skill.skill_id].budget_usd, 960.0, places=2)

    def test_global_exploration_cap(self):
        """Exploration budgets are globally capped at 20% of total budget."""
        learner = EconomicLearner(self.config)

        skill_a = _make_skill("exp_a", SkillStatus.EXPLORATION.value)
        skill_b = _make_skill("exp_b", SkillStatus.EXPLORATION.value)
        learner.add_skill(skill_a)
        learner.add_skill(skill_b)

        # Set deterministic starting budgets to 1500 each (exploration status)
        for sid in (skill_a.skill_id, skill_b.skill_id):
            learner.budgets[sid] = ExplorationBudget(
                skill_id=sid,
                budget_usd=1500.0,
                spent_usd=0.0,
                remaining_usd=1500.0,
                data_collection_pct=0.6,
                compute_training_pct=0.3,
                human_supervision_pct=0.1,
                max_episodes=900,
            )

        learner._reallocate_budgets({skill_a.skill_id: 100.0, skill_b.skill_id: 100.0})

        cap = 0.2 * self.config["total_exploration_budget"]  # 2000
        budget_a = learner.budgets[skill_a.skill_id]
        budget_b = learner.budgets[skill_b.skill_id]

        self.assertLessEqual(budget_a.budget_usd + budget_b.budget_usd, cap + 1e-3)
        self.assertAlmostEqual(budget_a.budget_usd, 1000.0, places=1)
        self.assertAlmostEqual(budget_b.budget_usd, 1000.0, places=1)
        self.assertEqual(budget_a.data_collection_pct, 0.6)
        self.assertEqual(budget_b.data_collection_pct, 0.6)
        self.assertEqual(budget_a.max_episodes, 600)
        self.assertEqual(budget_b.max_episodes, 600)


if __name__ == "__main__":
    unittest.main()

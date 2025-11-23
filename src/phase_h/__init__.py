"""
Phase H: Economic Learner & Skill Market.

Skills as assets with measurable ROI:
- Cost to train (data, compute, supervision)
- Returns (MPL, energy efficiency, quality)
- Dynamic budget allocation based on ROI
"""

from src.phase_h.models import Skill, SkillStatus, ExplorationBudget, SkillReturns
from src.phase_h.economic_learner import EconomicLearner, allocate_exploration_budget, compute_skill_roi

__all__ = [
    "Skill",
    "SkillStatus",
    "ExplorationBudget",
    "SkillReturns",
    "EconomicLearner",
    "allocate_exploration_budget",
    "compute_skill_roi",
]

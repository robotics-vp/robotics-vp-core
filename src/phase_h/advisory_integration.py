"""
Phase H Advisory Integration: Wire Economic Learner outputs into Sampler + Orchestrator.

ADVISORY ONLY - does not mutate reward or econ-controller.
All changes bounded to ±20%.
Flag-gated: enable_phase_h_advisories.

Per Phase H System Integration Requirements.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.phase_h.models import ExplorationBudget, Skill, SkillReturns
from src.utils.json_safe import to_json_safe


# Advisory boundaries (hard limits)
MIN_MULTIPLIER = 0.8
MAX_MULTIPLIER = 1.2
MAX_ROUTING_DELTA = 0.20  # 20% max change


def load_skill_market_state(ontology_root: Path) -> Dict[str, Skill]:
    """
    Load skill market state from ontology.

    Returns:
        Dict[skill_id, Skill]
    """
    market_path = ontology_root / "phase_h" / "skill_market_state.json"
    if not market_path.exists():
        return {}

    with open(market_path, "r") as f:
        data = json.load(f)

    skills = {}
    for skill_id, skill_dict in data.get("skills", {}).items():
        skills[skill_id] = Skill(
            skill_id=skill_dict["skill_id"],
            display_name=skill_dict["display_name"],
            description=skill_dict["description"],
            mpl_baseline=float(skill_dict["mpl_baseline"]),
            mpl_current=float(skill_dict["mpl_current"]),
            mpl_target=float(skill_dict["mpl_target"]),
            training_cost_usd=float(skill_dict["training_cost_usd"]),
            data_cost_per_episode=float(skill_dict["data_cost_per_episode"]),
            success_rate=float(skill_dict["success_rate"]),
            failure_rate=float(skill_dict["failure_rate"]),
            recovery_rate=float(skill_dict["recovery_rate"]),
            fragility_score=float(skill_dict["fragility_score"]),
            ood_exposure=float(skill_dict["ood_exposure"]),
            novelty_tier_avg=float(skill_dict["novelty_tier_avg"]),
            training_episodes=int(skill_dict["training_episodes"]),
            last_updated=str(skill_dict["last_updated"]),
            status=str(skill_dict.get("status", "training")),
        )

    return skills


def load_exploration_budget(ontology_root: Path) -> Dict[str, ExplorationBudget]:
    """
    Load exploration budgets from ontology.

    Returns:
        Dict[skill_id, ExplorationBudget]
    """
    budget_path = ontology_root / "phase_h" / "exploration_budget.json"
    if not budget_path.exists():
        return {}

    with open(budget_path, "r") as f:
        data = json.load(f)

    budgets = {}
    for skill_id, budget_dict in data.get("budgets_by_skill", {}).items():
        budgets[skill_id] = ExplorationBudget(
            skill_id=budget_dict["skill_id"],
            budget_usd=float(budget_dict["budget_usd"]),
            spent_usd=float(budget_dict["spent_usd"]),
            remaining_usd=float(budget_dict["remaining_usd"]),
            data_collection_pct=float(budget_dict["data_collection_pct"]),
            compute_training_pct=float(budget_dict["compute_training_pct"]),
            human_supervision_pct=float(budget_dict["human_supervision_pct"]),
            max_episodes=int(budget_dict["max_episodes"]),
        )

    return budgets


def load_skill_returns(ontology_root: Path) -> List[SkillReturns]:
    """
    Load skill returns history from ontology.

    Returns:
        List of SkillReturns (latest first)
    """
    returns_path = ontology_root / "phase_h" / "skill_returns.json"
    if not returns_path.exists():
        return []

    with open(returns_path, "r") as f:
        data = json.load(f)

    returns = []
    for return_dict in data.get("returns", []):
        returns.append(SkillReturns(
            skill_id=return_dict["skill_id"],
            delta_mpl=float(return_dict["delta_mpl"]),
            delta_mpl_pct=float(return_dict["delta_mpl_pct"]),
            delta_energy_wh=float(return_dict["delta_energy_wh"]),
            delta_time_sec=float(return_dict["delta_time_sec"]),
            delta_damage=float(return_dict["delta_damage"]),
            delta_success_rate=float(return_dict["delta_success_rate"]),
            delta_novelty_coverage=float(return_dict["delta_novelty_coverage"]),
            unique_failure_modes_discovered=int(return_dict["unique_failure_modes_discovered"]),
            roi_pct=float(return_dict["roi_pct"]),
        ))

    return returns


class PhaseHAdvisory:
    """
    Advisory signals from Phase H Economic Learner.

    Read-only, bounded, flag-gated.
    """

    def __init__(
        self,
        skills: Dict[str, Skill],
        budgets: Dict[str, ExplorationBudget],
        returns: List[SkillReturns],
    ):
        self.skills = skills
        self.budgets = budgets
        self.returns = returns

        # Precompute advisory signals
        self.skill_multipliers = self._compute_skill_multipliers()
        self.skill_quality_signals = self._compute_skill_quality_signals()
        self.exploration_priorities = self._compute_exploration_priorities()
        self.routing_advisories = self._compute_routing_advisories()

    def _compute_skill_multipliers(self) -> Dict[str, float]:
        """
        Compute skill_suggested_multiplier for each skill.

        Logic:
        - Skills with high ROI → upweight (1.1-1.2)
        - Skills with low/negative ROI → downweight (0.8-0.9)
        - Exploration status skills → upweight for data collection (1.15)

        Bounded to [0.8, 1.2].
        """
        multipliers = {}

        # Build ROI map from latest returns
        roi_by_skill = {}
        for skill_return in self.returns:
            roi_by_skill[skill_return.skill_id] = skill_return.roi_pct

        for skill_id, skill in self.skills.items():
            roi = roi_by_skill.get(skill_id, 0.0)

            # Base multiplier
            if roi > 50.0:
                # High ROI: encourage more sampling
                mult = 1.2
            elif roi > 20.0:
                mult = 1.1
            elif roi > 0.0:
                mult = 1.0
            elif roi > -20.0:
                mult = 0.9
            else:
                # Negative ROI: reduce sampling
                mult = 0.8

            # Status adjustments
            if skill.status == "exploration":
                # Exploration skills need more data
                mult = min(MAX_MULTIPLIER, mult * 1.05)
            elif skill.status == "deprecated":
                # Deprecated skills should be downweighted
                mult = MIN_MULTIPLIER

            # Clamp to bounds
            mult = max(MIN_MULTIPLIER, min(MAX_MULTIPLIER, mult))
            multipliers[skill_id] = mult

        return multipliers

    def _compute_skill_quality_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute quality signals for each skill.

        Returns:
            Dict[skill_id, {
                "success_rate": float,
                "fragility_score": float,
                "ood_exposure": float,
                "mpl_progress": float,  # (current - baseline) / (target - baseline)
            }]
        """
        quality = {}

        for skill_id, skill in self.skills.items():
            mpl_range = skill.mpl_target - skill.mpl_baseline
            mpl_progress = 0.0
            if mpl_range > 0:
                mpl_progress = (skill.mpl_current - skill.mpl_baseline) / mpl_range
            mpl_progress = max(0.0, min(1.0, mpl_progress))

            quality[skill_id] = {
                "success_rate": skill.success_rate,
                "fragility_score": skill.fragility_score,
                "ood_exposure": skill.ood_exposure,
                "mpl_progress": mpl_progress,
            }

        return quality

    def _compute_exploration_priorities(self) -> Dict[str, float]:
        """
        Compute exploration priority for each skill.

        Logic:
        - High budget remaining → high priority
        - Low MPL progress → high priority
        - Exploration/Training status → high priority

        Returns priority in [0, 1].
        """
        priorities = {}

        for skill_id, skill in self.skills.items():
            budget = self.budgets.get(skill_id)
            if not budget:
                priorities[skill_id] = 0.0
                continue

            # Budget factor: remaining / total
            budget_factor = budget.remaining_usd / budget.budget_usd if budget.budget_usd > 0 else 0.0

            # Progress factor: inverse of MPL progress
            quality = self.skill_quality_signals.get(skill_id, {})
            progress = quality.get("mpl_progress", 0.0)
            progress_factor = 1.0 - progress

            # Status factor
            if skill.status == "exploration":
                status_factor = 1.0
            elif skill.status == "training":
                status_factor = 0.7
            elif skill.status == "mature":
                status_factor = 0.3
            else:  # deprecated
                status_factor = 0.0

            # Combine factors
            priority = (0.4 * budget_factor + 0.4 * progress_factor + 0.2 * status_factor)
            priority = max(0.0, min(1.0, priority))

            priorities[skill_id] = priority

        return priorities

    def _compute_routing_advisories(self) -> Dict[str, Any]:
        """
        Compute routing advisories for Orchestrator.

        Returns:
            {
                "frontier_emphasis": float [0, 1],
                "safety_emphasis": float [0, 1],
                "efficiency_emphasis": float [0, 1],
                "skill_mode_suggestion": str,
            }
        """
        # Aggregate skill signals
        total_skills = len(self.skills)
        if total_skills == 0:
            return {
                "frontier_emphasis": 0.5,
                "safety_emphasis": 0.5,
                "efficiency_emphasis": 0.5,
                "skill_mode_suggestion": "efficiency_throughput",
            }

        # Count skills by status
        exploration_count = sum(1 for s in self.skills.values() if s.status == "exploration")
        mature_count = sum(1 for s in self.skills.values() if s.status == "mature")

        # Frontier emphasis: high if many exploration skills
        frontier_emphasis = exploration_count / total_skills

        # Safety emphasis: high if low average success rate
        avg_success = sum(s.success_rate for s in self.skills.values()) / total_skills
        safety_emphasis = 1.0 - avg_success

        # Efficiency emphasis: high if many mature skills
        efficiency_emphasis = mature_count / total_skills

        # Normalize to ensure sum ≈ 1.5 (not exactly 1, to allow flexibility)
        total_emphasis = frontier_emphasis + safety_emphasis + efficiency_emphasis
        if total_emphasis > 0:
            scale = 1.5 / total_emphasis
            frontier_emphasis *= scale
            safety_emphasis *= scale
            efficiency_emphasis *= scale

        # Clamp to [0, 1]
        frontier_emphasis = max(0.0, min(1.0, frontier_emphasis))
        safety_emphasis = max(0.0, min(1.0, safety_emphasis))
        efficiency_emphasis = max(0.0, min(1.0, efficiency_emphasis))

        # Suggest skill mode
        if frontier_emphasis > 0.6:
            skill_mode_suggestion = "frontier_exploration"
        elif safety_emphasis > 0.6:
            skill_mode_suggestion = "safety_critical"
        elif efficiency_emphasis > 0.6:
            skill_mode_suggestion = "efficiency_throughput"
        else:
            skill_mode_suggestion = "balanced"

        return {
            "frontier_emphasis": frontier_emphasis,
            "safety_emphasis": safety_emphasis,
            "efficiency_emphasis": efficiency_emphasis,
            "skill_mode_suggestion": skill_mode_suggestion,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export advisory signals as JSON-safe dict."""
        return to_json_safe({
            "skill_multipliers": self.skill_multipliers,
            "skill_quality_signals": self.skill_quality_signals,
            "exploration_priorities": self.exploration_priorities,
            "routing_advisories": self.routing_advisories,
        })


def apply_sampler_advisory(
    base_weights: Dict[str, float],
    advisory: PhaseHAdvisory,
    enable_phase_h_advisories: bool = False,
) -> Dict[str, float]:
    """
    Apply Phase H advisory to sampler weights.

    Args:
        base_weights: Original weights from HeuristicSamplerWeightPolicy
        advisory: Phase H advisory signals
        enable_phase_h_advisories: Flag to enable advisory (default False)

    Returns:
        Adjusted weights (bounded to ±20% change)
    """
    if not enable_phase_h_advisories:
        return base_weights

    # Match episodes to skills (stub: would use metadata)
    # For now, apply a global uplift based on average exploration priority
    avg_priority = sum(advisory.exploration_priorities.values()) / max(len(advisory.exploration_priorities), 1)

    # Global multiplier in [0.9, 1.1] based on exploration priority
    global_mult = 0.9 + (avg_priority * 0.2)

    adjusted_weights = {}
    for episode_key, base_weight in base_weights.items():
        # Apply global multiplier
        adjusted = base_weight * global_mult

        # Ensure bounded change (±20%)
        max_allowed = base_weight * (1.0 + MAX_ROUTING_DELTA)
        min_allowed = base_weight * (1.0 - MAX_ROUTING_DELTA)
        adjusted = max(min_allowed, min(max_allowed, adjusted))

        adjusted_weights[episode_key] = adjusted

    return adjusted_weights


def apply_orchestrator_advisory(
    base_advisory: Any,
    phase_h_advisory: PhaseHAdvisory,
    enable_phase_h_advisories: bool = False,
) -> Any:
    """
    Apply Phase H routing advisory to Orchestrator.

    Args:
        base_advisory: OrchestratorAdvisory from SemanticOrchestratorV2
        phase_h_advisory: Phase H advisory signals
        enable_phase_h_advisories: Flag to enable advisory (default False)

    Returns:
        Updated OrchestratorAdvisory (bounded to ±20% routing change)
    """
    if not enable_phase_h_advisories:
        return base_advisory

    routing = phase_h_advisory.routing_advisories

    # Blend safety_emphasis (bounded to ±20%)
    base_safety = base_advisory.safety_emphasis
    suggested_safety = routing["safety_emphasis"]

    # Bounded blend: move toward suggested by at most 20%
    delta = suggested_safety - base_safety
    clamped_delta = max(-MAX_ROUTING_DELTA, min(MAX_ROUTING_DELTA, delta))
    adjusted_safety = base_safety + clamped_delta
    adjusted_safety = max(0.0, min(1.0, adjusted_safety))

    # Update advisory
    base_advisory.safety_emphasis = adjusted_safety

    # Add Phase H metadata
    if not hasattr(base_advisory, "metadata"):
        base_advisory.metadata = {}
    base_advisory.metadata["phase_h_routing"] = {
        "frontier_emphasis": routing["frontier_emphasis"],
        "efficiency_emphasis": routing["efficiency_emphasis"],
        "skill_mode_suggestion": routing["skill_mode_suggestion"],
    }

    return base_advisory


def build_phase_h_condition_fields(
    advisory: PhaseHAdvisory,
    skill_id: Optional[str] = None,
    enable_phase_h_advisories: bool = False,
) -> Dict[str, Any]:
    """
    Build Phase H fields for ConditionVector.

    Args:
        advisory: Phase H advisory signals
        skill_id: Optional skill ID to retrieve specific signals
        enable_phase_h_advisories: Flag to enable fields (default False)

    Returns:
        Dict with exploration_uplift and skill_roi_estimate (only if flag enabled)
    """
    if not enable_phase_h_advisories:
        return {}

    # Default values
    exploration_uplift = 0.0
    skill_roi_estimate = 0.0

    if skill_id and skill_id in advisory.exploration_priorities:
        exploration_uplift = advisory.exploration_priorities[skill_id]
    else:
        # Global average
        if advisory.exploration_priorities:
            exploration_uplift = sum(advisory.exploration_priorities.values()) / len(advisory.exploration_priorities)

    # ROI estimate from latest returns
    if skill_id:
        for skill_return in advisory.returns:
            if skill_return.skill_id == skill_id:
                skill_roi_estimate = skill_return.roi_pct
                break
    else:
        # Global average ROI
        if advisory.returns:
            skill_roi_estimate = sum(r.roi_pct for r in advisory.returns) / len(advisory.returns)

    return {
        "exploration_uplift": float(exploration_uplift),
        "skill_roi_estimate": float(skill_roi_estimate),
    }


def load_phase_h_advisory(ontology_root: Path) -> Optional[PhaseHAdvisory]:
    """
    Load Phase H advisory from ontology artifacts.

    Args:
        ontology_root: Path to ontology root

    Returns:
        PhaseHAdvisory or None if artifacts not found
    """
    try:
        skills = load_skill_market_state(ontology_root)
        budgets = load_exploration_budget(ontology_root)
        returns = load_skill_returns(ontology_root)

        if not skills:
            return None

        return PhaseHAdvisory(skills, budgets, returns)
    except Exception:
        return None

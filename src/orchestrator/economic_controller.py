from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np

from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams
from src.valuation.datapack_schema import AttributionProfile, DataPackMeta, ObjectiveProfile
from src.valuation.reward_builder import build_reward_terms
from src.valuation.energy_response_model import EnergyResponseNet
from src.orchestrator.semantic_metrics import SemanticMetrics


@dataclass
class EconSignals:
    """
    Economic signals computed from episodes/datapacks.

    These are the PRIMARY economic constraints that downstream modules
    (SemanticOrchestrator, VLA, SIMA, etc.) must respect.

    IMPORTANT: This is UPSTREAM of SemanticOrchestrator.
    SemanticOrchestrator consumes these signals - it does not define them.
    """
    # Core MPL metrics
    current_mpl: float = 0.0
    baseline_mpl_human: float = 60.0
    mpl_delta: float = 0.0
    mpl_trend: float = 0.0

    # Error and quality metrics
    error_rate: float = 0.0
    error_trend: float = 0.0
    damage_cost_total: float = 0.0

    # Wage parity (core economic goal)
    implied_wage: float = 0.0
    human_wage: float = 18.0
    wage_parity: float = 0.0
    wage_parity_gap: float = 0.0

    # Energy economics
    energy_Wh_per_unit: float = 0.0
    energy_cost_per_unit: float = 0.0
    energy_efficiency_trend: float = 0.0

    # Economic parameters
    price_per_unit: float = 5.0
    damage_cost_per_error: float = 50.0
    energy_price_kWh: float = 0.12

    # Derived objectives
    objective_weights: List[float] = field(default_factory=lambda: [1.0, 0.2, 0.1, 0.05, 0.0])

    # Urgency signals
    mpl_urgency: float = 0.0
    error_urgency: float = 0.0
    energy_urgency: float = 0.0

    # Market context
    customer_segment: str = "balanced"
    market_region: str = "US"
    task_family: str = "dishwashing"

    # Data economics
    rebate_pct: float = 0.0
    attributable_spread_capture: float = 0.0
    data_premium: float = 0.0

    def compute_urgencies(self):
        """Compute urgency signals based on gaps from targets."""
        self.mpl_urgency = max(0.0, 1.0 - (self.current_mpl / self.baseline_mpl_human)) if self.baseline_mpl_human > 0 else 0.0
        self.error_urgency = min(1.0, self.error_rate * 10.0)
        energy_fraction = self.energy_cost_per_unit / self.price_per_unit if self.price_per_unit > 0 else 0
        self.energy_urgency = min(1.0, energy_fraction / 0.2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_mpl": self.current_mpl,
            "baseline_mpl_human": self.baseline_mpl_human,
            "mpl_delta": self.mpl_delta,
            "mpl_trend": self.mpl_trend,
            "error_rate": self.error_rate,
            "error_trend": self.error_trend,
            "damage_cost_total": self.damage_cost_total,
            "implied_wage": self.implied_wage,
            "human_wage": self.human_wage,
            "wage_parity": self.wage_parity,
            "wage_parity_gap": self.wage_parity_gap,
            "energy_Wh_per_unit": self.energy_Wh_per_unit,
            "energy_cost_per_unit": self.energy_cost_per_unit,
            "energy_efficiency_trend": self.energy_efficiency_trend,
            "price_per_unit": self.price_per_unit,
            "damage_cost_per_error": self.damage_cost_per_error,
            "energy_price_kWh": self.energy_price_kWh,
            "objective_weights": self.objective_weights,
            "mpl_urgency": self.mpl_urgency,
            "error_urgency": self.error_urgency,
            "energy_urgency": self.energy_urgency,
            "customer_segment": self.customer_segment,
            "market_region": self.market_region,
            "task_family": self.task_family,
            "rebate_pct": self.rebate_pct,
            "attributable_spread_capture": self.attributable_spread_capture,
            "data_premium": self.data_premium,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EconSignals":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class EconomicController:
    """
    Source of truth for economic signals.

    IMPORTANT HIERARCHY:
    - EconomicController is UPSTREAM (defines physics of value)
    - SemanticOrchestrator is DOWNSTREAM (applies value to meaning)
    - VLA/SIMA/Diffusion/RL are FURTHER DOWNSTREAM (act within meaning)

    This module DOES NOT import SemanticOrchestrator or MetaTransformer.
    """
    def __init__(
        self,
        econ_params: EconParams,
        objective_profile: Optional[ObjectiveProfile] = None,
        energy_response_model: Any = None,
        attribution_config: Optional[Dict[str, Any]] = None,
    ):
        self.econ_params = econ_params
        self.objective_profile = objective_profile
        self.energy_response_model = energy_response_model
        self.attribution_config = attribution_config or {}

    @classmethod
    def from_econ_params(
        cls,
        econ_params: EconParams,
        objective_profile: Optional[ObjectiveProfile] = None,
        energy_response_model: Any = None,
        attribution_config: Optional[Dict[str, Any]] = None,
    ) -> "EconomicController":
        return cls(econ_params, objective_profile, energy_response_model, attribution_config)

    def compute_episode_metrics(
        self,
        summary: EpisodeInfoSummary,
        attribution: Optional[AttributionProfile] = None,
    ) -> Dict[str, float]:
        terms = build_reward_terms(summary, self.econ_params)
        wage_parity = getattr(summary, "wage_parity", None) or 1.0
        rebate_pct = 0.0
        attributable_spread_capture = 0.0
        data_premium = 0.0
        if attribution:
            rebate_pct = attribution.rebate_pct
            attributable_spread_capture = attribution.attributable_spread_capture
            data_premium = attribution.data_premium
        return {
            "mpl_episode": summary.mpl_episode,
            "error_rate_episode": summary.error_rate_episode,
            "energy_Wh": summary.energy_Wh,
            "energy_Wh_per_unit": summary.energy_Wh_per_unit,
            "wage_parity": wage_parity,
            "rebate_pct": rebate_pct,
            "attributable_spread_capture": attributable_spread_capture,
            "data_premium": data_premium,
            "r_mpl": terms["r_mpl"],
            "r_error": terms["r_error"],
            "r_energy": terms["r_energy"],
            "r_safety": terms["r_safety"],
        }

    def estimate_profile_effects(
        self,
        objective_profile: ObjectiveProfile,
        datapacks: Optional[List[DataPackMeta]] = None,
    ) -> Dict[str, float]:
        # Simple aggregation; future: call EnergyResponseNet or solver
        metrics = {
            "mpl": 0.0,
            "error": 0.0,
            "energy_Wh": 0.0,
            "wage_parity": 1.0,
            "rebate_pct": 0.0,
            "attributable_spread_capture": 0.0,
            "data_premium": 0.0,
        }
        if datapacks:
            metrics["mpl"] = float(np.mean([dp.attribution.delta_mpl for dp in datapacks]))
            metrics["error"] = float(np.mean([dp.attribution.delta_error for dp in datapacks]))
            metrics["energy_Wh"] = float(np.mean([dp.energy.total_Wh for dp in datapacks]))
            metrics["rebate_pct"] = float(np.mean([getattr(dp.attribution, "rebate_pct", 0.0) for dp in datapacks]))
            metrics["attributable_spread_capture"] = float(np.mean([getattr(dp.attribution, "attributable_spread_capture", 0.0) for dp in datapacks]))
            metrics["data_premium"] = float(np.mean([getattr(dp.attribution, "data_premium", 0.0) for dp in datapacks]))
        return metrics

    def suggest_objective_and_profiles(
        self,
        constraints: Dict[str, float],
    ) -> Dict[str, Any]:
        # Heuristic suggestion: adjust energy profile based on budget/error constraints
        mpl_min = constraints.get("mpl_min_human")
        energy_budget = constraints.get("energy_budget_Wh")
        error_max = constraints.get("error_max")
        rebate_min = constraints.get("rebate_min_pct", 0.0)

        energy_profile = "BASE"
        if energy_budget is not None and energy_budget < 0.1:
            energy_profile = "SAVER"
        elif error_max is not None and error_max < 0.1:
            energy_profile = "SAFE"
        elif mpl_min is not None and mpl_min > 8000:
            energy_profile = "BOOST"

        data_mix_hint = "real_heavy" if rebate_min > 0.1 else "synthetic_boosted"

        return {
            "objective_preset": "balanced",
            "energy_profile": energy_profile,
            "data_mix_hint": data_mix_hint,
            "expected_rebate_pct": rebate_min,
            "expected_attributable_spread_capture": 0.0,
            "expected_data_premium": 0.0,
        }

    def update_from_semantic_metrics(self, metrics: SemanticMetrics) -> None:
        """Advisory-only: store semantic metrics."""
        self._last_semantic_metrics = metrics

    def suggest_objective_adjustments_from_semantics(
        self,
        metrics: Optional[SemanticMetrics] = None,
    ) -> Dict[str, float]:
        """
        Advisory tweaks to objective weights for orchestrator/meta-transformer.
        """
        if metrics is None:
            metrics = getattr(self, "_last_semantic_metrics", None)
        if metrics is None:
            return {}
        adjustments: Dict[str, float] = {}
        if metrics.econ_relevant_task_fraction < 0.5:
            adjustments["w_novelty"] = -0.05
        if metrics.concept_drift_score > 0.3:
            adjustments["w_energy"] = +0.05
        return adjustments

    # === Pareto tooling (advisory only) ===
    def _dominates(self, a: Dict[str, float], b: Dict[str, float], keys: List[str]) -> bool:
        def val(item, key):
            if key.startswith("-"):
                return -item.get(key[1:], 0.0)
            return item.get(key, 0.0)
        return all(val(a, k) <= val(b, k) for k in keys) and any(val(a, k) < val(b, k) for k in keys)

    def compute_pareto_frontiers(self, datapacks_or_interventions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compute simple Pareto frontiers for MPL vs Energy, MPL vs Error, Energy vs Error, and multi-objective.
        Input items should contain 'mpl', 'energy_Wh', 'error', and optionally 'profile'.
        """
        def build_items():
            items = []
            for x in datapacks_or_interventions:
                if isinstance(x, DataPackMeta):
                    items.append({
                        "mpl": x.attribution.delta_mpl,
                        "energy_Wh": x.energy.total_Wh,
                        "error": x.attribution.delta_error,
                        "profile": getattr(x.condition, "econ_preset", None),
                        "rebate_pct": getattr(x.attribution, "rebate_pct", 0.0),
                        "attributable_spread_capture": getattr(x.attribution, "attributable_spread_capture", 0.0),
                        "data_premium": getattr(x.attribution, "data_premium", 0.0),
                    })
                else:
                    if hasattr(x, "attribution") and hasattr(x, "energy"):
                        items.append({
                            "mpl": getattr(x.attribution, "delta_mpl", 0.0),
                            "energy_Wh": getattr(x.energy, "total_Wh", 0.0),
                            "error": getattr(x.attribution, "delta_error", 0.0),
                            "profile": getattr(x, "profile", None),
                            "rebate_pct": getattr(x.attribution, "rebate_pct", 0.0),
                            "attributable_spread_capture": getattr(x.attribution, "attributable_spread_capture", 0.0),
                            "data_premium": getattr(x.attribution, "data_premium", 0.0),
                        })
                    else:
                        if hasattr(x, "__dict__"):
                            d = x.__dict__
                        elif isinstance(x, dict):
                            d = x
                        else:
                            d = {}
                        items.append({
                            "mpl": d.get("mpl", 0.0),
                            "energy_Wh": d.get("energy_Wh", 0.0),
                            "error": d.get("error", 0.0),
                            "profile": d.get("profile", None),
                            "rebate_pct": d.get("rebate_pct", 0.0),
                            "attributable_spread_capture": d.get("attributable_spread_capture", 0.0),
                            "data_premium": d.get("data_premium", 0.0),
                        })
            return items

        items = build_items()

        def pareto(keys):
            frontier = []
            for a in items:
                if not any(self._dominates(b, a, keys) for b in items):
                    frontier.append(a)
            return frontier

        frontiers = {
            "mpl_vs_energy": pareto(["energy_Wh", "-mpl"]) if items else [],
            "mpl_vs_error": pareto(["error", "-mpl"]) if items else [],
            "energy_vs_error": pareto(["energy_Wh", "error"]) if items else [],
        }
        # Multi-objective using utility with default objective vector
        if items:
            obj_vec = self.objective_profile.objective_vector if self.objective_profile else [1.0, 0.2, 0.1, 0.05, 0.0]
            for it in items:
                it["utility"] = obj_vec[0] * it["mpl"] - obj_vec[1] * it["error"] - obj_vec[2] * it["energy_Wh"]
            frontiers["objective"] = sorted(items, key=lambda x: -x["utility"])
        return frontiers

    def filter_frontier_by_constraints(self, frontier: List[Dict[str, Any]], constraints: Dict[str, float]) -> List[Dict[str, Any]]:
        mp_min = constraints.get("mpl_min_human")
        e_max = constraints.get("energy_budget_Wh")
        err_max = constraints.get("error_max")
        filtered = []
        for pt in frontier:
            if mp_min is not None and pt.get("mpl", 0.0) < mp_min:
                continue
            if e_max is not None and pt.get("energy_Wh", 0.0) > e_max:
                continue
            if err_max is not None and pt.get("error", 0.0) > err_max:
                continue
            filtered.append(pt)
        return filtered

    def evaluate_frontier_for_spread(self, frontier: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        evaluated = []
        for pt in frontier:
            evaluated.append({
                **pt,
                "rebate_pct": pt.get("rebate_pct", 0.0),
                "attributable_spread_capture": pt.get("attributable_spread_capture", 0.0),
                "data_premium": pt.get("data_premium", 0.0),
            })
        return evaluated

    def select_frontier_optimum(self, frontier: List[Dict[str, Any]], objective_vector: List[float]) -> Optional[Dict[str, Any]]:
        if not frontier:
            return None
        best = None
        best_u = -np.inf
        for pt in frontier:
            u = objective_vector[0] * pt.get("mpl", 0.0) - objective_vector[1] * pt.get("error", 0.0) - objective_vector[2] * pt.get("energy_Wh", 0.0)
            if u > best_u:
                best_u = u
                best = pt
        return best

    def compute_signals(
        self,
        datapacks: List[DataPackMeta],
        episodes: Optional[List[EpisodeInfoSummary]] = None,
    ) -> EconSignals:
        """
        Compute economic signals from datapacks and episodes.

        PRIMARY entry point for downstream modules (SemanticOrchestrator, VLA, etc.)
        to get current economic state.

        Args:
            datapacks: List of DataPackMeta objects
            episodes: Optional list of EpisodeInfoSummary objects

        Returns:
            EconSignals with computed metrics
        """
        signals = EconSignals(
            baseline_mpl_human=self.econ_params.mpl_human if hasattr(self.econ_params, 'mpl_human') else 60.0,
            human_wage=self.econ_params.wage_human if hasattr(self.econ_params, 'wage_human') else 18.0,
            price_per_unit=self.econ_params.price_per_unit,
            damage_cost_per_error=self.econ_params.damage_cost,
            energy_price_kWh=getattr(self.econ_params, 'energy_price_kWh', 0.12),
            customer_segment=getattr(self.econ_params, 'customer_segment', 'balanced'),
            market_region=getattr(self.econ_params, 'market_region', 'US'),
            task_family=getattr(self.econ_params, 'task_family', 'dishwashing'),
        )

        if self.objective_profile:
            signals.objective_weights = self.objective_profile.objective_vector

        if not datapacks:
            signals.compute_urgencies()
            return signals

        # Extract metrics from datapacks
        mpls = []
        error_rates = []
        energy_per_units = []
        damage_costs = []
        rebates = []
        spreads = []
        premiums = []

        for dp in datapacks:
            mpls.append(dp.attribution.delta_mpl)
            error_rates.append(dp.attribution.delta_error)
            energy_per_units.append(dp.energy.Wh_per_unit)
            rebates.append(getattr(dp.attribution, "rebate_pct", 0.0))
            spreads.append(getattr(dp.attribution, "attributable_spread_capture", 0.0))
            premiums.append(getattr(dp.attribution, "data_premium", 0.0))

            # Damage costs from episode_metrics if available
            if dp.episode_metrics:
                damage_costs.append(
                    dp.episode_metrics.get('damage_cost', 0.0) or
                    dp.episode_metrics.get('vase_breaks', 0) * self.econ_params.damage_cost
                )

        # Compute current metrics
        if mpls:
            signals.current_mpl = float(np.mean(mpls))
            signals.mpl_delta = float(np.max(mpls) - np.min(mpls)) if len(mpls) > 1 else 0.0

        if error_rates:
            signals.error_rate = float(np.mean(error_rates))

        if energy_per_units:
            signals.energy_Wh_per_unit = float(np.mean(energy_per_units))
            signals.energy_cost_per_unit = signals.energy_Wh_per_unit * signals.energy_price_kWh / 1000.0

        if damage_costs:
            signals.damage_cost_total = float(np.sum(damage_costs))

        # Data economics
        signals.rebate_pct = float(np.mean(rebates)) if rebates else 0.0
        signals.attributable_spread_capture = float(np.mean(spreads)) if spreads else 0.0
        signals.data_premium = float(np.mean(premiums)) if premiums else 0.0

        # Compute implied wage
        signals.implied_wage = (
            signals.price_per_unit * signals.current_mpl -
            signals.error_rate * signals.current_mpl * signals.damage_cost_per_error
        )

        # Compute wage parity
        if signals.human_wage > 0:
            signals.wage_parity = signals.implied_wage / signals.human_wage
            signals.wage_parity_gap = 1.0 - signals.wage_parity

        # Compute urgencies
        signals.compute_urgencies()

        return signals

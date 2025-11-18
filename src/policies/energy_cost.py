"""
Heuristic EnergyCostPolicy aligned with RewardEngine aggregation.

Accumulates energy_penalty components (if present) without changing rewards.
"""
from typing import Any, Dict, Sequence

from src.policies.interfaces import EnergyCostPolicy
from src.utils.json_safe import to_json_safe


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _components(event: Any) -> Dict[str, Any]:
    if hasattr(event, "reward_components"):
        comps = getattr(event, "reward_components", {}) or {}
    elif isinstance(event, dict):
        comps = event.get("reward_components", {}) or {}
    else:
        comps = {}
    return comps if isinstance(comps, dict) else {}


class HeuristicEnergyCostPolicy(EnergyCostPolicy):
    def build_features(self, events: Sequence[Any]) -> Dict[str, Any]:
        energy_terms = []
        for ev in events or []:
            comps = _components(ev)
            energy_terms.append(_safe_float(comps.get("energy_penalty", comps.get("energy", 0.0))))
        return {"energy_terms": energy_terms}

    def evaluate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        terms = [_safe_float(v) for v in features.get("energy_terms", [])]
        energy_cost = sum(terms)
        metadata = {"terms": terms, "num_events": len(terms)}
        return {"energy_cost": energy_cost, "metadata": to_json_safe(metadata)}

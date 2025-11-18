"""
Heuristic SafetyRiskPolicy that mirrors existing reward component usage.

Summarizes collision/damage components without altering scalar rewards.
"""
from typing import Any, Dict, Sequence

from src.policies.interfaces import SafetyRiskPolicy
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


class HeuristicSafetyRiskPolicy(SafetyRiskPolicy):
    def build_features(self, events: Sequence[Any]) -> Dict[str, Any]:
        damage_terms = []
        collision_counts = 0
        for ev in events or []:
            comps = _components(ev)
            dmg = _safe_float(comps.get("collision_penalty", comps.get("damage_cost", 0.0)))
            damage_terms.append(dmg)
            if dmg != 0.0:
                collision_counts += 1
        return {
            "damage_terms": damage_terms,
            "collision_counts": collision_counts,
        }

    def evaluate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        damage_terms = [ _safe_float(v) for v in features.get("damage_terms", []) ]
        total_damage = sum(damage_terms)
        collision_counts = int(_safe_float(features.get("collision_counts", 0), 0.0))
        risk_level = "low"
        if total_damage < -1.0:
            risk_level = "medium"
        if total_damage < -5.0 or collision_counts > 2:
            risk_level = "high"
        metadata = {
            "total_damage_cost": total_damage,
            "collision_counts": collision_counts,
        }
        return {
            "risk_level": risk_level,
            "damage_estimate": total_damage,
            "metadata": to_json_safe(metadata),
        }

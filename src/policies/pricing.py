"""
Heuristic PricingPolicy wrapper that mirrors compute_pricing_snapshot logic.

Purely advisory and deterministic; does not alter any reward/econ math.
"""
from typing import Any, Dict, Optional

from src.economics.pricing import compute_consumer_surplus
from src.policies.interfaces import PricingPolicy
from src.utils.json_safe import to_json_safe


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


class HeuristicPricingPolicy(PricingPolicy):
    def __init__(self, price_floor_multiplier: float = 0.1):
        self.price_floor_multiplier = float(price_floor_multiplier)

    def build_features(
        self,
        task_econ: Dict[str, Any],
        datapack_value: Optional[float] = None,
        semantic_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        task = task_econ.get("task", {}) if isinstance(task_econ, dict) else {}
        mpl_stats = task_econ.get("mpl", {}) if isinstance(task_econ, dict) else {}
        wage_stats = task_econ.get("wage_parity", {}) if isinstance(task_econ, dict) else {}
        features = {
            "human_mpl_units_per_hour": _safe_float(task.get("human_mpl_units_per_hour")),
            "human_wage_per_hour": _safe_float(task.get("human_wage_per_hour")),
            "robot_mpl_units_per_hour": _safe_float(mpl_stats.get("mean")),
            "robot_wage_parity": _safe_float(wage_stats.get("mean"), 1.0),
            "datapack_value": _safe_float(datapack_value),
            "semantic_context": to_json_safe(semantic_context or {}),
        }
        return features

    def evaluate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        human_mpl = _safe_float(features.get("human_mpl_units_per_hour"))
        human_wage = _safe_float(features.get("human_wage_per_hour"))
        robot_mpl = _safe_float(features.get("robot_mpl_units_per_hour"))
        wage_parity = _safe_float(features.get("robot_wage_parity"), 1.0)

        human_unit_cost = human_wage / max(human_mpl, 1e-6) if human_mpl else 0.0
        robot_unit_cost = (human_wage * wage_parity) / max(robot_mpl, 1e-6) if robot_mpl else 0.0
        spread = human_unit_cost - robot_unit_cost if human_unit_cost and robot_unit_cost else 0.0
        datapack_price_floor = max(0.0, self.price_floor_multiplier * spread)

        unit_price = robot_unit_cost
        robot_hour_price = robot_unit_cost * robot_mpl if robot_mpl else 0.0
        consumer_surplus = compute_consumer_surplus(human_unit_cost, unit_price) if human_unit_cost else 0.0

        metadata = {
            "human_unit_cost": human_unit_cost,
            "robot_unit_cost": robot_unit_cost,
            "spread_per_unit": spread,
            "datapack_price_floor": datapack_price_floor,
            "datapack_value": _safe_float(features.get("datapack_value")),
            "semantic_context": features.get("semantic_context", {}),
        }
        return {
            "unit_price": unit_price,
            "robot_hour_price": robot_hour_price,
            "consumer_surplus": consumer_surplus,
            "metadata": to_json_safe(metadata),
        }

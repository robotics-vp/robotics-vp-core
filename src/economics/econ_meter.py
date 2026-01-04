from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.economics.wage import implied_robot_wage
from src.ontology.models import EconVector, Robot, Task


@dataclass
class EconomicMeter:
    task: Task
    robot: Robot | None = None
    config: Mapping[str, Any] | None = None

    def summarize(self, raw_metrics: Mapping[str, Any]) -> Mapping[str, float]:
        config = dict(self.config or {})
        task = self.task
        energy_cost = self._extract_metric(raw_metrics, ("energy_cost",))
        energy_wh = self._extract_metric(raw_metrics, ("energy_wh", "energy_Wh", "energy_wh_total"))
        energy_kwh = self._extract_metric(raw_metrics, ("energy_kwh", "energy_kWh"))
        if energy_cost == 0.0:
            if energy_wh == 0.0 and energy_kwh:
                energy_wh = energy_kwh * 1000.0
            cost_per_wh = self._energy_cost_per_wh(task)
            if energy_wh and cost_per_wh:
                energy_cost = energy_wh * cost_per_wh

        mpl_units_per_hour = self._extract_metric(
            raw_metrics,
            (
                "mpl_units_per_hour",
                "mpl_per_hour",
                "mpl",
                "throughput_per_hour",
                "throughput",
            ),
        )
        if mpl_units_per_hour == 0.0:
            mpl_units_per_hour = self._estimate_mpl_from_duration(raw_metrics, config)
        error_rate = self._extract_metric(raw_metrics, ("error_rate", "errors", "error_fraction"))
        if error_rate == 0.0 and raw_metrics.get("success_rate") is not None:
            try:
                error_rate = 1.0 - float(raw_metrics.get("success_rate"))
            except (TypeError, ValueError):
                pass
        novelty_delta = self._extract_metric(raw_metrics, ("novelty_delta", "novelty_score", "novelty"))
        reward_scalar_sum = self._extract_metric(
            raw_metrics,
            ("reward_scalar_sum", "episode_reward", "reward_sum", "total_reward", "mean_reward"),
        )
        damage_cost = self._extract_metric(raw_metrics, ("damage_cost",))
        if damage_cost == 0.0:
            damage_per_error = float(config.get("damage_cost_per_error", 0.0))
            if damage_per_error and error_rate and mpl_units_per_hour:
                damage_cost = damage_per_error * (error_rate * mpl_units_per_hour)

        wage_parity = self._extract_metric(raw_metrics, ("wage_parity",))
        if wage_parity == 0.0:
            wage_parity = self._compute_wage_parity(
                mpl_units_per_hour=mpl_units_per_hour,
                error_rate=error_rate,
                damage_cost_per_error=float(config.get("damage_cost_per_error", 0.0)),
            )

        return {
            "mpl_units_per_hour": mpl_units_per_hour,
            "wage_parity": wage_parity,
            "energy_cost": energy_cost,
            "damage_cost": damage_cost,
            "novelty_delta": novelty_delta,
            "reward_scalar_sum": reward_scalar_sum,
            "error_rate": error_rate,
        }

    def to_econ_vector(
        self,
        episode_id: str,
        raw_metrics: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        source_domain: str = "holosoma",
    ) -> EconVector:
        econ = self.summarize(raw_metrics)
        components = self._numeric_components(raw_metrics)
        return EconVector(
            episode_id=episode_id,
            mpl_units_per_hour=econ["mpl_units_per_hour"],
            wage_parity=econ["wage_parity"],
            energy_cost=econ["energy_cost"],
            damage_cost=econ["damage_cost"],
            novelty_delta=econ["novelty_delta"],
            reward_scalar_sum=econ["reward_scalar_sum"],
            components=components,
            source_domain=source_domain,
            metadata=dict(metadata or {}),
        )

    def _energy_cost_per_wh(self, task: Task) -> float:
        if self.robot and self.robot.energy_cost_per_wh is not None:
            return float(self.robot.energy_cost_per_wh)
        try:
            return float(task.default_energy_cost_per_wh)
        except Exception:
            return 0.0

    def _compute_wage_parity(self, mpl_units_per_hour: float, error_rate: float, damage_cost_per_error: float) -> float:
        task = self.task
        if mpl_units_per_hour <= 0.0:
            return 0.0
        if task.human_mpl_units_per_hour > 0 and task.human_wage_per_hour <= 0:
            return mpl_units_per_hour / task.human_mpl_units_per_hour
        price_per_unit = self._price_per_unit(task)
        if price_per_unit <= 0:
            return mpl_units_per_hour / task.human_mpl_units_per_hour if task.human_mpl_units_per_hour > 0 else 0.0
        implied = implied_robot_wage(price_per_unit, mpl_units_per_hour, error_rate, damage_cost_per_error)
        if task.human_wage_per_hour <= 0:
            return 0.0
        return implied / task.human_wage_per_hour

    def _price_per_unit(self, task: Task) -> float:
        if task.human_mpl_units_per_hour <= 0:
            return 0.0
        return task.human_wage_per_hour / task.human_mpl_units_per_hour

    def _estimate_mpl_from_duration(self, raw_metrics: Mapping[str, Any], config: Mapping[str, Any]) -> float:
        duration_s = self._extract_metric(raw_metrics, ("mean_episode_length_s", "episode_length_s"))
        if duration_s <= 0.0:
            return 0.0
        units_per_episode = float(config.get("units_per_episode", 1.0))
        success_rate = self._extract_metric(raw_metrics, ("success_rate",))
        if success_rate <= 0.0:
            success_rate = 1.0
        return (units_per_episode * success_rate) * (3600.0 / duration_s)

    @staticmethod
    def _extract_metric(raw_metrics: Mapping[str, Any], keys: tuple[str, ...]) -> float:
        for key in keys:
            if key in raw_metrics and raw_metrics[key] is not None:
                try:
                    return float(raw_metrics[key])
                except (TypeError, ValueError):
                    continue
        return 0.0

    @staticmethod
    def _numeric_components(raw_metrics: Mapping[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in raw_metrics.items():
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return out

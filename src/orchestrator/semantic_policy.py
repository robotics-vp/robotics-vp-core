from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.motor_backend.datapacks import DatapackConfig
from src.scenarios.metadata import ScenarioMetadata


def apply_arh_penalty(
    econ_metrics: Mapping[str, float],
    *,
    suspicious_key: str = "anti_reward_hacking_suspicious",
    penalty_factor: float = 0.5,
) -> dict[str, float]:
    metrics = dict(econ_metrics)
    suspicious = metrics.get(suspicious_key, 0.0)
    try:
        suspicious_val = float(suspicious)
    except (TypeError, ValueError):
        suspicious_val = 1.0 if suspicious else 0.0
    if suspicious_val <= 0.0:
        return metrics

    mpl = metrics.get("mpl_units_per_hour", 0.0)
    try:
        mpl_val = float(mpl)
    except (TypeError, ValueError):
        mpl_val = 0.0
    adjusted = mpl_val * max(0.0, min(1.0, 1.0 - penalty_factor))
    metrics["mpl_units_per_hour_adjusted"] = adjusted
    metrics["anti_reward_hacking_penalty"] = penalty_factor
    return metrics


@dataclass(frozen=True)
class MissingScenarioSpec:
    tags: Sequence[str]
    robot_family: str | None
    objective_hint: str | None = None


def select_datapacks_for_intent(
    tags: Sequence[str],
    robot_family: str | None,
    objective_hint: str | None,
    candidates: Sequence[DatapackConfig],
    scenarios: Sequence[ScenarioMetadata | Mapping[str, Any]],
) -> list[DatapackConfig]:
    if not candidates:
        return []

    required_tags = {t.strip().lower() for t in tags if t and str(t).strip()}
    robot_norm = robot_family.strip().lower() if robot_family else None
    objective_norm = objective_hint.strip().lower() if objective_hint else None

    arh_by_datapack = _arh_flags_by_datapack(scenarios)
    scored: list[tuple[float, DatapackConfig]] = []
    for cfg in candidates:
        cfg_tags = {t.lower() for t in cfg.tags}
        if required_tags and not required_tags.issubset(cfg_tags):
            continue
        if robot_norm and cfg.robot_families:
            if robot_norm not in {t.lower() for t in cfg.robot_families}:
                continue

        score = float(len(required_tags & cfg_tags))
        if objective_norm and cfg.objective_hint:
            if objective_norm in cfg.objective_hint.lower():
                score += 1.0

        arh_penalty = max(arh_by_datapack.get(cfg.id, 0.0), 0.0)
        score -= arh_penalty * 2.0
        scored.append((score, cfg))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [cfg for _, cfg in scored]


def detect_semantic_gaps(
    tags: Sequence[str],
    robot_family: str | None,
    scenarios: Sequence[ScenarioMetadata | Mapping[str, Any]],
) -> list[MissingScenarioSpec]:
    required_tags = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
    if not required_tags:
        return []
    robot_norm = robot_family.strip().lower() if robot_family else None
    observed: set[str] = set()
    for scenario in scenarios:
        scenario_tags = _scenario_tags(scenario)
        if robot_norm:
            families = _scenario_robot_families(scenario)
            if robot_norm not in families:
                continue
        observed |= scenario_tags
    missing = required_tags - observed
    if not missing:
        return []
    return [MissingScenarioSpec(tags=sorted(missing), robot_family=robot_family)]


def _scenario_tags(scenario: ScenarioMetadata | Mapping[str, Any]) -> set[str]:
    if isinstance(scenario, ScenarioMetadata):
        return {t.lower() for t in scenario.datapack_tags}
    return {str(t).strip().lower() for t in scenario.get("datapack_tags") or [] if str(t).strip()}


def _scenario_robot_families(scenario: ScenarioMetadata | Mapping[str, Any]) -> set[str]:
    if isinstance(scenario, ScenarioMetadata):
        return {t.lower() for t in scenario.robot_families}
    return {str(t).strip().lower() for t in scenario.get("robot_families") or [] if str(t).strip()}


def _arh_flags_by_datapack(
    scenarios: Sequence[ScenarioMetadata | Mapping[str, Any]],
) -> dict[str, float]:
    flags: dict[str, float] = {}
    for scenario in scenarios:
        if isinstance(scenario, ScenarioMetadata):
            continue
        datapack_ids = scenario.get("datapack_ids") or []
        arh_flag = _extract_arh_flag(scenario)
        if arh_flag <= 0.0:
            continue
        for dp_id in datapack_ids:
            if not dp_id:
                continue
            flags[dp_id] = max(flags.get(dp_id, 0.0), arh_flag)
    return flags


def _extract_arh_flag(scenario: Mapping[str, Any]) -> float:
    for key in (
        "train_metrics_anti_reward_hacking_suspicious",
        "eval_metrics_anti_reward_hacking_suspicious",
    ):
        if key in scenario:
            return _safe_float(scenario.get(key))
    train = scenario.get("train_metrics")
    if isinstance(train, Mapping):
        return _safe_float(train.get("anti_reward_hacking_suspicious"))
    return 0.0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

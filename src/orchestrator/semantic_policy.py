from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.evidence.benchmark_gating import collect_benchmark_gating_signals

from src.economics.arh_config import current_arh_config
from src.motor_backend.datapacks import DatapackConfig
from src.scenarios.metadata import ScenarioMetadata


def apply_arh_penalty(
    econ_metrics: Mapping[str, float],
    *,
    suspicious_key: str = "anti_reward_hacking_suspicious",
    penalty_factor: float | None = None,
    hard_exclusion_threshold: float | None = None,
) -> dict[str, float]:
    if penalty_factor is None or hard_exclusion_threshold is None:
        cfg = current_arh_config()
        if penalty_factor is None:
            penalty_factor = cfg.suspicious_penalty_factor
        if hard_exclusion_threshold is None:
            hard_exclusion_threshold = cfg.hard_exclusion_threshold
    metrics = dict(econ_metrics)
    arh_score = _extract_arh_score(metrics)
    if hard_exclusion_threshold is not None and arh_score is not None and arh_score > hard_exclusion_threshold:
        metrics["mpl_units_per_hour_adjusted"] = 0.0
        metrics["arh_excluded"] = 1.0
        return metrics

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
    adjusted = mpl_val * max(0.0, min(1.0, 1.0 - (penalty_factor or 0.0)))
    metrics["mpl_units_per_hour_adjusted"] = adjusted
    if penalty_factor is not None:
        metrics["anti_reward_hacking_penalty"] = penalty_factor
    return metrics


@dataclass(frozen=True)
class MissingScenarioSpec:
    tags: Sequence[str]
    robot_family: str | None
    objective_hint: str | None = None


@dataclass(frozen=True)
class DatapackSelectionDecision:
    datapack: DatapackConfig
    score: float
    source: str
    matched_tags: Sequence[str] = field(default_factory=tuple)
    missing_tags: Sequence[str] = field(default_factory=tuple)
    gap_fill_tags: Sequence[str] = field(default_factory=tuple)
    exact_tag_match: bool = False
    objective_match: bool = False
    robot_match: bool = True
    historical_support: Mapping[str, Any] = field(default_factory=dict)
    benchmark_support: Mapping[str, Any] = field(default_factory=dict)
    candidate_metadata: Mapping[str, Any] = field(default_factory=dict)
    reasons: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "datapack_id": self.datapack.id,
            "score": float(self.score),
            "source": self.source,
            "matched_tags": list(self.matched_tags),
            "missing_tags": list(self.missing_tags),
            "gap_fill_tags": list(self.gap_fill_tags),
            "exact_tag_match": bool(self.exact_tag_match),
            "objective_match": bool(self.objective_match),
            "robot_match": bool(self.robot_match),
            "historical_support": dict(self.historical_support),
            "benchmark_support": dict(self.benchmark_support),
            "candidate_metadata": dict(self.candidate_metadata),
            "reasons": list(self.reasons),
        }


def rank_datapacks_for_intent(
    tags: Sequence[str],
    robot_family: str | None,
    objective_hint: str | None,
    candidates: Sequence[DatapackConfig],
    scenarios: Sequence[ScenarioMetadata | Mapping[str, Any]],
    *,
    candidate_metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    source: str = "ontology",
) -> list[DatapackSelectionDecision]:
    if not candidates:
        return []

    required_tags = {t.strip().lower() for t in tags if t and str(t).strip()}
    robot_norm = robot_family.strip().lower() if robot_family else None
    objective_norm = objective_hint.strip().lower() if objective_hint else None
    observed_tags = _observed_tags_for_robot(scenarios, robot_family=robot_family)
    gap_tags = required_tags - observed_tags
    scenario_support = _scenario_support_by_datapack(scenarios)
    metadata_by_id = {
        str(key): dict(value or {})
        for key, value in dict(candidate_metadata_by_id or {}).items()
    }

    decisions: list[DatapackSelectionDecision] = []
    for cfg in candidates:
        cfg_tags = {t.lower() for t in cfg.tags}
        matched_tags = sorted(required_tags & cfg_tags)
        if required_tags and not matched_tags:
            continue

        robot_match = True
        if robot_norm and cfg.robot_families:
            robot_match = robot_norm in {t.lower() for t in cfg.robot_families}
            if not robot_match:
                continue

        candidate_metadata = dict(metadata_by_id.get(cfg.id, {}) or {})
        benchmark_support = _candidate_benchmark_support(candidate_metadata)
        execution_ready = bool(benchmark_support.get("execution_ready", False))
        quality_score = _clamp01(_safe_float(candidate_metadata.get("quality_score", 0.0)))
        novelty_score = _clamp01(_safe_float(candidate_metadata.get("novelty_score", 0.0)))
        objective_match = bool(
            objective_norm and cfg.objective_hint and objective_norm in cfg.objective_hint.lower()
        )
        exact_tag_match = not required_tags or required_tags.issubset(cfg_tags)
        gap_fill_tags = sorted(gap_tags & cfg_tags)

        history = dict(
            scenario_support.get(
                cfg.id,
                {
                    "scenario_count": 0,
                    "eval_count": 0,
                    "support_score": 0.0,
                    "max_arh_penalty": 0.0,
                    "mean_adjusted_mpl": 0.0,
                    "mean_reward": 0.0,
                },
            )
        )
        tag_coverage = (len(matched_tags) / float(len(required_tags))) if required_tags else 1.0
        gap_fill_score = (len(gap_fill_tags) / float(len(gap_tags))) if gap_tags else 0.0
        score = (
            2.5 * tag_coverage
            + (0.8 if exact_tag_match else 0.0)
            + 0.9 * gap_fill_score
            + (0.55 if objective_match else 0.0)
            + 0.65 * _safe_float(history.get("support_score", 0.0))
            + 0.35 * quality_score
            + 0.2 * novelty_score
            + (0.35 if benchmark_support.get("semantic_grounding_non_heuristic", False) else 0.0)
            + (0.2 if benchmark_support.get("benchmark_eligible", False) else 0.0)
            + (0.15 if execution_ready else 0.0)
            + (0.1 if history.get("scenario_count", 0) == 0 and exact_tag_match else 0.0)
            - 2.0 * _safe_float(history.get("max_arh_penalty", 0.0))
        )
        reasons: list[str] = [
            f"tag_coverage:{tag_coverage:.2f}",
            f"history_support:{_safe_float(history.get('support_score', 0.0)):.2f}",
        ]
        if exact_tag_match:
            reasons.append("exact_tag_match")
        if gap_fill_tags:
            reasons.append(f"gap_fill:{','.join(gap_fill_tags)}")
        if objective_match:
            reasons.append("objective_match")
        if benchmark_support.get("semantic_grounding_non_heuristic", False):
            reasons.append("non_heuristic_grounding")
        if benchmark_support.get("benchmark_eligible", False):
            reasons.append("benchmark_eligible")
        if execution_ready:
            reasons.append("execution_ready")
        if _safe_float(history.get("max_arh_penalty", 0.0)) > 0.0:
            reasons.append(f"arh_penalty:{_safe_float(history.get('max_arh_penalty', 0.0)):.2f}")

        decisions.append(
            DatapackSelectionDecision(
                datapack=cfg,
                score=float(score),
                source=source,
                matched_tags=matched_tags,
                missing_tags=sorted(required_tags - cfg_tags),
                gap_fill_tags=gap_fill_tags,
                exact_tag_match=exact_tag_match,
                objective_match=objective_match,
                robot_match=robot_match,
                historical_support=history,
                benchmark_support=benchmark_support,
                candidate_metadata=candidate_metadata,
                reasons=reasons,
            )
        )

    decisions.sort(
        key=lambda item: (
            item.score,
            bool(item.benchmark_support.get("benchmark_eligible", False)),
            bool(item.exact_tag_match),
            _safe_float(item.candidate_metadata.get("quality_score", 0.0)),
            item.datapack.id,
        ),
        reverse=True,
    )
    return decisions


def summarize_datapack_selection(
    ranked: Sequence[DatapackSelectionDecision],
    *,
    selected: Sequence[DatapackSelectionDecision] = (),
    tags: Sequence[str] = (),
    robot_family: str | None = None,
    objective_hint: str | None = None,
) -> dict[str, Any]:
    selected_rows = list(selected or ranked)
    return {
        "required_tags": sorted({str(tag).strip().lower() for tag in tags if str(tag).strip()}),
        "robot_family": robot_family,
        "objective_hint": objective_hint,
        "candidate_count": len(ranked),
        "selected_ids": [row.datapack.id for row in selected_rows],
        "selected_sources": sorted({row.source for row in selected_rows}),
        "selected_gap_fill_tags": sorted(
            {tag for row in selected_rows for tag in row.gap_fill_tags}
        ),
        "top_candidates": [row.to_dict() for row in list(ranked)[:5]],
    }


def select_datapacks_for_intent(
    tags: Sequence[str],
    robot_family: str | None,
    objective_hint: str | None,
    candidates: Sequence[DatapackConfig],
    scenarios: Sequence[ScenarioMetadata | Mapping[str, Any]],
    *,
    candidate_metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    source: str = "ontology",
) -> list[DatapackConfig]:
    ranked = rank_datapacks_for_intent(
        tags,
        robot_family,
        objective_hint,
        candidates,
        scenarios,
        candidate_metadata_by_id=candidate_metadata_by_id,
        source=source,
    )
    return [row.datapack for row in ranked]


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


def _scenario_support_by_datapack(
    scenarios: Sequence[ScenarioMetadata | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    support: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        if isinstance(scenario, ScenarioMetadata):
            continue
        datapack_ids = [str(dp_id) for dp_id in scenario.get("datapack_ids") or [] if str(dp_id)]
        if not datapack_ids:
            continue
        adjusted_train = _adjusted_metrics(scenario.get("train_metrics"))
        adjusted_eval = _adjusted_metrics(scenario.get("eval_metrics"))
        adjusted_mpl = max(
            _safe_float(adjusted_eval.get("mpl_units_per_hour_adjusted", adjusted_eval.get("mpl_units_per_hour", 0.0))),
            _safe_float(adjusted_train.get("mpl_units_per_hour_adjusted", adjusted_train.get("mpl_units_per_hour", 0.0))),
        )
        reward = max(
            _safe_float(adjusted_eval.get("reward_scalar_sum", 0.0)),
            _safe_float(adjusted_train.get("reward_scalar_sum", 0.0)),
        )
        arh_penalty = max(_extract_arh_flag(scenario), 0.0)
        for dp_id in datapack_ids:
            row = support.setdefault(
                dp_id,
                {
                    "scenario_count": 0,
                    "eval_count": 0,
                    "adjusted_mpl_samples": [],
                    "reward_samples": [],
                    "max_arh_penalty": 0.0,
                },
            )
            row["scenario_count"] += 1
            if adjusted_eval:
                row["eval_count"] += 1
            row["adjusted_mpl_samples"].append(adjusted_mpl)
            row["reward_samples"].append(reward)
            row["max_arh_penalty"] = max(_safe_float(row.get("max_arh_penalty", 0.0)), arh_penalty)

    finalized: dict[str, dict[str, Any]] = {}
    for datapack_id, row in support.items():
        mpl_values = list(row.get("adjusted_mpl_samples", []) or [])
        reward_values = list(row.get("reward_samples", []) or [])
        mean_adjusted_mpl = sum(mpl_values) / float(max(len(mpl_values), 1))
        mean_reward = sum(reward_values) / float(max(len(reward_values), 1))
        support_score = _clamp01((mean_adjusted_mpl / 100.0) * 0.7 + (mean_reward / 10.0) * 0.3)
        finalized[datapack_id] = {
            "scenario_count": int(row.get("scenario_count", 0)),
            "eval_count": int(row.get("eval_count", 0)),
            "mean_adjusted_mpl": float(mean_adjusted_mpl),
            "mean_reward": float(mean_reward),
            "support_score": float(support_score),
            "max_arh_penalty": float(row.get("max_arh_penalty", 0.0)),
        }
    return finalized


def _adjusted_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    return apply_arh_penalty({str(key): value for key, value in payload.items()})


def _observed_tags_for_robot(
    scenarios: Sequence[ScenarioMetadata | Mapping[str, Any]],
    *,
    robot_family: str | None,
) -> set[str]:
    robot_norm = robot_family.strip().lower() if robot_family else None
    observed: set[str] = set()
    for scenario in scenarios:
        if robot_norm:
            families = _scenario_robot_families(scenario)
            if robot_norm not in families:
                continue
        observed |= _scenario_tags(scenario)
    return observed


def _candidate_benchmark_support(candidate_metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(candidate_metadata or {})
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        benchmark_payload = dict(metadata)
    else:
        benchmark_payload = {}
    future_training_signals = benchmark_payload.get("future_training_signals")
    if isinstance(future_training_signals, Mapping):
        benchmark_payload.update(dict(future_training_signals))
    benchmark_payload.update(
        {
            key: value
            for key, value in payload.items()
            if key
            in {
                "scene_tracks_backend",
                "teacher_runtime_backend_selected",
                "vision_backbone_selected",
                "semantic_grounding_mode",
                "semantic_memory_grounded",
                "grounded_track_object_count",
            }
        }
    )
    signals = collect_benchmark_gating_signals(benchmark_payload)
    execution_preconditions = benchmark_payload.get("execution_preconditions")
    if not isinstance(execution_preconditions, Mapping):
        execution_preconditions = payload.get("execution_preconditions")
    signals["execution_ready"] = bool(
        isinstance(execution_preconditions, Mapping) and execution_preconditions.get("ready", False)
    )
    return signals


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def _extract_arh_score(metrics: Mapping[str, Any]) -> float | None:
    for key in ("anti_reward_hacking_score", "arh_score", "reward_hacking_score"):
        if key in metrics and metrics[key] is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                return None
    return None


__all__ = [
    "DatapackSelectionDecision",
    "MissingScenarioSpec",
    "apply_arh_penalty",
    "detect_semantic_gaps",
    "rank_datapacks_for_intent",
    "select_datapacks_for_intent",
    "summarize_datapack_selection",
]

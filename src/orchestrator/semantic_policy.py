from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

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
    if (
        hard_exclusion_threshold is not None
        and arh_score is not None
        and arh_score > hard_exclusion_threshold
    ):
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
class DatapackSelectionFeatures:
    tag_coverage: float = 0.0
    exact_tag_match: float = 0.0
    gap_fill_score: float = 0.0
    objective_match: float = 0.0
    history_support_score: float = 0.0
    quality_score: float = 0.0
    novelty_score: float = 0.0
    semantic_grounding_non_heuristic: float = 0.0
    benchmark_eligible: float = 0.0
    execution_ready: float = 0.0
    cold_start_bonus: float = 0.0
    max_arh_penalty: float = 0.0
    mean_adjusted_mpl_norm: float = 0.0
    mean_reward_norm: float = 0.0
    scenario_count_norm: float = 0.0
    eval_count_norm: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "tag_coverage": float(self.tag_coverage),
            "exact_tag_match": float(self.exact_tag_match),
            "gap_fill_score": float(self.gap_fill_score),
            "objective_match": float(self.objective_match),
            "history_support_score": float(self.history_support_score),
            "quality_score": float(self.quality_score),
            "novelty_score": float(self.novelty_score),
            "semantic_grounding_non_heuristic": float(
                self.semantic_grounding_non_heuristic
            ),
            "benchmark_eligible": float(self.benchmark_eligible),
            "execution_ready": float(self.execution_ready),
            "cold_start_bonus": float(self.cold_start_bonus),
            "max_arh_penalty": float(self.max_arh_penalty),
            "mean_adjusted_mpl_norm": float(self.mean_adjusted_mpl_norm),
            "mean_reward_norm": float(self.mean_reward_norm),
            "scenario_count_norm": float(self.scenario_count_norm),
            "eval_count_norm": float(self.eval_count_norm),
        }


@dataclass(frozen=True)
class DatapackSelectionContext:
    required_tag_count_norm: float = 0.0
    gap_pressure: float = 0.0
    candidate_pool_size_norm: float = 0.0
    benchmark_ready_ratio: float = 0.0
    execution_ready_ratio: float = 0.0
    history_density: float = 0.0
    cold_start_pressure: float = 0.0
    objective_present: float = 0.0
    robot_specificity: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "required_tag_count_norm": float(self.required_tag_count_norm),
            "gap_pressure": float(self.gap_pressure),
            "candidate_pool_size_norm": float(self.candidate_pool_size_norm),
            "benchmark_ready_ratio": float(self.benchmark_ready_ratio),
            "execution_ready_ratio": float(self.execution_ready_ratio),
            "history_density": float(self.history_density),
            "cold_start_pressure": float(self.cold_start_pressure),
            "objective_present": float(self.objective_present),
            "robot_specificity": float(self.robot_specificity),
        }


@dataclass(frozen=True)
class DatapackSelectionScorerPackage:
    package_id: str
    schema_version: str
    feature_weights: Mapping[str, float]
    context_weights: Mapping[str, float] = field(default_factory=dict)
    bias: float = 0.0
    context_bias: float = 0.0
    min_adjustment: float = 0.0
    max_adjustment: float = 0.75
    model_kind: str = "linear_feature_weights_plus_context_conditioned_adjustment_v1"
    neural_feature_order: Sequence[str] = field(default_factory=tuple)
    neural_hidden_weights: Sequence[Sequence[float]] = field(default_factory=tuple)
    neural_hidden_bias: Sequence[float] = field(default_factory=tuple)
    neural_output_weights: Sequence[float] = field(default_factory=tuple)
    neural_output_bias: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "schema_version": self.schema_version,
            "feature_weights": {
                str(key): float(value) for key, value in self.feature_weights.items()
            },
            "context_weights": {
                str(key): float(value) for key, value in self.context_weights.items()
            },
            "bias": float(self.bias),
            "context_bias": float(self.context_bias),
            "min_adjustment": float(self.min_adjustment),
            "max_adjustment": float(self.max_adjustment),
            "model_kind": self.model_kind,
            "neural_feature_order": [str(name) for name in self.neural_feature_order],
            "neural_hidden_weights": [
                [_safe_float(value) for value in row]
                for row in self.neural_hidden_weights
            ],
            "neural_hidden_bias": [
                _safe_float(value) for value in self.neural_hidden_bias
            ],
            "neural_output_weights": [
                _safe_float(value) for value in self.neural_output_weights
            ],
            "neural_output_bias": float(self.neural_output_bias),
            "metadata": dict(self.metadata),
        }


DEFAULT_DATAPACK_SELECTION_PRIOR_WEIGHTS: dict[str, float] = {
    "tag_coverage": 2.5,
    "exact_tag_match": 0.8,
    "gap_fill_score": 0.9,
    "objective_match": 0.55,
    "history_support_score": 0.65,
    "quality_score": 0.35,
    "novelty_score": 0.2,
    "semantic_grounding_non_heuristic": 0.35,
    "benchmark_eligible": 0.2,
    "execution_ready": 0.15,
    "cold_start_bonus": 0.1,
    "max_arh_penalty": -2.0,
}


@dataclass(frozen=True)
class DatapackSelectionDecision:
    datapack: DatapackConfig
    score: float
    source: str
    heuristic_score: float = 0.0
    learned_score: float = 0.0
    selection_policy: str = "heuristic_only"
    scorer_package_id: str | None = None
    matched_tags: Sequence[str] = field(default_factory=tuple)
    missing_tags: Sequence[str] = field(default_factory=tuple)
    gap_fill_tags: Sequence[str] = field(default_factory=tuple)
    exact_tag_match: bool = False
    objective_match: bool = False
    robot_match: bool = True
    historical_support: Mapping[str, Any] = field(default_factory=dict)
    benchmark_support: Mapping[str, Any] = field(default_factory=dict)
    candidate_metadata: Mapping[str, Any] = field(default_factory=dict)
    selection_features: Mapping[str, float] = field(default_factory=dict)
    scorer_trace: Mapping[str, Any] = field(default_factory=dict)
    reasons: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "datapack_id": self.datapack.id,
            "score": float(self.score),
            "source": self.source,
            "heuristic_score": float(self.heuristic_score),
            "learned_score": float(self.learned_score),
            "selection_policy": self.selection_policy,
            "scorer_package_id": self.scorer_package_id,
            "matched_tags": list(self.matched_tags),
            "missing_tags": list(self.missing_tags),
            "gap_fill_tags": list(self.gap_fill_tags),
            "exact_tag_match": bool(self.exact_tag_match),
            "objective_match": bool(self.objective_match),
            "robot_match": bool(self.robot_match),
            "historical_support": dict(self.historical_support),
            "benchmark_support": dict(self.benchmark_support),
            "candidate_metadata": dict(self.candidate_metadata),
            "selection_features": {
                str(key): float(value) for key, value in self.selection_features.items()
            },
            "scorer_trace": dict(self.scorer_trace),
            "reasons": list(self.reasons),
        }


def coerce_datapack_selection_scorer_package(
    payload: DatapackSelectionScorerPackage | Mapping[str, Any] | None,
) -> DatapackSelectionScorerPackage | None:
    if payload is None:
        return None
    if isinstance(payload, DatapackSelectionScorerPackage):
        return payload
    feature_weights = {
        str(key): float(value)
        for key, value in dict(payload.get("feature_weights", {}) or {}).items()
    }
    context_weights = {
        str(key): float(value)
        for key, value in dict(payload.get("context_weights", {}) or {}).items()
    }
    return DatapackSelectionScorerPackage(
        package_id=str(payload.get("package_id", "datapack_selection_helper")),
        schema_version=str(
            payload.get("schema_version", "datapack_selection_scorer_v1")
        ),
        feature_weights=feature_weights,
        context_weights=context_weights,
        bias=_safe_float(payload.get("bias", 0.0)),
        context_bias=_safe_float(payload.get("context_bias", 0.0)),
        min_adjustment=max(0.0, _safe_float(payload.get("min_adjustment", 0.0))),
        max_adjustment=max(0.0, _safe_float(payload.get("max_adjustment", 0.75))),
        model_kind=str(
            payload.get(
                "model_kind",
                "linear_feature_weights_plus_context_conditioned_adjustment_v1",
            )
        ),
        neural_feature_order=[
            str(name) for name in list(payload.get("neural_feature_order", []) or [])
        ],
        neural_hidden_weights=[
            [_safe_float(value) for value in list(row or [])]
            for row in list(payload.get("neural_hidden_weights", []) or [])
        ],
        neural_hidden_bias=[
            _safe_float(value)
            for value in list(payload.get("neural_hidden_bias", []) or [])
        ],
        neural_output_weights=[
            _safe_float(value)
            for value in list(payload.get("neural_output_weights", []) or [])
        ],
        neural_output_bias=_safe_float(payload.get("neural_output_bias", 0.0)),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def load_datapack_selection_scorer_package(
    path: str | Path,
) -> DatapackSelectionScorerPackage:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    package = coerce_datapack_selection_scorer_package(payload)
    if package is None:
        raise ValueError(f"datapack selection scorer package is empty: {path}")
    return package


def rank_datapacks_for_intent(
    tags: Sequence[str],
    robot_family: str | None,
    objective_hint: str | None,
    candidates: Sequence[DatapackConfig],
    scenarios: Sequence[ScenarioMetadata | Mapping[str, Any]],
    *,
    candidate_metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    selection_scorer_package: DatapackSelectionScorerPackage
    | Mapping[str, Any]
    | None = None,
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
    scorer_package = coerce_datapack_selection_scorer_package(selection_scorer_package)
    eligible_candidates: list[
        tuple[DatapackConfig, set[str], list[str], dict[str, Any], dict[str, Any]]
    ] = []
    for cfg in candidates:
        cfg_tags = {t.lower() for t in cfg.tags}
        matched_tags = sorted(required_tags & cfg_tags)
        if required_tags and not matched_tags:
            continue
        if robot_norm and cfg.robot_families:
            robot_match = robot_norm in {t.lower() for t in cfg.robot_families}
            if not robot_match:
                continue
        candidate_metadata = dict(metadata_by_id.get(cfg.id, {}) or {})
        benchmark_support = _candidate_benchmark_support(candidate_metadata)
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
        eligible_candidates.append(
            (
                cfg,
                cfg_tags,
                matched_tags,
                candidate_metadata,
                benchmark_support | {"history": history},
            )
        )
    selection_context = _build_datapack_selection_context(
        required_tags=required_tags,
        gap_tags=gap_tags,
        objective_norm=objective_norm,
        robot_norm=robot_norm,
        eligible_candidates=eligible_candidates,
    )

    decisions: list[DatapackSelectionDecision] = []
    for (
        cfg,
        cfg_tags,
        matched_tags,
        candidate_metadata,
        benchmark_payload,
    ) in eligible_candidates:
        benchmark_support = {
            key: value for key, value in benchmark_payload.items() if key != "history"
        }
        history = dict(benchmark_payload.get("history", {}) or {})
        robot_match = True
        execution_ready = bool(benchmark_support.get("execution_ready", False))
        quality_score = _clamp01(
            _safe_float(candidate_metadata.get("quality_score", 0.0))
        )
        novelty_score = _clamp01(
            _safe_float(candidate_metadata.get("novelty_score", 0.0))
        )
        objective_match = bool(
            objective_norm
            and cfg.objective_hint
            and objective_norm in cfg.objective_hint.lower()
        )
        exact_tag_match = not required_tags or required_tags.issubset(cfg_tags)
        gap_fill_tags = sorted(gap_tags & cfg_tags)
        tag_coverage = (
            (len(matched_tags) / float(len(required_tags))) if required_tags else 1.0
        )
        gap_fill_score = (
            (len(gap_fill_tags) / float(len(gap_tags))) if gap_tags else 0.0
        )
        features = DatapackSelectionFeatures(
            tag_coverage=tag_coverage,
            exact_tag_match=1.0 if exact_tag_match else 0.0,
            gap_fill_score=gap_fill_score,
            objective_match=1.0 if objective_match else 0.0,
            history_support_score=_clamp01(
                _safe_float(history.get("support_score", 0.0))
            ),
            quality_score=quality_score,
            novelty_score=novelty_score,
            semantic_grounding_non_heuristic=(
                1.0
                if benchmark_support.get("semantic_grounding_non_heuristic", False)
                else 0.0
            ),
            benchmark_eligible=1.0
            if benchmark_support.get("benchmark_eligible", False)
            else 0.0,
            execution_ready=1.0 if execution_ready else 0.0,
            cold_start_bonus=1.0
            if history.get("scenario_count", 0) == 0 and exact_tag_match
            else 0.0,
            max_arh_penalty=_clamp01(_safe_float(history.get("max_arh_penalty", 0.0))),
            mean_adjusted_mpl_norm=_clamp01(
                _safe_float(history.get("mean_adjusted_mpl", 0.0)) / 100.0
            ),
            mean_reward_norm=_clamp01(
                _safe_float(history.get("mean_reward", 0.0)) / 10.0
            ),
            scenario_count_norm=_clamp01(
                _safe_float(history.get("scenario_count", 0)) / 10.0
            ),
            eval_count_norm=_clamp01(_safe_float(history.get("eval_count", 0)) / 10.0),
        )
        heuristic_score = _score_datapack_selection_prior(features)
        scorer_trace = _score_datapack_selection_helper(
            features,
            scorer_package,
            selection_context=selection_context,
        )
        learned_score = _safe_float(scorer_trace.get("bounded_adjustment", 0.0))
        score = heuristic_score + learned_score
        selection_policy = (
            "heuristic_plus_learned_helper"
            if scorer_package is not None
            else "heuristic_only"
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
            reasons.append(
                f"arh_penalty:{_safe_float(history.get('max_arh_penalty', 0.0)):.2f}"
            )
        if scorer_package is not None:
            reasons.append(f"learned_adjustment:{learned_score:+.2f}")
            top_contributors = scorer_trace.get("top_contributors", []) or []
            if top_contributors:
                reasons.append(
                    "learned_features:"
                    + ",".join(
                        str(item.get("feature"))
                        for item in list(top_contributors)[:3]
                        if item.get("feature")
                    )
                )
            top_context_contributors = (
                scorer_trace.get("context_trace", {}).get("top_contributors", []) or []
            )
            if top_context_contributors:
                reasons.append(
                    "learned_context:"
                    + ",".join(
                        str(item.get("feature"))
                        for item in list(top_context_contributors)[:3]
                        if item.get("feature")
                    )
                )

        decisions.append(
            DatapackSelectionDecision(
                datapack=cfg,
                score=float(score),
                source=source,
                heuristic_score=float(heuristic_score),
                learned_score=float(learned_score),
                selection_policy=selection_policy,
                scorer_package_id=(
                    scorer_package.package_id if scorer_package is not None else None
                ),
                matched_tags=matched_tags,
                missing_tags=sorted(required_tags - cfg_tags),
                gap_fill_tags=gap_fill_tags,
                exact_tag_match=exact_tag_match,
                objective_match=objective_match,
                robot_match=robot_match,
                historical_support=history,
                benchmark_support=benchmark_support,
                candidate_metadata=candidate_metadata,
                selection_features=features.to_dict(),
                scorer_trace=scorer_trace,
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
    selection_helper_status: Mapping[str, Any] | None = None,
    selection_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_rows = list(selected or ranked)
    scorer_package_ids = sorted(
        {str(row.scorer_package_id) for row in selected_rows if row.scorer_package_id}
    )
    return {
        "required_tags": sorted(
            {str(tag).strip().lower() for tag in tags if str(tag).strip()}
        ),
        "robot_family": robot_family,
        "objective_hint": objective_hint,
        "candidate_count": len(ranked),
        "selection_policy": (
            selected_rows[0].selection_policy if selected_rows else "heuristic_only"
        ),
        "scorer_package_id": scorer_package_ids[0] if scorer_package_ids else None,
        "selected_ids": [row.datapack.id for row in selected_rows],
        "selected_sources": sorted({row.source for row in selected_rows}),
        "selected_gap_fill_tags": sorted(
            {tag for row in selected_rows for tag in row.gap_fill_tags}
        ),
        "selection_helper_status": dict(selection_helper_status or {}),
        "selection_context": dict(selection_context or {}),
        "selection_meta_choice": _selection_meta_choice_summary(
            ranked,
            selected_rows=selected_rows,
            required_tags=tags,
            selection_helper_status=selection_helper_status,
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
    selection_scorer_package: DatapackSelectionScorerPackage
    | Mapping[str, Any]
    | None = None,
    source: str = "ontology",
) -> list[DatapackConfig]:
    ranked = rank_datapacks_for_intent(
        tags,
        robot_family,
        objective_hint,
        candidates,
        scenarios,
        candidate_metadata_by_id=candidate_metadata_by_id,
        selection_scorer_package=selection_scorer_package,
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
    return {
        str(t).strip().lower()
        for t in scenario.get("datapack_tags") or []
        if str(t).strip()
    }


def _scenario_robot_families(
    scenario: ScenarioMetadata | Mapping[str, Any],
) -> set[str]:
    if isinstance(scenario, ScenarioMetadata):
        return {t.lower() for t in scenario.robot_families}
    return {
        str(t).strip().lower()
        for t in scenario.get("robot_families") or []
        if str(t).strip()
    }


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
        datapack_ids = [
            str(dp_id) for dp_id in scenario.get("datapack_ids") or [] if str(dp_id)
        ]
        if not datapack_ids:
            continue
        adjusted_train = _adjusted_metrics(scenario.get("train_metrics"))
        adjusted_eval = _adjusted_metrics(scenario.get("eval_metrics"))
        adjusted_mpl = max(
            _safe_float(
                adjusted_eval.get(
                    "mpl_units_per_hour_adjusted",
                    adjusted_eval.get("mpl_units_per_hour", 0.0),
                )
            ),
            _safe_float(
                adjusted_train.get(
                    "mpl_units_per_hour_adjusted",
                    adjusted_train.get("mpl_units_per_hour", 0.0),
                )
            ),
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
            row["max_arh_penalty"] = max(
                _safe_float(row.get("max_arh_penalty", 0.0)), arh_penalty
            )

    finalized: dict[str, dict[str, Any]] = {}
    for datapack_id, row in support.items():
        mpl_values = list(row.get("adjusted_mpl_samples", []) or [])
        reward_values = list(row.get("reward_samples", []) or [])
        mean_adjusted_mpl = sum(mpl_values) / float(max(len(mpl_values), 1))
        mean_reward = sum(reward_values) / float(max(len(reward_values), 1))
        support_score = _clamp01(
            (mean_adjusted_mpl / 100.0) * 0.7 + (mean_reward / 10.0) * 0.3
        )
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


def _candidate_benchmark_support(
    candidate_metadata: Mapping[str, Any],
) -> dict[str, Any]:
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
        isinstance(execution_preconditions, Mapping)
        and execution_preconditions.get("ready", False)
    )
    return signals


def _score_datapack_selection_prior(features: DatapackSelectionFeatures) -> float:
    feature_map = features.to_dict()
    return sum(
        _safe_float(DEFAULT_DATAPACK_SELECTION_PRIOR_WEIGHTS.get(name, 0.0))
        * _safe_float(value)
        for name, value in feature_map.items()
    )


def _score_datapack_selection_helper(
    features: DatapackSelectionFeatures,
    scorer_package: DatapackSelectionScorerPackage | None,
    *,
    selection_context: DatapackSelectionContext | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    feature_map = features.to_dict()
    if scorer_package is None:
        return {
            "policy": "heuristic_only",
            "raw_score": 0.0,
            "bounded_adjustment": 0.0,
            "effective_max_adjustment": 0.0,
            "context_trace": {
                "policy": "heuristic_only",
                "effective_max_adjustment": 0.0,
                "top_contributors": [],
            },
            "top_contributors": [],
        }
    neural_trace = _score_datapack_selection_neural_helper(feature_map, scorer_package)
    contributions = list(neural_trace.get("top_contributors", []) or [])
    raw_score = _safe_float(neural_trace.get("raw_score", 0.0))
    context_trace = _score_datapack_selection_context(
        selection_context,
        scorer_package=scorer_package,
    )
    effective_max_adjustment = max(
        0.0,
        _safe_float(
            context_trace.get("effective_max_adjustment", scorer_package.max_adjustment)
        ),
    )
    bounded_adjustment = (
        math.tanh(raw_score) * effective_max_adjustment
        if effective_max_adjustment > 0.0
        else 0.0
    )
    return {
        "policy": "heuristic_plus_learned_helper",
        "package_id": scorer_package.package_id,
        "schema_version": scorer_package.schema_version,
        "model_kind": scorer_package.model_kind,
        "raw_score": float(raw_score),
        "bounded_adjustment": float(bounded_adjustment),
        "max_adjustment": float(max(0.0, _safe_float(scorer_package.max_adjustment))),
        "effective_max_adjustment": float(effective_max_adjustment),
        "context_trace": context_trace,
        "top_contributors": contributions[:5],
        "network_trace": neural_trace.get("network_trace", {}),
        "metadata": dict(scorer_package.metadata),
    }


def _build_datapack_selection_context(
    *,
    required_tags: set[str],
    gap_tags: set[str],
    objective_norm: str | None,
    robot_norm: str | None,
    eligible_candidates: Sequence[
        tuple[DatapackConfig, set[str], list[str], Mapping[str, Any], Mapping[str, Any]]
    ],
) -> DatapackSelectionContext:
    candidate_count = len(eligible_candidates)
    if candidate_count <= 0:
        return DatapackSelectionContext(
            required_tag_count_norm=_clamp01(len(required_tags) / 8.0),
            gap_pressure=1.0 if gap_tags else 0.0,
            objective_present=1.0 if objective_norm else 0.0,
            robot_specificity=1.0 if robot_norm else 0.0,
            cold_start_pressure=1.0,
        )
    benchmark_ready_ratio = sum(
        1.0
        for _cfg, _cfg_tags, _matched_tags, _metadata, benchmark_payload in eligible_candidates
        if bool(benchmark_payload.get("benchmark_eligible", False))
    ) / float(candidate_count)
    execution_ready_ratio = sum(
        1.0
        for _cfg, _cfg_tags, _matched_tags, _metadata, benchmark_payload in eligible_candidates
        if bool(benchmark_payload.get("execution_ready", False))
    ) / float(candidate_count)
    history_density = sum(
        _clamp01(
            _safe_float(
                dict(benchmark_payload.get("history", {}) or {}).get(
                    "scenario_count", 0
                )
            )
            / 10.0
        )
        for _cfg, _cfg_tags, _matched_tags, _metadata, benchmark_payload in eligible_candidates
    ) / float(candidate_count)
    return DatapackSelectionContext(
        required_tag_count_norm=_clamp01(len(required_tags) / 8.0),
        gap_pressure=(len(gap_tags) / float(len(required_tags)))
        if required_tags
        else 0.0,
        candidate_pool_size_norm=_clamp01(candidate_count / 10.0),
        benchmark_ready_ratio=_clamp01(benchmark_ready_ratio),
        execution_ready_ratio=_clamp01(execution_ready_ratio),
        history_density=_clamp01(history_density),
        cold_start_pressure=_clamp01(1.0 - history_density),
        objective_present=1.0 if objective_norm else 0.0,
        robot_specificity=1.0 if robot_norm else 0.0,
    )


def _score_datapack_selection_context(
    selection_context: DatapackSelectionContext | Mapping[str, Any] | None,
    *,
    scorer_package: DatapackSelectionScorerPackage,
) -> dict[str, Any]:
    if isinstance(selection_context, DatapackSelectionContext):
        context_map = selection_context.to_dict()
    elif isinstance(selection_context, Mapping):
        context_map = {
            str(key): _safe_float(value)
            for key, value in dict(selection_context).items()
        }
    else:
        context_map = {}
    min_adjustment = max(0.0, _safe_float(scorer_package.min_adjustment))
    max_adjustment = max(min_adjustment, _safe_float(scorer_package.max_adjustment))
    if not scorer_package.context_weights:
        return {
            "policy": "unconditioned_max_adjustment",
            "raw_score": 0.0,
            "scale": 1.0,
            "min_adjustment": float(min_adjustment),
            "max_adjustment": float(max_adjustment),
            "effective_max_adjustment": float(max_adjustment),
            "top_contributors": [],
            "context": context_map,
        }
    contributions: list[dict[str, Any]] = []
    raw_score = _safe_float(scorer_package.context_bias)
    for feature_name, feature_value in sorted(context_map.items()):
        weight = _safe_float(scorer_package.context_weights.get(feature_name, 0.0))
        contribution = weight * _safe_float(feature_value)
        raw_score += contribution
        if abs(contribution) > 0.0:
            contributions.append(
                {
                    "feature": feature_name,
                    "feature_value": _safe_float(feature_value),
                    "weight": weight,
                    "contribution": contribution,
                }
            )
    contributions.sort(
        key=lambda row: abs(_safe_float(row.get("contribution", 0.0))),
        reverse=True,
    )
    scale = 1.0 / (1.0 + math.exp(-raw_score))
    effective_max_adjustment = (
        min_adjustment + (max_adjustment - min_adjustment) * scale
    )
    return {
        "policy": "context_conditioned_max_adjustment",
        "raw_score": float(raw_score),
        "scale": float(scale),
        "min_adjustment": float(min_adjustment),
        "max_adjustment": float(max_adjustment),
        "effective_max_adjustment": float(effective_max_adjustment),
        "top_contributors": contributions[:5],
        "context": context_map,
    }


def _score_datapack_selection_neural_helper(
    feature_map: Mapping[str, float],
    scorer_package: DatapackSelectionScorerPackage,
) -> dict[str, Any]:
    feature_names = list(
        scorer_package.neural_feature_order or sorted(feature_map.keys())
    )
    if (
        scorer_package.model_kind.startswith("neural_")
        and scorer_package.neural_hidden_weights
        and scorer_package.neural_output_weights
        and len(scorer_package.neural_hidden_weights[0]) == len(feature_names)
        and len(scorer_package.neural_hidden_weights)
        == len(scorer_package.neural_output_weights)
    ):
        feature_vector = np.asarray(
            [_safe_float(feature_map.get(name, 0.0)) for name in feature_names],
            dtype=np.float32,
        )
        hidden_weights = np.asarray(
            scorer_package.neural_hidden_weights, dtype=np.float32
        )
        hidden_bias = np.asarray(scorer_package.neural_hidden_bias, dtype=np.float32)
        if hidden_bias.size != hidden_weights.shape[0]:
            hidden_bias = np.zeros(hidden_weights.shape[0], dtype=np.float32)
        output_weights = np.asarray(
            scorer_package.neural_output_weights, dtype=np.float32
        )
        hidden_pre = hidden_weights @ feature_vector + hidden_bias
        hidden_act = np.maximum(hidden_pre, 0.0)
        raw_score = float(
            np.dot(output_weights, hidden_act)
            + _safe_float(scorer_package.neural_output_bias)
        )

        active_mask = (hidden_pre > 0.0).astype(np.float32)
        local_linear = (output_weights * active_mask) @ hidden_weights
        neural_contributions = [
            {
                "feature": name,
                "feature_value": _safe_float(feature_map.get(name, 0.0)),
                "weight": float(local_linear[index]),
                "contribution": float(local_linear[index] * feature_vector[index]),
            }
            for index, name in enumerate(feature_names)
            if abs(float(local_linear[index] * feature_vector[index])) > 0.0
        ]
        neural_contributions.sort(
            key=lambda row: abs(_safe_float(row.get("contribution", 0.0))),
            reverse=True,
        )
        active_units = [
            {
                "unit": int(index),
                "pre_activation": float(hidden_pre[index]),
                "activation": float(hidden_act[index]),
                "output_weight": float(output_weights[index]),
            }
            for index in np.argsort(-np.abs(hidden_act * output_weights))[:5]
            if index < hidden_act.shape[0] and abs(float(hidden_act[index])) > 0.0
        ]
        return {
            "policy": "neural_feature_mlp",
            "raw_score": raw_score,
            "top_contributors": neural_contributions[:5],
            "network_trace": {
                "active_hidden_units": active_units,
                "feature_order": feature_names,
            },
        }

    contributions: list[dict[str, Any]] = []
    raw_score = _safe_float(scorer_package.bias)
    for feature_name, feature_value in sorted(feature_map.items()):
        weight = _safe_float(scorer_package.feature_weights.get(feature_name, 0.0))
        contribution = weight * _safe_float(feature_value)
        raw_score += contribution
        if abs(contribution) > 0.0:
            contributions.append(
                {
                    "feature": feature_name,
                    "feature_value": _safe_float(feature_value),
                    "weight": weight,
                    "contribution": contribution,
                }
            )
    contributions.sort(
        key=lambda row: abs(_safe_float(row.get("contribution", 0.0))),
        reverse=True,
    )
    return {
        "policy": "linear_feature_weights",
        "raw_score": float(raw_score),
        "top_contributors": contributions[:5],
        "network_trace": {},
    }


def _selection_meta_choice_summary(
    ranked: Sequence[DatapackSelectionDecision],
    *,
    selected_rows: Sequence[DatapackSelectionDecision],
    required_tags: Sequence[str],
    selection_helper_status: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not selected_rows:
        return {
            "selection_policy": "heuristic_only",
            "candidate_count": len(ranked),
            "required_tag_count": len(list(required_tags or [])),
        }
    top_row = selected_rows[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    required_tag_count = len([tag for tag in required_tags if str(tag).strip()])
    gap_fill_ratio = (
        len(list(top_row.gap_fill_tags or [])) / float(max(required_tag_count, 1))
        if required_tag_count
        else 0.0
    )
    selected_execution_ready = bool(
        top_row.benchmark_support.get("execution_ready", False)
        or _safe_float(top_row.selection_features.get("execution_ready", 0.0)) >= 0.5
    )
    selected_non_heuristic_grounding = bool(
        top_row.benchmark_support.get("semantic_grounding_non_heuristic", False)
        or _safe_float(
            top_row.selection_features.get("semantic_grounding_non_heuristic", 0.0)
        )
        >= 0.5
    )
    return {
        "selected_datapack_id": top_row.datapack.id,
        "selection_policy": top_row.selection_policy,
        "scorer_package_id": top_row.scorer_package_id,
        "candidate_count": len(ranked),
        "required_tag_count": required_tag_count,
        "selected_gap_fill_ratio": float(gap_fill_ratio),
        "top_score": float(top_row.score),
        "margin_to_runner_up": float(top_row.score - runner_up.score)
        if runner_up is not None
        else float(top_row.score),
        "heuristic_score": float(top_row.heuristic_score),
        "learned_score": float(top_row.learned_score),
        "selected_execution_ready": selected_execution_ready,
        "selected_non_heuristic_grounding": selected_non_heuristic_grounding,
        "selected_benchmark_eligible": bool(
            top_row.benchmark_support.get("benchmark_eligible", False)
        ),
        "selected_quality_score": float(
            top_row.candidate_metadata.get("quality_score", 0.0) or 0.0
        ),
        "selected_history_support": float(
            top_row.historical_support.get("support_score", 0.0) or 0.0
        ),
        "helper_status": dict(selection_helper_status or {}),
        "top_reasons": list(top_row.reasons)[:5],
        "top_contributors": list(
            top_row.scorer_trace.get("top_contributors", []) or []
        )[:3],
        "top_context_contributors": list(
            (top_row.scorer_trace.get("context_trace", {}) or {}).get(
                "top_contributors", []
            )
            or []
        )[:3],
    }


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
    "DatapackSelectionContext",
    "DatapackSelectionFeatures",
    "DatapackSelectionDecision",
    "DatapackSelectionScorerPackage",
    "MissingScenarioSpec",
    "apply_arh_penalty",
    "coerce_datapack_selection_scorer_package",
    "detect_semantic_gaps",
    "load_datapack_selection_scorer_package",
    "rank_datapacks_for_intent",
    "select_datapacks_for_intent",
    "summarize_datapack_selection",
]

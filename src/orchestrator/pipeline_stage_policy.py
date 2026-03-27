from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


PIPELINE_STAGE_LABELS: tuple[str, ...] = (
    "objective_solving",
    "data_collection",
    "policy_training",
    "evaluation",
    "feedback_iteration",
)

PIPELINE_CONFIG_FLAG_KEYS: tuple[str, ...] = (
    "increase_safety_weight",
    "increase_data_collection",
    "repair_execution_preconditions",
)

PIPELINE_OBJECTIVE_PRESET_LABELS: tuple[str, ...] = (
    "balanced",
    "safety",
    "throughput",
    "energy_saver",
)

PIPELINE_STAGE_POLICY_FEATURE_NAMES: tuple[str, ...] = (
    "iteration_count_norm",
    "completed_fraction",
    "failed_fraction",
    "last_error_rate",
    "last_mpl_delta",
    "last_energy_efficiency",
    "trend_error_rate",
    "trend_mpl_delta",
    "trend_energy_efficiency",
    "execution_report_count_norm",
    "execution_ready_fraction",
    "execution_blocked_fraction",
    "execution_mean_readiness",
    "satisfied_precondition_count_norm",
    "blocking_precondition_count_norm",
    "activation_ready",
    "activation_readiness_score",
    "future_training_ready",
    "objective_is_balanced",
    "objective_is_safety",
    "objective_is_throughput",
    "objective_is_energy_saver",
    "flag_increase_safety_weight",
    "flag_increase_data_collection",
    "flag_repair_execution_preconditions",
    "objective_solving_success_rate",
    "objective_solving_failure_rate",
    "objective_solving_duration_norm",
    "data_collection_success_rate",
    "data_collection_failure_rate",
    "data_collection_duration_norm",
    "policy_training_success_rate",
    "policy_training_failure_rate",
    "policy_training_duration_norm",
    "evaluation_success_rate",
    "evaluation_failure_rate",
    "evaluation_duration_norm",
    "feedback_iteration_success_rate",
    "feedback_iteration_failure_rate",
    "feedback_iteration_duration_norm",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _count_norm(value: Any, *, scale: float = 10.0, cap: float = 1.0) -> float:
    normalized = _safe_float(value, 0.0) / max(scale, 1.0)
    return max(0.0, min(cap, normalized))


def _normalize_distribution(values: Mapping[str, Any], labels: Sequence[str]) -> Dict[str, float]:
    total = 0.0
    normalized: Dict[str, float] = {}
    for label in labels:
        candidate = max(0.0, _safe_float(values.get(label, 0.0)))
        normalized[str(label)] = candidate
        total += candidate
    if total <= 0.0:
        fallback = {str(label): 0.0 for label in labels}
        if labels:
            fallback[str(labels[0])] = 1.0
        return fallback
    return {str(label): float(normalized[str(label)] / total) for label in labels}


def _iteration_stage_results(iteration: Any) -> Mapping[str, Any]:
    stage_results = getattr(iteration, "stage_results", None)
    if isinstance(stage_results, Mapping):
        return stage_results
    if isinstance(iteration, Mapping):
        maybe = iteration.get("stage_results")
        if isinstance(maybe, Mapping):
            return maybe
    return {}


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def summarize_pipeline_stage_history(iterations: Sequence[Any]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for stage_label in PIPELINE_STAGE_LABELS:
        total_runs = 0
        completed_runs = 0
        failed_runs = 0
        duration_total = 0.0
        duration_count = 0
        for iteration in iterations:
            stage_results = _iteration_stage_results(iteration)
            result = stage_results.get(stage_label)
            if result is None:
                continue
            total_runs += 1
            status = str(_result_value(result, "status", "not_started") or "not_started")
            if status == "completed":
                completed_runs += 1
            elif status == "failed":
                failed_runs += 1
            duration = _safe_float(_result_value(result, "duration_seconds", 0.0), 0.0)
            if duration > 0.0:
                duration_total += duration
                duration_count += 1
        mean_duration = duration_total / duration_count if duration_count else 0.0
        denom = float(max(total_runs, 1))
        summary[stage_label] = {
            "total_runs": float(total_runs),
            "success_rate": float(completed_runs / denom),
            "failure_rate": float(failed_runs / denom),
            "mean_duration_s": float(mean_duration),
        }
    return summary


def build_pipeline_stage_feature_map(
    *,
    config: Optional[Mapping[str, Any]] = None,
    iterations: Sequence[Any] = (),
    progress: Optional[Mapping[str, Any]] = None,
    execution_summary: Optional[Mapping[str, Any]] = None,
    shell_activation: Optional[Mapping[str, Any]] = None,
    last_results: Optional[Mapping[str, Any]] = None,
    suggested_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    cfg = dict(config or {})
    progress_payload = dict(progress or {})
    execution = dict(execution_summary or {})
    activation_payload = dict(shell_activation or {})
    last = dict(last_results or {})
    suggested = dict(suggested_config or {})
    trends = dict(progress_payload.get("trends", {}) or {})
    last_metrics = dict(last.get("summary_metrics", {}) or {})
    stage_history = summarize_pipeline_stage_history(iterations)

    objective = str(cfg.get("objective_preset", "balanced") or "balanced")
    report_count = int(execution.get("report_count", 0) or 0)
    ready_count = int(execution.get("ready_count", 0) or 0)
    blocked_count = int(execution.get("blocked_count", 0) or 0)
    satisfied = execution.get("satisfied_preconditions")
    blocking = execution.get("blocking_preconditions")
    satisfied_count = len(satisfied) if isinstance(satisfied, Mapping) else len(list(satisfied or []))
    blocking_count = len(blocking) if isinstance(blocking, Mapping) else len(list(blocking or []))

    activated = list(activation_payload.get("activated", []) or [])
    activation_ready_rows = list(activation_payload.get("activation_ready", []) or [])
    future_training = list(activation_payload.get("future_training", []) or [])
    activation_row = activated[0] if activated else {}
    activation_readiness = dict(activation_row.get("readiness", {}) or {})

    feature_map: Dict[str, float] = {
        "iteration_count_norm": _count_norm(progress_payload.get("iterations", len(iterations)), scale=8.0),
        "completed_fraction": _clamp01(
            _safe_float(progress_payload.get("completed_iterations", 0.0))
            / max(_safe_float(progress_payload.get("iterations", len(iterations)), 1.0), 1.0)
        ),
        "failed_fraction": _clamp01(
            _safe_float(progress_payload.get("failed_iterations", 0.0))
            / max(_safe_float(progress_payload.get("iterations", len(iterations)), 1.0), 1.0)
        ),
        "last_error_rate": _clamp01(last_metrics.get("error_rate", 0.0)),
        "last_mpl_delta": _clamp01((_safe_float(last_metrics.get("mpl_delta", 0.0)) + 2.0) / 6.0),
        "last_energy_efficiency": _clamp01((_safe_float(last_metrics.get("energy_efficiency", 0.0)) + 1.0) / 2.0),
        "trend_error_rate": _clamp01((_safe_float(trends.get("error_rate", 0.0)) + 0.2) / 0.4),
        "trend_mpl_delta": _clamp01((_safe_float(trends.get("mpl_delta", 0.0)) + 2.0) / 6.0),
        "trend_energy_efficiency": _clamp01((_safe_float(trends.get("energy_efficiency", 0.0)) + 1.0) / 2.0),
        "execution_report_count_norm": _count_norm(report_count, scale=4.0),
        "execution_ready_fraction": _clamp01(ready_count / float(max(report_count, 1))),
        "execution_blocked_fraction": _clamp01(blocked_count / float(max(report_count, 1))),
        "execution_mean_readiness": _clamp01(execution.get("mean_readiness_score", 0.0)),
        "satisfied_precondition_count_norm": _count_norm(satisfied_count, scale=8.0),
        "blocking_precondition_count_norm": _count_norm(blocking_count, scale=8.0),
        "activation_ready": 1.0 if activated else 0.0,
        "activation_readiness_score": _clamp01(activation_readiness.get("readiness_score", 0.0)),
        "future_training_ready": 1.0 if activation_ready_rows or future_training else 0.0,
        "objective_is_balanced": 1.0 if objective == "balanced" else 0.0,
        "objective_is_safety": 1.0 if objective == "safety" else 0.0,
        "objective_is_throughput": 1.0 if objective == "throughput" else 0.0,
        "objective_is_energy_saver": 1.0 if objective == "energy_saver" else 0.0,
        "flag_increase_safety_weight": 1.0 if suggested.get("increase_safety_weight") else 0.0,
        "flag_increase_data_collection": 1.0 if suggested.get("increase_data_collection") else 0.0,
        "flag_repair_execution_preconditions": 1.0 if suggested.get("repair_execution_preconditions") else 0.0,
    }
    for stage_label in PIPELINE_STAGE_LABELS:
        stage_summary = stage_history.get(stage_label, {})
        feature_map[f"{stage_label}_success_rate"] = _clamp01(stage_summary.get("success_rate", 0.0))
        feature_map[f"{stage_label}_failure_rate"] = _clamp01(stage_summary.get("failure_rate", 0.0))
        feature_map[f"{stage_label}_duration_norm"] = _count_norm(
            stage_summary.get("mean_duration_s", 0.0),
            scale=10.0,
        )
    return feature_map


def heuristic_stage_priority_distribution(feature_map: Mapping[str, Any]) -> Dict[str, float]:
    features = dict(feature_map or {})
    blocked = _clamp01(features.get("execution_blocked_fraction", 0.0))
    readiness = _clamp01(features.get("execution_mean_readiness", 0.0))
    activation_ready = _clamp01(features.get("activation_ready", 0.0))
    failed_fraction = _clamp01(features.get("failed_fraction", 0.0))
    last_error = _clamp01(features.get("last_error_rate", 0.0))
    trend_error = max(0.0, (_safe_float(features.get("trend_error_rate", 0.5)) - 0.5) * 2.0)
    trend_mpl_down = max(0.0, (0.5 - _safe_float(features.get("trend_mpl_delta", 0.5))) * 2.0)

    weights = {
        "objective_solving": (
            1.0
            + (0.45 * trend_error)
            + (0.25 * blocked)
            + (0.2 * _clamp01(features.get("objective_solving_failure_rate", 0.0)))
        ),
        "data_collection": (
            1.0
            + (0.65 * _clamp01(features.get("flag_increase_data_collection", 0.0)))
            + (0.45 * trend_mpl_down)
            + (0.15 * (1.0 - readiness))
            + (0.2 * _clamp01(features.get("data_collection_failure_rate", 0.0)))
        ),
        "policy_training": (
            1.0
            + (0.35 * activation_ready)
            + (0.25 * readiness)
            + (0.2 * _clamp01(features.get("flag_increase_safety_weight", 0.0)))
            + (0.2 * _clamp01(features.get("policy_training_failure_rate", 0.0)))
        ),
        "evaluation": (
            1.0
            + (0.35 * readiness)
            + (0.25 * last_error)
            + (0.15 * _clamp01(features.get("evaluation_failure_rate", 0.0)))
            + (0.15 * _clamp01(features.get("completed_fraction", 0.0)))
        ),
        "feedback_iteration": (
            1.0
            + (0.45 * failed_fraction)
            + (0.35 * trend_error)
            + (0.3 * blocked)
            + (0.2 * _clamp01(features.get("feedback_iteration_failure_rate", 0.0)))
        ),
    }
    return _normalize_distribution(weights, PIPELINE_STAGE_LABELS)


def heuristic_config_flag_scores(feature_map: Mapping[str, Any]) -> Dict[str, float]:
    features = dict(feature_map or {})
    last_error = _clamp01(features.get("last_error_rate", 0.0))
    mpl_delta = _clamp01(features.get("last_mpl_delta", 0.0))
    blocked = _clamp01(features.get("execution_blocked_fraction", 0.0))
    trend_mpl_down = max(0.0, (0.5 - _safe_float(features.get("trend_mpl_delta", 0.5))) * 2.0)
    return {
        "increase_safety_weight": _clamp01(max(features.get("flag_increase_safety_weight", 0.0), last_error)),
        "increase_data_collection": _clamp01(
            max(features.get("flag_increase_data_collection", 0.0), trend_mpl_down, 1.0 - mpl_delta)
        ),
        "repair_execution_preconditions": _clamp01(
            max(features.get("flag_repair_execution_preconditions", 0.0), blocked)
        ),
    }


def extract_pipeline_stage_policy_target(payload: Mapping[str, Any]) -> Dict[str, Any]:
    trace = dict(payload.get("stage_policy_trace", {}) or {})
    config_flags = dict(trace.get("final_config_flags", {}) or {})
    stage_distribution = dict(trace.get("final_stage_distribution", {}) or {})
    if not stage_distribution:
        stage_distribution = dict(trace.get("prior_stage_distribution", {}) or {})
    return {
        "stage_distribution": _normalize_distribution(stage_distribution, PIPELINE_STAGE_LABELS),
        "config_flag_scores": {
            key: _clamp01(config_flags.get(key, 0.0))
            for key in PIPELINE_CONFIG_FLAG_KEYS
        },
        "activation_label": 0.0 if str(payload.get("execution_mode", "advisory")) == "advisory" else 1.0,
        "policy_source": str(payload.get("policy_source", "heuristic_fallback") or "heuristic_fallback"),
        "promotion_stage": str(payload.get("promotion_stage", "heuristic_fallback") or "heuristic_fallback"),
        "stage_policy_trace": trace,
    }


__all__ = [
    "PIPELINE_CONFIG_FLAG_KEYS",
    "PIPELINE_OBJECTIVE_PRESET_LABELS",
    "PIPELINE_STAGE_LABELS",
    "PIPELINE_STAGE_POLICY_FEATURE_NAMES",
    "build_pipeline_stage_feature_map",
    "extract_pipeline_stage_policy_target",
    "heuristic_config_flag_scores",
    "heuristic_stage_priority_distribution",
    "summarize_pipeline_stage_history",
]

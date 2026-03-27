from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


QUEUE_POSITIVE_TAGS = {
    "high_value_uncertain",
    "frontier_candidate",
    "collect_more_like_this",
    "pricing_truth_review",
    "pricing_review",
    "reward_safety_review",
}

QUEUE_NEGATIVE_TAGS = {
    "low_provenance_review",
    "low_provenance",
    "downweight_candidate",
    "holdout_candidate",
}

QUEUE_DISPATCH_FEATURE_NAMES: tuple[str, ...] = (
    "priority_score",
    "replay_action_upweight",
    "replay_action_downweight",
    "replay_action_holdout",
    "replay_action_collect_more",
    "positive_tag_count_norm",
    "negative_tag_count_norm",
    "tag_count_norm",
    "promotion_compare_only",
    "promotion_advisory",
    "promotion_budget_gate",
    "promotion_narrow_hard_gate",
    "influence_heuristic",
    "influence_hybrid",
    "influence_learned",
    "deploy_allow_shadow",
    "deploy_require_review",
    "deploy_deny_shadow",
    "pricing_publish",
    "pricing_publish_discounted",
    "pricing_require_review",
    "datapack_keep",
    "datapack_downweight",
    "datapack_review",
    "semantic_route_success_prob",
    "semantic_orchestration_success_prob",
    "semantic_authority_success_prob",
    "semantic_regret_inverse",
    "inferential_signal_yield",
    "inferential_reward_value",
    "execution_blocked_fraction",
    "receipt_task_success",
    "receipt_objective_satisfied",
    "receipt_realized_value",
    "receipt_pricing_accepted",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _count_norm(values: Sequence[Any], *, scale: float = 4.0) -> float:
    return _clamp01(len(list(values or [])) / max(scale, 1.0))


def build_queue_dispatch_feature_map(entry: Mapping[str, Any]) -> Dict[str, float]:
    metadata = dict(entry.get("metadata", {}) or {})
    evidence = dict(metadata.get("evidence", {}) or {})
    semantic_runtime_score = dict(metadata.get("semantic_runtime_score", {}) or evidence.get("semantic_runtime_score", {}) or {})
    receipt_feedback = dict(evidence.get("receipt_feedback", {}) or {})
    deployment_outcome = dict(receipt_feedback.get("deployment_outcome", {}) or {})
    execution_preconditions = dict(metadata.get("execution_preconditions", {}) or evidence.get("execution_preconditions", {}) or {})
    replay_action = str(entry.get("replay_action", "holdout") or "holdout")
    tags = [str(value) for value in entry.get("tags", []) or []]
    positive_tag_count = sum(1 for tag in tags if tag in QUEUE_POSITIVE_TAGS)
    negative_tag_count = sum(1 for tag in tags if tag in QUEUE_NEGATIVE_TAGS)
    promotion_stage = str(metadata.get("promotion_stage", "compare_only") or "compare_only")
    influence_source = str(metadata.get("influence_source", "heuristic") or "heuristic")
    deploy_recommendation = str(metadata.get("deploy_recommendation", "") or "")
    pricing_recommendation = str(metadata.get("pricing_recommendation", "") or "")
    datapack_recommendation = str(metadata.get("datapack_recommendation", "") or "")
    blocked_count = int(execution_preconditions.get("blocked_count", 0) or 0)
    report_count = int(execution_preconditions.get("report_count", 0) or 0)
    inferential_reward = dict(evidence.get("inferential_reward", {}) or {})

    feature_map = {
        "priority_score": _clamp01(entry.get("priority_score", 0.0)),
        "replay_action_upweight": 1.0 if replay_action in {"upweight", "collect_more_like_this"} else 0.0,
        "replay_action_downweight": 1.0 if replay_action == "downweight" else 0.0,
        "replay_action_holdout": 1.0 if replay_action == "holdout" else 0.0,
        "replay_action_collect_more": 1.0 if replay_action == "collect_more_like_this" else 0.0,
        "positive_tag_count_norm": _clamp01(positive_tag_count / 3.0),
        "negative_tag_count_norm": _clamp01(negative_tag_count / 3.0),
        "tag_count_norm": _count_norm(tags),
        "promotion_compare_only": 1.0 if promotion_stage == "compare_only" else 0.0,
        "promotion_advisory": 1.0 if promotion_stage == "advisory" else 0.0,
        "promotion_budget_gate": 1.0 if promotion_stage == "budget_gate" else 0.0,
        "promotion_narrow_hard_gate": 1.0 if promotion_stage == "narrow_hard_gate" else 0.0,
        "influence_heuristic": 1.0 if influence_source == "heuristic" else 0.0,
        "influence_hybrid": 1.0 if influence_source == "hybrid" else 0.0,
        "influence_learned": 1.0 if influence_source == "learned" else 0.0,
        "deploy_allow_shadow": 1.0 if deploy_recommendation == "allow_shadow" else 0.0,
        "deploy_require_review": 1.0 if deploy_recommendation in {"require_review", "review"} else 0.0,
        "deploy_deny_shadow": 1.0 if deploy_recommendation in {"deny_shadow", "suppress"} else 0.0,
        "pricing_publish": 1.0 if pricing_recommendation == "publish" else 0.0,
        "pricing_publish_discounted": 1.0 if pricing_recommendation == "publish_discounted" else 0.0,
        "pricing_require_review": 1.0 if pricing_recommendation in {"review", "require_review"} else 0.0,
        "datapack_keep": 1.0 if datapack_recommendation in {"keep", "allow", ""} else 0.0,
        "datapack_downweight": 1.0 if datapack_recommendation == "downweight" else 0.0,
        "datapack_review": 1.0 if datapack_recommendation in {"review", "deny", "suppress"} else 0.0,
        "semantic_route_success_prob": _clamp01(
            semantic_runtime_score.get("meta_route_success_probability", 0.0)
        ),
        "semantic_orchestration_success_prob": _clamp01(
            semantic_runtime_score.get("orchestration_route_success_probability", 0.0)
        ),
        "semantic_authority_success_prob": _clamp01(
            semantic_runtime_score.get("authority_success_probability", 0.0)
        ),
        "semantic_regret_inverse": _clamp01(
            1.0 - _safe_float(semantic_runtime_score.get("estimated_regret", 1.0), 1.0)
        ),
        "inferential_signal_yield": _clamp01(
            metadata.get("inferential_signal_yield", evidence.get("inferential_signal_yield", 0.0))
        ),
        "inferential_reward_value": _clamp01(
            max(_safe_float(inferential_reward.get("value_score", 0.0), 0.0), 0.0)
        ),
        "execution_blocked_fraction": _clamp01(blocked_count / float(max(report_count, 1))),
        "receipt_task_success": 1.0 if deployment_outcome.get("task_success") else 0.0,
        "receipt_objective_satisfied": 1.0 if deployment_outcome.get("objective_satisfied") else 0.0,
        "receipt_realized_value": _clamp01(
            (_safe_float(deployment_outcome.get("realized_value", 0.0), 0.0) + 2.0) / 4.0
        ),
        "receipt_pricing_accepted": 1.0 if deployment_outcome.get("pricing_accepted") else 0.0,
    }
    return feature_map


def extract_queue_dispatch_target(entry: Mapping[str, Any]) -> Dict[str, Any]:
    feature_map = build_queue_dispatch_feature_map(entry)
    receipt_available = (
        feature_map["receipt_task_success"] > 0.0
        or feature_map["receipt_objective_satisfied"] > 0.0
        or feature_map["receipt_pricing_accepted"] > 0.0
        or feature_map["receipt_realized_value"] > 0.5
    )
    if receipt_available:
        target_score = (
            0.35 * feature_map["receipt_task_success"]
            + 0.2 * feature_map["receipt_objective_satisfied"]
            + 0.3 * feature_map["receipt_realized_value"]
            + 0.15 * feature_map["receipt_pricing_accepted"]
        )
        target_source = "receipt_feedback"
    else:
        target_score = (
            0.45 * feature_map["priority_score"]
            + 0.2 * feature_map["semantic_route_success_prob"]
            + 0.15 * feature_map["semantic_orchestration_success_prob"]
            + 0.1 * feature_map["inferential_signal_yield"]
            + 0.1 * (1.0 - feature_map["execution_blocked_fraction"])
        )
        target_source = "heuristic_bootstrap"
    return {
        "dispatch_score": _clamp01(target_score),
        "target_source": target_source,
    }


__all__ = [
    "QUEUE_DISPATCH_FEATURE_NAMES",
    "QUEUE_NEGATIVE_TAGS",
    "QUEUE_POSITIVE_TAGS",
    "build_queue_dispatch_feature_map",
    "extract_queue_dispatch_target",
]

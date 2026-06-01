from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from src.semantic.models import SemanticSnapshot


SHELL_POLICY_PRESET_LABELS: tuple[str, ...] = (
    "balanced",
    "safety",
    "energy_saver",
    "throughput",
)
SHELL_POLICY_STRATEGY_KEYS: tuple[str, ...] = (
    "balanced",
    "frontier_prioritized",
    "econ_urgency",
)
SHELL_POLICY_FEATURE_NAMES: tuple[str, ...] = (
    "avg_wage_parity",
    "wage_gap",
    "avg_energy_cost",
    "avg_error_rate",
    "frontier_episode_count_norm",
    "recap_mean_goodness",
    "recap_top_episode_fraction",
    "num_segments_norm",
    "recovery_segment_fraction",
    "mobility_drift_rate",
    "blocked_count_norm",
    "ready_count_norm",
    "mean_readiness_score",
    "max_ood_severity",
    "ood_trust",
    "risk_triage_score",
    "recovery_router_score",
    "semantic_memory_refresh_score",
    "fusion_bridge_score",
    "ontology_router_score",
    "task_graph_router_score",
    "efficiency_router_score",
    "risk_reasoning",
    "stage2_bridge",
    "fusion_bridge_capability",
    "object_memory",
    "affordance_grounding",
    "meta_node_orchestration",
    "expected_delta_mpl_norm",
    "expected_delta_error_norm",
    "expected_delta_energy_norm",
    "preset_balanced_available",
    "preset_safety_available",
    "preset_energy_saver_available",
    "preset_throughput_available",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_distribution(
    values: Mapping[str, Any], keys: Sequence[str]
) -> Dict[str, float]:
    clean = {str(key): max(0.0, _safe_float(values.get(key, 0.0))) for key in keys}
    total = sum(clean.values())
    if total <= 0.0:
        fallback = {str(key): 0.0 for key in keys}
        if keys:
            fallback[str(keys[0])] = 1.0
        return fallback
    return {str(key): float(value / total) for key, value in clean.items()}


def _count_norm(value: Any, *, scale: float) -> float:
    return _clamp01(_safe_float(value, 0.0) / float(max(scale, 1.0)))


def heuristic_preset_distribution(
    focus_objective_presets: Sequence[str],
) -> Dict[str, float]:
    weights = {label: 0.0 for label in SHELL_POLICY_PRESET_LABELS}
    selected = [
        str(label) for label in (focus_objective_presets or []) if str(label) in weights
    ]
    if not selected:
        weights["balanced"] = 1.0
        return weights
    unit = 1.0 / float(len(selected))
    for label in selected:
        weights[label] += unit
    return weights


def normalize_strategy_overrides(values: Mapping[str, Any]) -> Dict[str, float]:
    return _normalize_distribution(values, SHELL_POLICY_STRATEGY_KEYS)


def build_shell_policy_feature_map(
    snapshot: SemanticSnapshot,
    *,
    trust_matrix: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    econ = snapshot.econ_slice
    meta = snapshot.meta_slice
    recap = (
        snapshot.metadata.get("recap", {})
        if isinstance(snapshot.metadata, Mapping)
        else {}
    )
    execution_summary: Mapping[str, Any] = {}
    if isinstance(snapshot.metadata, Mapping):
        execution_summary = (
            snapshot.metadata.get("execution_precondition_summary")
            or snapshot.metadata.get("execution_preconditions")
            or {}
        )
    execution_summary = (
        execution_summary if isinstance(execution_summary, Mapping) else {}
    )
    world_model = snapshot.semantic_world_model
    meta_node_weights: Dict[str, float] = {}
    capability_scores: Dict[str, float] = {}
    if world_model is not None:
        meta_node_weights = {
            str(node.node_type): _safe_float(getattr(node, "score", 0.0))
            for node in list(world_model.meta_nodes or [])
        }
        capability_scores = {
            str(key): _safe_float(value)
            for key, value in dict(world_model.capability_scores or {}).items()
        }
    presets = {str(item) for item in list(meta.presets or [])}
    expected_deltas = dict(meta.expected_deltas or {})
    ready_count = int(_safe_float(execution_summary.get("ready_count", 0)))
    blocked_count = int(_safe_float(execution_summary.get("blocked_count", 0)))
    top_episodes = list(recap.get("top_episodes", []) or [])
    ood_trust = 0.0
    if isinstance(trust_matrix, Mapping):
        ood_payload = trust_matrix.get("OODTag", {})
        if isinstance(ood_payload, Mapping):
            ood_trust = _safe_float(ood_payload.get("trust_score", 0.0))
    return {
        "avg_wage_parity": _clamp01(_safe_float(getattr(econ, "avg_wage_parity", 0.0))),
        "wage_gap": _clamp01(
            max(0.0, 1.0 - _safe_float(getattr(econ, "avg_wage_parity", 0.0)))
        ),
        "avg_energy_cost": _clamp01(_safe_float(getattr(econ, "avg_energy_cost", 0.0))),
        "avg_error_rate": _clamp01(_safe_float(getattr(econ, "avg_error_rate", 0.0))),
        "frontier_episode_count_norm": _count_norm(
            len(list(getattr(econ, "frontier_episodes", []) or [])), scale=12.0
        ),
        "recap_mean_goodness": _clamp01(
            _safe_float(recap.get("mean_goodness", 0.0), 0.0)
        ),
        "recap_top_episode_fraction": _count_norm(len(top_episodes), scale=6.0),
        "num_segments_norm": _count_norm(
            getattr(snapshot, "num_segments", 0), scale=12.0
        ),
        "recovery_segment_fraction": _clamp01(
            _safe_float(getattr(snapshot, "recovery_segment_fraction", 0.0))
        ),
        "mobility_drift_rate": _clamp01(
            _safe_float(getattr(snapshot, "mobility_drift_rate", 0.0))
        ),
        "blocked_count_norm": _count_norm(blocked_count, scale=6.0),
        "ready_count_norm": _count_norm(ready_count, scale=6.0),
        "mean_readiness_score": _clamp01(
            _safe_float(execution_summary.get("mean_readiness_score", 0.0))
        ),
        "max_ood_severity": _clamp01(
            _safe_float(
                snapshot.metadata.get(
                    "max_ood_severity", snapshot.metadata.get("ood_severity", 0.0)
                )
                if isinstance(snapshot.metadata, Mapping)
                else 0.0
            )
        ),
        "ood_trust": _clamp01(ood_trust),
        "risk_triage_score": _clamp01(meta_node_weights.get("risk_triage", 0.0)),
        "recovery_router_score": _clamp01(
            meta_node_weights.get("recovery_router", 0.0)
        ),
        "semantic_memory_refresh_score": _clamp01(
            meta_node_weights.get("semantic_memory_refresh", 0.0)
        ),
        "fusion_bridge_score": _clamp01(meta_node_weights.get("fusion_bridge", 0.0)),
        "ontology_router_score": _clamp01(
            meta_node_weights.get("ontology_router", 0.0)
        ),
        "task_graph_router_score": _clamp01(
            meta_node_weights.get("task_graph_router", 0.0)
        ),
        "efficiency_router_score": _clamp01(
            meta_node_weights.get("efficiency_router", 0.0)
        ),
        "risk_reasoning": _clamp01(capability_scores.get("risk_reasoning", 0.0)),
        "stage2_bridge": _clamp01(capability_scores.get("stage2_bridge", 0.0)),
        "fusion_bridge_capability": _clamp01(
            capability_scores.get("fusion_bridge", 0.0)
        ),
        "object_memory": _clamp01(capability_scores.get("object_memory", 0.0)),
        "affordance_grounding": _clamp01(
            capability_scores.get("affordance_grounding", 0.0)
        ),
        "meta_node_orchestration": _clamp01(
            capability_scores.get("meta_node_orchestration", 0.0)
        ),
        "expected_delta_mpl_norm": _clamp01(
            max(_safe_float(expected_deltas.get("mpl", 0.0), 0.0), 0.0) / 5.0
        ),
        "expected_delta_error_norm": _clamp01(
            abs(_safe_float(expected_deltas.get("error", 0.0), 0.0))
        ),
        "expected_delta_energy_norm": _clamp01(
            abs(_safe_float(expected_deltas.get("energy", 0.0), 0.0)) / 5.0
        ),
        "preset_balanced_available": 1.0 if "balanced" in presets else 0.0,
        "preset_safety_available": 1.0 if "safety" in presets else 0.0,
        "preset_energy_saver_available": 1.0 if "energy_saver" in presets else 0.0,
        "preset_throughput_available": 1.0 if "throughput" in presets else 0.0,
    }


def build_shell_policy_feature_vector(
    snapshot: SemanticSnapshot,
    *,
    trust_matrix: Optional[Mapping[str, Any]] = None,
) -> np.ndarray:
    feature_map = build_shell_policy_feature_map(snapshot, trust_matrix=trust_matrix)
    return np.asarray(
        [float(feature_map.get(name, 0.0)) for name in SHELL_POLICY_FEATURE_NAMES],
        dtype=np.float32,
    )


def extract_orchestrator_advisory_target(payload: Mapping[str, Any]) -> Dict[str, Any]:
    focus_presets = [
        str(item) for item in list(payload.get("focus_objective_presets", []) or [])
    ]
    strategy = normalize_strategy_overrides(
        payload.get("sampler_strategy_overrides", {}) or {}
    )
    execution_mode = str(payload.get("execution_mode", "advisory") or "advisory")
    promotion_stage = payload.get("promotion_stage")
    if promotion_stage is not None:
        promotion_stage = str(promotion_stage)
    return {
        "focus_objective_presets": focus_presets,
        "preset_distribution": heuristic_preset_distribution(focus_presets),
        "sampler_strategy_overrides": strategy,
        "safety_emphasis": _clamp01(_safe_float(payload.get("safety_emphasis", 0.0))),
        "execution_mode": execution_mode,
        "activation_label": 0.0 if execution_mode == "advisory" else 1.0,
        "policy_source": str(
            payload.get("policy_source", "heuristic_fallback") or "heuristic_fallback"
        ),
        "promotion_stage": promotion_stage or "heuristic_fallback",
        "activation_plan": dict(payload.get("activation_plan", {}) or {}),
    }


__all__ = [
    "SHELL_POLICY_FEATURE_NAMES",
    "SHELL_POLICY_PRESET_LABELS",
    "SHELL_POLICY_STRATEGY_KEYS",
    "build_shell_policy_feature_map",
    "build_shell_policy_feature_vector",
    "extract_orchestrator_advisory_target",
    "heuristic_preset_distribution",
    "normalize_strategy_overrides",
]

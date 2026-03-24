"""Shared semantic-world-model featurization and bounded-decision helpers for transformer shells."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from src.semantic.models import SemanticSnapshot
from src.utils.json_safe import to_json_safe
from src.world_model.semantic_world_model import SemanticWorldModelState


ORCHESTRATION_BASE_CTX_DIM = 36
SEMANTIC_WM_FEATURE_DIM = 20
ORCHESTRATION_CTX_DIM = ORCHESTRATION_BASE_CTX_DIM + SEMANTIC_WM_FEATURE_DIM


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    clean = {str(key): max(0.0, float(value)) for key, value in dict(weights).items()}
    total = sum(clean.values())
    if total <= 0.0:
        return {key: 0.0 for key in clean}
    return {key: value / total for key, value in clean.items()}


def coerce_semantic_world_model(
    semantic_world_model: Any = None,
    *,
    semantic_snapshot: Any = None,
    context: Any = None,
) -> Optional[SemanticWorldModelState]:
    candidates = [
        semantic_world_model,
        getattr(semantic_snapshot, "semantic_world_model", None),
        getattr(context, "semantic_world_model", None),
        getattr(context, "semantic_snapshot", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, SemanticWorldModelState):
            return candidate
        if isinstance(candidate, SemanticSnapshot):
            world_model = getattr(candidate, "semantic_world_model", None)
            if isinstance(world_model, SemanticWorldModelState):
                return world_model
            candidate = candidate.metadata.get("semantic_world_model")
        if isinstance(candidate, Mapping):
            payload = dict(candidate)
            if payload.get("version") == "semantic_world_model_v1" or "objects" in payload or "meta_nodes" in payload:
                try:
                    return SemanticWorldModelState.from_dict(payload)
                except Exception:
                    continue
            nested_payload = payload.get("semantic_world_model")
            if isinstance(nested_payload, Mapping):
                try:
                    return SemanticWorldModelState.from_dict(nested_payload)
                except Exception:
                    continue
    return None


def build_semantic_world_model_summary(
    semantic_world_model: Any = None,
    *,
    semantic_snapshot: Any = None,
    context: Any = None,
) -> Dict[str, Any]:
    world_model = coerce_semantic_world_model(
        semantic_world_model,
        semantic_snapshot=semantic_snapshot,
        context=context,
    )
    if world_model is None:
        metadata = getattr(context, "semantic_metadata", None)
        if isinstance(metadata, Mapping):
            summary_payload = metadata.get("semantic_world_model_summary")
            if isinstance(summary_payload, Mapping):
                payload = dict(to_json_safe(dict(summary_payload)))
                payload.setdefault("present", True)
                return payload
        return {
            "present": False,
            "world_model_id": "",
            "top_object_labels": [],
            "top_meta_nodes": [],
            "semantic_tags": [],
            "active_capabilities": [],
        }

    objects = list(world_model.objects or [])
    relations = list(world_model.relations or [])
    meta_nodes = list(world_model.meta_nodes or [])
    capabilities = dict(world_model.capability_scores or {})
    topology = dict(world_model.topology or {})
    object_count = len(objects)
    relation_count = len(relations)
    meta_node_count = len(meta_nodes)
    risk_object_count = sum(1 for item in objects if list(item.risk_tags or []))
    fragile_object_count = sum(
        1
        for item in objects
        if "fragile" in set(item.state_tags or []) or "fragility" in set(item.risk_tags or []) or "fragile" in item.category
    )
    affordance_density = (
        sum(len(item.affordances or []) for item in objects) / float(max(object_count, 1))
        if objects
        else 0.0
    )
    meta_node_scores = {
        item.node_type: float(item.score)
        for item in meta_nodes
    }
    high_priority_fraction = (
        sum(1 for item in meta_nodes if item.priority in {"high", "critical"}) / float(max(meta_node_count, 1))
        if meta_nodes
        else 0.0
    )
    capability_values = list(capabilities.values())
    top_object_labels = [
        item.label
        for item in sorted(objects, key=lambda row: (row.salience, row.confidence), reverse=True)[:4]
    ]
    top_meta_nodes = [
        item.node_type
        for item in sorted(meta_nodes, key=lambda row: row.score, reverse=True)[:4]
    ]
    active_capabilities = [
        key
        for key, value in sorted(capabilities.items(), key=lambda item: item[0])
        if float(value) >= 0.5
    ]
    return {
        "present": True,
        "world_model_id": world_model.world_model_id,
        "episode_id": world_model.episode_id,
        "task_id": world_model.task_id,
        "objective_preset": world_model.objective_preset,
        "object_count": object_count,
        "relation_count": relation_count,
        "meta_node_count": meta_node_count,
        "grounded_track_object_count": int(topology.get("grounded_track_object_count", 0) or 0),
        "relation_density": relation_count / float(max(object_count, 1)) if object_count else 0.0,
        "affordance_density": float(affordance_density),
        "risk_object_fraction": risk_object_count / float(max(object_count, 1)) if object_count else 0.0,
        "fragile_object_fraction": fragile_object_count / float(max(object_count, 1)) if object_count else 0.0,
        "priority_high_fraction": float(high_priority_fraction),
        "capability_mean": float(sum(capability_values) / float(max(len(capability_values), 1))),
        "capability_max": float(max(capability_values) if capability_values else 0.0),
        "risk_reasoning": float(capabilities.get("risk_reasoning", 0.0)),
        "object_memory": float(capabilities.get("object_memory", 0.0)),
        "affordance_grounding": float(capabilities.get("affordance_grounding", 0.0)),
        "fusion_bridge": float(capabilities.get("fusion_bridge", 0.0)),
        "stage2_bridge": float(capabilities.get("stage2_bridge", 0.0)),
        "meta_node_orchestration": float(capabilities.get("meta_node_orchestration", 0.0)),
        "risk_triage_score": float(meta_node_scores.get("risk_triage", 0.0)),
        "recovery_router_score": float(meta_node_scores.get("recovery_router", 0.0)),
        "efficiency_router_score": float(meta_node_scores.get("efficiency_router", 0.0)),
        "semantic_memory_refresh_score": float(meta_node_scores.get("semantic_memory_refresh", 0.0)),
        "ontology_router_score": float(meta_node_scores.get("ontology_router", 0.0)),
        "task_graph_router_score": float(meta_node_scores.get("task_graph_router", 0.0)),
        "top_object_labels": list(top_object_labels),
        "top_meta_nodes": list(top_meta_nodes),
        "semantic_tags": list(world_model.semantic_tags or [])[:8],
        "active_capabilities": list(active_capabilities),
        "topology": dict(topology),
        "capability_scores": dict(capabilities),
        "meta_node_scores": dict(sorted(meta_node_scores.items(), key=lambda item: item[0])),
    }


def encode_semantic_world_model_features(summary: Mapping[str, Any]) -> np.ndarray:
    payload = dict(summary or {})
    vector = np.array(
        [
            1.0 if payload.get("present") else 0.0,
            _safe_float(payload.get("capability_mean", 0.0)),
            _safe_float(payload.get("capability_max", 0.0)),
            min(_safe_float(payload.get("object_count", 0.0)) / 16.0, 1.0),
            min(_safe_float(payload.get("relation_density", 0.0)) / 4.0, 1.0),
            min(_safe_float(payload.get("grounded_track_object_count", 0.0)) / 8.0, 1.0),
            min(_safe_float(payload.get("affordance_density", 0.0)) / 4.0, 1.0),
            _safe_float(payload.get("risk_object_fraction", 0.0)),
            _safe_float(payload.get("fragile_object_fraction", 0.0)),
            _safe_float(payload.get("priority_high_fraction", 0.0)),
            _safe_float(payload.get("risk_reasoning", 0.0)),
            _safe_float(payload.get("object_memory", 0.0)),
            _safe_float(payload.get("affordance_grounding", 0.0)),
            _safe_float(payload.get("fusion_bridge", 0.0)),
            _safe_float(payload.get("stage2_bridge", 0.0)),
            _safe_float(payload.get("meta_node_orchestration", 0.0)),
            _safe_float(payload.get("risk_triage_score", 0.0)),
            _safe_float(payload.get("recovery_router_score", 0.0)),
            _safe_float(payload.get("efficiency_router_score", 0.0)),
            _safe_float(payload.get("semantic_memory_refresh_score", 0.0)),
        ],
        dtype=np.float32,
    )
    if vector.size < SEMANTIC_WM_FEATURE_DIM:
        vector = np.pad(vector, (0, SEMANTIC_WM_FEATURE_DIM - vector.size))
    return vector[:SEMANTIC_WM_FEATURE_DIM]


def semantic_tokens(summary: Mapping[str, Any]) -> List[str]:
    payload = dict(summary or {})
    tokens: List[str] = []
    for label in list(payload.get("top_object_labels", []) or []):
        tokens.append(f"object:{label}")
    for node_type in list(payload.get("top_meta_nodes", []) or []):
        tokens.append(f"meta_node:{node_type}")
    for capability in list(payload.get("active_capabilities", []) or []):
        tokens.append(f"capability:{capability}")
    for tag in list(payload.get("semantic_tags", []) or []):
        tokens.append(str(tag))
    ordered: List[str] = []
    seen = set()
    for token in tokens:
        normalized = str(token)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered[:16]


def derive_objective_preset(
    summary: Mapping[str, Any],
    *,
    econ_signals: Optional[Mapping[str, Any]] = None,
    datapack_signals: Optional[Mapping[str, Any]] = None,
    instruction: str = "",
) -> str:
    econ = dict(econ_signals or {})
    datapacks = dict(datapack_signals or {})
    text = instruction.lower()
    if (
        "safe" in text
        or "risk" in text
        or _safe_float(summary.get("risk_triage_score", 0.0)) >= 0.5
        or _safe_float(summary.get("risk_object_fraction", 0.0)) >= 0.3
        or _safe_float(econ.get("error_urgency", 0.0)) >= 0.45
    ):
        return "safety"
    if (
        "energy" in text
        or _safe_float(summary.get("efficiency_router_score", 0.0)) >= 0.45
        or _safe_float(econ.get("energy_urgency", 0.0)) >= 0.45
    ):
        return "energy_saver"
    if (
        "throughput" in text
        or "fast" in text
        or (
            _safe_float(econ.get("mpl_urgency", 0.0)) >= 0.45
            and _safe_float(summary.get("risk_object_fraction", 0.0)) < 0.25
            and _safe_float(datapacks.get("data_coverage_score", 0.0)) >= 0.2
        )
    ):
        return "throughput"
    return str(summary.get("objective_preset") or "balanced")


def derive_energy_profile_mix(
    summary: Mapping[str, Any],
    *,
    econ_signals: Optional[Mapping[str, Any]] = None,
    objective_preset: str = "balanced",
) -> Dict[str, float]:
    econ = dict(econ_signals or {})
    weights = {"BASE": 0.25, "BOOST": 0.25, "SAVER": 0.25, "SAFE": 0.25}
    weights["SAFE"] += 0.6 * _safe_float(summary.get("risk_triage_score", 0.0))
    weights["SAFE"] += 0.4 * _safe_float(summary.get("risk_object_fraction", 0.0))
    weights["SAVER"] += 0.6 * _safe_float(summary.get("efficiency_router_score", 0.0))
    weights["SAVER"] += 0.4 * _safe_float(econ.get("energy_urgency", 0.0))
    weights["BOOST"] += 0.5 * _safe_float(econ.get("mpl_urgency", 0.0))
    weights["BOOST"] -= 0.4 * _safe_float(summary.get("risk_object_fraction", 0.0))
    if objective_preset == "safety":
        weights["SAFE"] += 0.6
    elif objective_preset == "energy_saver":
        weights["SAVER"] += 0.6
    elif objective_preset == "throughput":
        weights["BOOST"] += 0.5
    else:
        weights["BASE"] += 0.2
    return _normalize_weights(weights)


def derive_data_mix_weights(
    summary: Mapping[str, Any],
    *,
    datapack_signals: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    datapacks = dict(datapack_signals or {})
    weights = {"real": 0.45, "synthetic": 0.35, "hybrid": 0.2}
    coverage = _safe_float(datapacks.get("data_coverage_score", 0.0))
    stage2_bridge = _safe_float(summary.get("stage2_bridge", 0.0))
    refresh = _safe_float(summary.get("semantic_memory_refresh_score", 0.0))
    fusion_bridge = _safe_float(summary.get("fusion_bridge", 0.0))
    if coverage >= 0.5 and fusion_bridge >= 0.5:
        weights["real"] += 0.25
        weights["hybrid"] += 0.1
    else:
        weights["synthetic"] += 0.2
        weights["hybrid"] += 0.1
    if stage2_bridge < 0.4:
        weights["synthetic"] += 0.2
    if refresh >= 0.45:
        weights["hybrid"] += 0.2
    vla_fraction = _safe_float(datapacks.get("vla_annotation_fraction", 0.0))
    if vla_fraction >= 0.5:
        weights["real"] += 0.15
    return _normalize_weights(weights)


def derive_backend(
    summary: Mapping[str, Any],
    *,
    econ_signals: Optional[Mapping[str, Any]] = None,
    current_backend: str = "pybullet",
) -> str:
    econ = dict(econ_signals or {})
    if _safe_float(econ.get("energy_urgency", 0.0)) >= 0.55 and current_backend != "isaac":
        return "isaac"
    if _safe_float(summary.get("fusion_bridge", 0.0)) < 0.25:
        return current_backend or "pybullet"
    return current_backend or "pybullet"


def estimate_expected_deltas(
    summary: Mapping[str, Any],
    *,
    econ_signals: Optional[Mapping[str, Any]] = None,
    datapack_signals: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    econ = dict(econ_signals or {})
    datapacks = dict(datapack_signals or {})
    capability_mean = _safe_float(summary.get("capability_mean", 0.0))
    coverage = _safe_float(datapacks.get("data_coverage_score", 0.0))
    risk_penalty = _safe_float(summary.get("risk_object_fraction", 0.0))
    mpl_delta = 2.5 * capability_mean + 1.5 * coverage + 2.0 * _safe_float(econ.get("mpl_urgency", 0.0)) - 1.5 * risk_penalty
    error_delta = -(
        0.8 * _safe_float(summary.get("risk_triage_score", 0.0))
        + 0.6 * _safe_float(summary.get("risk_reasoning", 0.0))
        + 0.3 * _safe_float(econ.get("error_urgency", 0.0))
    )
    energy_delta = (
        1.5 * _safe_float(summary.get("efficiency_router_score", 0.0))
        + 0.7 * _safe_float(econ.get("energy_urgency", 0.0))
        - 0.4 * coverage
    )
    return {
        "expected_delta_mpl": float(mpl_delta),
        "expected_delta_error": float(error_delta),
        "expected_delta_energy_Wh": float(energy_delta),
    }


def build_semantic_orchestration_plan(
    summary: Mapping[str, Any],
    *,
    objective_preset: str,
    data_mix_weights: Mapping[str, float],
    energy_profile_weights: Mapping[str, float],
    datapack_signals: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    datapacks = dict(datapack_signals or {})
    steps: List[Dict[str, Any]] = [
        {
            "action": "set_objective_preset",
            "preset": objective_preset,
            "reason": "semantic_world_model_routing",
        },
        {
            "action": "set_energy_profile",
            "profile_mix": dict(_normalize_weights(energy_profile_weights)),
            "reason": "semantic_meta_node_balance",
        },
        {
            "action": "set_data_mix",
            "data_mix": dict(_normalize_weights(data_mix_weights)),
            "reason": "semantic_grounding_and_coverage",
        },
    ]
    if _safe_float(summary.get("semantic_memory_refresh_score", 0.0)) >= 0.45:
        steps.append(
            {
                "action": "refresh_semantic_memory",
                "targets": list(summary.get("top_object_labels", []) or []),
                "reason": "semantic_memory_refresh_meta_node",
            }
        )
    if _safe_float(summary.get("stage2_bridge", 0.0)) < 0.4:
        steps.append(
            {
                "action": "request_stage2_enrichment",
                "focus": list(datapacks.get("data_gaps", []) or []),
                "reason": "stage2_bridge_gap",
            }
        )
    if _safe_float(summary.get("risk_triage_score", 0.0)) >= 0.45:
        steps.append(
            {
                "action": "route_risk_triage",
                "targets": list(summary.get("top_object_labels", []) or []),
                "reason": "risk_triage_meta_node",
            }
        )
    return steps[:6]


def build_tool_biases(
    summary: Mapping[str, Any],
    *,
    econ_signals: Optional[Mapping[str, Any]] = None,
    datapack_signals: Optional[Mapping[str, Any]] = None,
    instruction: str = "",
) -> Dict[str, float]:
    econ = dict(econ_signals or {})
    datapacks = dict(datapack_signals or {})
    text = instruction.lower()
    biases = {
        "SET_ENERGY_PROFILE": 0.4,
        "SET_OBJECTIVE_PRESET": 0.5,
        "SET_BACKEND": 0.2,
        "SET_DATA_MIX": 0.3,
        "QUERY_DATAPACKS": 0.0,
        "QUERY_ENERGY_SURFACE": 0.0,
        "CALL_VLA_SINGLE_STEP": 0.0,
        "CALL_VLA_FOR_DATAPACK_CLASS": 0.0,
    }
    biases["SET_OBJECTIVE_PRESET"] += 1.2 * _safe_float(summary.get("risk_triage_score", 0.0))
    biases["SET_ENERGY_PROFILE"] += 1.0 * _safe_float(summary.get("efficiency_router_score", 0.0))
    biases["SET_DATA_MIX"] += 0.9 * (1.0 - _safe_float(datapacks.get("data_coverage_score", 0.0)))
    biases["QUERY_DATAPACKS"] += 1.0 * max(0.0, 0.5 - _safe_float(summary.get("stage2_bridge", 0.0)))
    biases["QUERY_ENERGY_SURFACE"] += 0.8 * _safe_float(econ.get("energy_urgency", 0.0))
    biases["CALL_VLA_FOR_DATAPACK_CLASS"] += 0.9 * max(0.0, 0.5 - _safe_float(summary.get("fusion_bridge", 0.0)))
    biases["CALL_VLA_SINGLE_STEP"] += 0.8 * _safe_float(summary.get("recovery_router_score", 0.0))
    if "energy" in text:
        biases["QUERY_ENERGY_SURFACE"] += 0.5
        biases["SET_ENERGY_PROFILE"] += 0.4
    if "backend" in text:
        biases["SET_BACKEND"] += 0.6
    if "safety" in text or "risk" in text:
        biases["SET_OBJECTIVE_PRESET"] += 0.6
        biases["SET_ENERGY_PROFILE"] += 0.3
    if "data" in text or "collect" in text:
        biases["QUERY_DATAPACKS"] += 0.5
        biases["SET_DATA_MIX"] += 0.5
    return dict(sorted(biases.items(), key=lambda item: item[0]))


__all__ = [
    "ORCHESTRATION_BASE_CTX_DIM",
    "ORCHESTRATION_CTX_DIM",
    "SEMANTIC_WM_FEATURE_DIM",
    "build_semantic_orchestration_plan",
    "build_semantic_world_model_summary",
    "build_tool_biases",
    "coerce_semantic_world_model",
    "derive_backend",
    "derive_data_mix_weights",
    "derive_energy_profile_mix",
    "derive_objective_preset",
    "encode_semantic_world_model_features",
    "estimate_expected_deltas",
    "semantic_tokens",
]

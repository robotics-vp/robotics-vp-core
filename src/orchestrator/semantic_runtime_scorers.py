"""Lightweight runtime scorers and shadow reranking for semantic transformer lanes."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.learning.calibration import summarize_calibration
from src.orchestrator.semantic_runtime_learning import (
    SemanticRuntimeCounterfactual,
    SemanticRuntimeLearningCorpus,
    SemanticRuntimeLearningRow,
)
from src.orchestrator.semantic_transformer_bridge import build_semantic_world_model_summary
from src.semantic.models import SemanticSnapshot
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.world_model.semantic_world_model import SemanticWorldModelState


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "ready", "success"}
    return bool(value)


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _normalize_backend(value: Any) -> str:
    backend = str(value or "").strip().lower()
    if backend in {"isaac", "nvidia_isaac"}:
        return "isaac"
    if backend in {"workcell", "mujoco", "sim_workcell"}:
        return "workcell"
    return "pybullet"


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _mean_abs_error(predictions: Sequence[float], targets: Sequence[float]) -> float:
    if not predictions or len(predictions) != len(targets):
        return 1.0
    return float(sum(abs(float(pred) - float(target)) for pred, target in zip(predictions, targets)) / len(predictions))


BASE_FEATURE_NAMES: tuple[str, ...] = (
    "semantic_present",
    "object_count_norm",
    "relation_density",
    "affordance_density_norm",
    "risk_object_fraction",
    "fragile_object_fraction",
    "priority_high_fraction",
    "capability_mean",
    "capability_max",
    "risk_reasoning",
    "object_memory",
    "affordance_grounding",
    "fusion_bridge",
    "stage2_bridge",
    "meta_node_orchestration",
    "risk_triage_score",
    "recovery_router_score",
    "efficiency_router_score",
    "semantic_memory_refresh_score",
    "vla_available",
    "vla_confidence_mean",
    "teacher_trace_available",
    "teacher_confidence_mean",
    "teacher_object_ref_count_norm",
    "teacher_affordance_count_norm",
    "teacher_risk_count_norm",
    "dino_proxy_available",
    "dino_proxy_confidence_mean",
    "scene_tracks_available",
    "scene_track_count_norm",
    "scene_track_label_confidence_mean",
    "map_first_available",
    "map_first_confidence_mean",
    "fusion_available",
    "fusion_confidence_mean",
    "annotation_agreement_score",
    "fusion_advantage_score",
    "source_confidence_gap",
    "route_confidence_gap",
    "authority_vla_confidence",
    "authority_dino_confidence",
    "expected_delta_mpl_norm",
    "expected_delta_error_norm",
    "expected_delta_energy_norm",
    "execution_ready",
    "work_order_ready",
    "semantic_grounded",
    "scene_tracks_non_stub",
    "teacher_runtime_live",
    "readiness_score",
    "quality_score",
    "reward_signal",
    "objective_balanced",
    "objective_safety",
    "objective_energy_saver",
    "objective_throughput",
    "backend_pybullet",
    "backend_isaac",
    "backend_workcell",
)

META_ROUTE_FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + (
    "meta_lane",
    "meta_plan_length_norm",
    "meta_bounded_action_count_norm",
)

ORCHESTRATION_ROUTE_FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + (
    "orchestration_lane",
    "tool_count_norm",
    "tool_query_fraction",
    "tool_vla_fraction",
    "tool_backend_fraction",
    "activation_mode_bounded",
)

AUTHORITY_FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + (
    "authority_vla",
    "authority_dino",
    "authority_confidence",
    "authority_margin",
    "authority_annotation_support",
    "authority_grounding_support",
)

COUNTERFACTUAL_FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + (
    "candidate_balanced",
    "candidate_safety",
    "candidate_energy_saver",
    "candidate_throughput",
    "candidate_authority_vla",
    "candidate_authority_dino",
    "candidate_executable",
    "candidate_baseline_score",
    "candidate_baseline_regret",
    "candidate_authority_swap",
)


@dataclass(frozen=True)
class SemanticRuntimeLinearModel:
    target_name: str
    task: str
    feature_names: List[str]
    feature_means: List[float]
    feature_stds: List[float]
    weights: List[float]
    bias: float
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "task": self.task,
            "feature_names": list(self.feature_names),
            "feature_means": list(self.feature_means),
            "feature_stds": list(self.feature_stds),
            "weights": list(self.weights),
            "bias": float(self.bias),
            "training_metrics": _mapping(self.training_metrics),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticRuntimeLinearModel":
        return cls(
            target_name=str(payload.get("target_name", "")),
            task=str(payload.get("task", "probability")),
            feature_names=[str(item) for item in payload.get("feature_names", []) or []],
            feature_means=[float(item) for item in payload.get("feature_means", []) or []],
            feature_stds=[float(item) for item in payload.get("feature_stds", []) or []],
            weights=[float(item) for item in payload.get("weights", []) or []],
            bias=float(payload.get("bias", 0.0)),
            training_metrics=dict(payload.get("training_metrics", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def predict_raw(self, feature_map: Mapping[str, Any]) -> float:
        if not self.feature_names:
            return float(self.bias)
        raw = np.array([_safe_float(feature_map.get(name, 0.0)) for name in self.feature_names], dtype=np.float64)
        means = np.asarray(self.feature_means, dtype=np.float64)
        stds = np.asarray(self.feature_stds, dtype=np.float64)
        stds = np.where(stds <= 1e-6, 1.0, stds)
        normalized = (raw - means) / stds
        weights = np.asarray(self.weights, dtype=np.float64)
        return float(np.dot(normalized, weights) + float(self.bias))

    def predict(self, feature_map: Mapping[str, Any]) -> float:
        raw = self.predict_raw(feature_map)
        if self.task == "probability":
            return float(_sigmoid(np.asarray([raw], dtype=np.float64))[0])
        return float(raw)


@dataclass(frozen=True)
class SemanticRuntimeCounterfactualScore:
    counterfactual_id: str
    rationale: str
    rescored_value: float
    baseline_score: float
    executable: bool
    candidate: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counterfactual_id": self.counterfactual_id,
            "rationale": self.rationale,
            "rescored_value": float(self.rescored_value),
            "baseline_score": float(self.baseline_score),
            "executable": bool(self.executable),
            "candidate": _mapping(self.candidate),
        }


@dataclass(frozen=True)
class SemanticRuntimeScoreResult:
    score_id: str
    semantic_world_model_id: str
    meta_route_success_probability: float
    orchestration_route_success_probability: float
    predicted_regret: float
    preferred_authority: str
    calibrated_authority: str
    chosen_authority_confidence: float
    alternate_authority_confidence: float
    authority_switch_recommended: bool
    counterfactual_scores: List[SemanticRuntimeCounterfactualScore] = field(default_factory=list)
    feedback_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_id": self.score_id,
            "semantic_world_model_id": self.semantic_world_model_id,
            "meta_route_success_probability": float(self.meta_route_success_probability),
            "orchestration_route_success_probability": float(self.orchestration_route_success_probability),
            "predicted_regret": float(self.predicted_regret),
            "preferred_authority": self.preferred_authority,
            "calibrated_authority": self.calibrated_authority,
            "chosen_authority_confidence": float(self.chosen_authority_confidence),
            "alternate_authority_confidence": float(self.alternate_authority_confidence),
            "authority_switch_recommended": bool(self.authority_switch_recommended),
            "counterfactual_scores": [item.to_dict() for item in self.counterfactual_scores],
            "feedback_summary": _mapping(self.feedback_summary),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class SemanticRuntimeScorerPackage:
    version: str
    meta_route_success_model: SemanticRuntimeLinearModel
    orchestration_route_success_model: SemanticRuntimeLinearModel
    authority_calibration_model: SemanticRuntimeLinearModel
    counterfactual_value_model: SemanticRuntimeLinearModel
    regret_model: SemanticRuntimeLinearModel
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "meta_route_success_model": self.meta_route_success_model.to_dict(),
            "orchestration_route_success_model": self.orchestration_route_success_model.to_dict(),
            "authority_calibration_model": self.authority_calibration_model.to_dict(),
            "counterfactual_value_model": self.counterfactual_value_model.to_dict(),
            "regret_model": self.regret_model.to_dict(),
            "summary": _mapping(self.summary),
            "metadata": _mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticRuntimeScorerPackage":
        return cls(
            version=str(payload.get("version", "semantic_runtime_scorers_v1")),
            meta_route_success_model=SemanticRuntimeLinearModel.from_dict(payload.get("meta_route_success_model", {})),
            orchestration_route_success_model=SemanticRuntimeLinearModel.from_dict(
                payload.get("orchestration_route_success_model", {})
            ),
            authority_calibration_model=SemanticRuntimeLinearModel.from_dict(
                payload.get("authority_calibration_model", {})
            ),
            counterfactual_value_model=SemanticRuntimeLinearModel.from_dict(
                payload.get("counterfactual_value_model", {})
            ),
            regret_model=SemanticRuntimeLinearModel.from_dict(payload.get("regret_model", {})),
            summary=dict(payload.get("summary", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


def coerce_semantic_runtime_scorer_package(
    package: Any,
) -> Optional[SemanticRuntimeScorerPackage]:
    if package is None:
        return None
    if isinstance(package, SemanticRuntimeScorerPackage):
        return package
    if isinstance(package, Mapping):
        return SemanticRuntimeScorerPackage.from_dict(package)
    return None


def _base_feature_map(
    semantic_summary: Mapping[str, Any],
    vla_summary: Mapping[str, Any],
    dino_summary: Mapping[str, Any],
    fusion_summary: Mapping[str, Any],
    meta_target: Mapping[str, Any],
    orchestration_target: Mapping[str, Any],
    outcome_summary: Mapping[str, Any],
) -> Dict[str, float]:
    objective = str(
        meta_target.get("objective_preset")
        or orchestration_target.get("objective_preset")
        or semantic_summary.get("objective_preset")
        or "balanced"
    )
    backend = _normalize_backend(
        orchestration_target.get("chosen_backend")
        or meta_target.get("chosen_backend")
        or "pybullet"
    )
    return {
        "semantic_present": 1.0 if semantic_summary.get("present", True) else 0.0,
        "object_count_norm": min(_safe_float(semantic_summary.get("object_count", 0.0)) / 16.0, 2.0),
        "relation_density": _safe_float(semantic_summary.get("relation_density", 0.0)),
        "affordance_density_norm": min(_safe_float(semantic_summary.get("affordance_density", 0.0)) / 4.0, 2.0),
        "risk_object_fraction": _safe_float(semantic_summary.get("risk_object_fraction", 0.0)),
        "fragile_object_fraction": _safe_float(semantic_summary.get("fragile_object_fraction", 0.0)),
        "priority_high_fraction": _safe_float(semantic_summary.get("priority_high_fraction", 0.0)),
        "capability_mean": _safe_float(semantic_summary.get("capability_mean", 0.0)),
        "capability_max": _safe_float(semantic_summary.get("capability_max", 0.0)),
        "risk_reasoning": _safe_float(semantic_summary.get("risk_reasoning", 0.0)),
        "object_memory": _safe_float(semantic_summary.get("object_memory", 0.0)),
        "affordance_grounding": _safe_float(semantic_summary.get("affordance_grounding", 0.0)),
        "fusion_bridge": _safe_float(semantic_summary.get("fusion_bridge", 0.0)),
        "stage2_bridge": _safe_float(semantic_summary.get("stage2_bridge", 0.0)),
        "meta_node_orchestration": _safe_float(semantic_summary.get("meta_node_orchestration", 0.0)),
        "risk_triage_score": _safe_float(semantic_summary.get("risk_triage_score", 0.0)),
        "recovery_router_score": _safe_float(semantic_summary.get("recovery_router_score", 0.0)),
        "efficiency_router_score": _safe_float(semantic_summary.get("efficiency_router_score", 0.0)),
        "semantic_memory_refresh_score": _safe_float(semantic_summary.get("semantic_memory_refresh_score", 0.0)),
        "vla_available": 1.0 if vla_summary.get("vla_available", False) else 0.0,
        "vla_confidence_mean": _safe_float(vla_summary.get("vla_confidence_mean", 0.0)),
        "teacher_trace_available": 1.0 if vla_summary.get("teacher_trace_available", False) else 0.0,
        "teacher_confidence_mean": _safe_float(vla_summary.get("teacher_confidence_mean", 0.0)),
        "teacher_object_ref_count_norm": min(len(vla_summary.get("teacher_object_refs", []) or []) / 8.0, 2.0),
        "teacher_affordance_count_norm": min(len(vla_summary.get("teacher_affordance_hints", []) or []) / 8.0, 2.0),
        "teacher_risk_count_norm": min(len(vla_summary.get("teacher_risk_hints", []) or []) / 8.0, 2.0),
        "dino_proxy_available": 1.0 if dino_summary.get("dino_proxy_available", False) else 0.0,
        "dino_proxy_confidence_mean": _safe_float(dino_summary.get("dino_proxy_confidence_mean", 0.0)),
        "scene_tracks_available": 1.0 if dino_summary.get("scene_tracks_available", False) else 0.0,
        "scene_track_count_norm": min(_safe_float(dino_summary.get("scene_track_count", 0.0)) / 8.0, 2.0),
        "scene_track_label_confidence_mean": _safe_float(dino_summary.get("scene_track_label_confidence_mean", 0.0)),
        "map_first_available": 1.0 if dino_summary.get("map_first_available", False) else 0.0,
        "map_first_confidence_mean": _safe_float(dino_summary.get("map_first_confidence_mean", 0.0)),
        "fusion_available": 1.0 if fusion_summary.get("fusion_available", False) else 0.0,
        "fusion_confidence_mean": _safe_float(fusion_summary.get("semantic_fusion_confidence_mean", 0.0)),
        "annotation_agreement_score": _safe_float(fusion_summary.get("annotation_agreement_score", 0.0)),
        "fusion_advantage_score": _safe_float(fusion_summary.get("fusion_advantage_score", 0.0)),
        "source_confidence_gap": _safe_float(fusion_summary.get("source_confidence_gap", 0.0)),
        "route_confidence_gap": abs(
            _safe_float(meta_target.get("confidence_vla", 0.0)) - _safe_float(meta_target.get("confidence_dino", 0.0))
        ),
        "authority_vla_confidence": _safe_float(meta_target.get("confidence_vla", 0.0)),
        "authority_dino_confidence": _safe_float(meta_target.get("confidence_dino", 0.0)),
        "expected_delta_mpl_norm": max(_safe_float(meta_target.get("expected_deltas", {}).get("expected_delta_mpl", 0.0)), 0.0) / 8.0,
        "expected_delta_error_norm": min(abs(_safe_float(meta_target.get("expected_deltas", {}).get("expected_delta_error", 0.0))), 2.0),
        "expected_delta_energy_norm": min(
            abs(_safe_float(meta_target.get("expected_deltas", {}).get("expected_delta_energy_Wh", 0.0))) / 8.0,
            2.0,
        ),
        "execution_ready": 1.0 if outcome_summary.get("execution_ready", False) else 0.0,
        "work_order_ready": 1.0 if outcome_summary.get("work_order_ready", False) else 0.0,
        "semantic_grounded": 1.0 if outcome_summary.get("semantic_grounded", False) else 0.0,
        "scene_tracks_non_stub": 1.0 if outcome_summary.get("scene_tracks_non_stub", False) else 0.0,
        "teacher_runtime_live": 1.0 if outcome_summary.get("teacher_runtime_live", False) else 0.0,
        "readiness_score": _safe_float(outcome_summary.get("readiness_score", 0.0)),
        "quality_score": _safe_float(outcome_summary.get("quality_score", 0.0)),
        "reward_signal": _safe_float(outcome_summary.get("reward_signal", 0.0)),
        "objective_balanced": 1.0 if objective == "balanced" else 0.0,
        "objective_safety": 1.0 if objective == "safety" else 0.0,
        "objective_energy_saver": 1.0 if objective == "energy_saver" else 0.0,
        "objective_throughput": 1.0 if objective == "throughput" else 0.0,
        "backend_pybullet": 1.0 if backend == "pybullet" else 0.0,
        "backend_isaac": 1.0 if backend == "isaac" else 0.0,
        "backend_workcell": 1.0 if backend == "workcell" else 0.0,
    }


def _meta_route_feature_map(row: SemanticRuntimeLearningRow) -> Dict[str, float]:
    feature_map = _base_feature_map(
        row.semantic_world_model_summary,
        row.vla_summary,
        row.dino_summary,
        row.fusion_summary,
        row.meta_transformer_target,
        row.orchestration_transformer_target,
        row.outcome_summary,
    )
    feature_map.update(
        {
            "meta_lane": 1.0,
            "meta_plan_length_norm": min(len(row.meta_transformer_target.get("plan", []) or []) / 8.0, 2.0),
            "meta_bounded_action_count_norm": min(
                len(row.meta_transformer_target.get("bounded_actions", []) or []) / 8.0,
                2.0,
            ),
        }
    )
    return feature_map


def _orchestration_route_feature_map(row: SemanticRuntimeLearningRow) -> Dict[str, float]:
    feature_map = _base_feature_map(
        row.semantic_world_model_summary,
        row.vla_summary,
        row.dino_summary,
        row.fusion_summary,
        row.meta_transformer_target,
        row.orchestration_transformer_target,
        row.outcome_summary,
    )
    tool_sequence = list(row.orchestration_transformer_target.get("tool_sequence", []) or [])
    tool_names = [str(item.get("name", "")) for item in tool_sequence]
    query_count = sum(1 for name in tool_names if name.startswith("QUERY_"))
    vla_count = sum(1 for name in tool_names if name.startswith("CALL_VLA_"))
    backend_count = sum(1 for name in tool_names if name == "SET_BACKEND")
    feature_map.update(
        {
            "orchestration_lane": 1.0,
            "tool_count_norm": min(len(tool_sequence) / 8.0, 2.0),
            "tool_query_fraction": query_count / float(max(len(tool_sequence), 1)),
            "tool_vla_fraction": vla_count / float(max(len(tool_sequence), 1)),
            "tool_backend_fraction": backend_count / float(max(len(tool_sequence), 1)),
            "activation_mode_bounded": 1.0
            if str(row.orchestration_transformer_target.get("execution_mode", "advisory")) == "bounded_execution"
            else 0.0,
        }
    )
    return feature_map


def derive_authority_success_label(row: SemanticRuntimeLearningRow, authority: Optional[str] = None) -> bool:
    chosen_authority = str(authority or row.meta_transformer_target.get("authority_gt", "dino"))
    route_success = _bool(row.inferential_summary.get("route_success_label", False))
    if chosen_authority == "vla":
        return bool(
            route_success
            and row.vla_summary.get("vla_available", False)
            and (
                _safe_float(row.fusion_summary.get("annotation_agreement_score", 0.0)) >= 0.35
                or _safe_float(row.semantic_world_model_summary.get("affordance_grounding", 0.0)) >= 0.45
            )
        )
    return bool(
        route_success
        and row.dino_summary.get("dino_proxy_available", False)
        and row.outcome_summary.get("semantic_grounded", False)
    )


def _authority_feature_map(row: SemanticRuntimeLearningRow, authority: Optional[str] = None) -> Dict[str, float]:
    chosen_authority = str(authority or row.meta_transformer_target.get("authority_gt", "dino"))
    feature_map = _base_feature_map(
        row.semantic_world_model_summary,
        row.vla_summary,
        row.dino_summary,
        row.fusion_summary,
        row.meta_transformer_target,
        row.orchestration_transformer_target,
        row.outcome_summary,
    )
    if chosen_authority == "vla":
        authority_confidence = _safe_float(row.meta_transformer_target.get("confidence_vla", 0.0))
        grounding_support = _safe_float(row.semantic_world_model_summary.get("affordance_grounding", 0.0))
        annotation_support = _safe_float(row.fusion_summary.get("annotation_agreement_score", 0.0))
        margin = _safe_float(row.meta_transformer_target.get("confidence_vla", 0.0)) - _safe_float(
            row.meta_transformer_target.get("confidence_dino", 0.0)
        )
    else:
        authority_confidence = _safe_float(row.meta_transformer_target.get("confidence_dino", 0.0))
        grounding_support = _safe_float(row.semantic_world_model_summary.get("object_memory", 0.0))
        annotation_support = 1.0 if row.outcome_summary.get("semantic_grounded", False) else 0.0
        margin = _safe_float(row.meta_transformer_target.get("confidence_dino", 0.0)) - _safe_float(
            row.meta_transformer_target.get("confidence_vla", 0.0)
        )
    feature_map.update(
        {
            "authority_vla": 1.0 if chosen_authority == "vla" else 0.0,
            "authority_dino": 1.0 if chosen_authority == "dino" else 0.0,
            "authority_confidence": authority_confidence,
            "authority_margin": margin,
            "authority_annotation_support": annotation_support,
            "authority_grounding_support": grounding_support,
        }
    )
    return feature_map


def _counterfactual_feature_map(
    row: SemanticRuntimeLearningRow,
    counterfactual: SemanticRuntimeCounterfactual,
) -> Dict[str, float]:
    candidate = dict(counterfactual.candidate or {})
    preset = str(candidate.get("objective_preset", ""))
    authority = str(candidate.get("authority_gt", ""))
    feature_map = _base_feature_map(
        row.semantic_world_model_summary,
        row.vla_summary,
        row.dino_summary,
        row.fusion_summary,
        row.meta_transformer_target,
        row.orchestration_transformer_target,
        row.outcome_summary,
    )
    feature_map.update(
        {
            "candidate_balanced": 1.0 if preset == "balanced" else 0.0,
            "candidate_safety": 1.0 if preset == "safety" else 0.0,
            "candidate_energy_saver": 1.0 if preset == "energy_saver" else 0.0,
            "candidate_throughput": 1.0 if preset == "throughput" else 0.0,
            "candidate_authority_vla": 1.0 if authority == "vla" else 0.0,
            "candidate_authority_dino": 1.0 if authority == "dino" else 0.0,
            "candidate_executable": 1.0 if counterfactual.executable else 0.0,
            "candidate_baseline_score": float(counterfactual.predicted_outcome_score),
            "candidate_baseline_regret": float(counterfactual.predicted_regret),
            "candidate_authority_swap": 1.0 if "authority" in counterfactual.rationale else 0.0,
        }
    )
    return feature_map


def _matrix_from_feature_maps(
    feature_maps: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> np.ndarray:
    if not feature_maps:
        return np.zeros((0, len(feature_names)), dtype=np.float64)
    return np.asarray(
        [
            [_safe_float(feature_map.get(name, 0.0)) for name in feature_names]
            for feature_map in feature_maps
        ],
        dtype=np.float64,
    )


def _fit_probability_model(
    feature_maps: Sequence[Mapping[str, Any]],
    targets: Sequence[float],
    *,
    feature_names: Sequence[str],
    target_name: str,
    metadata: Optional[Mapping[str, Any]] = None,
    iterations: int = 300,
    learning_rate: float = 0.15,
    l2: float = 0.05,
) -> SemanticRuntimeLinearModel:
    X = _matrix_from_feature_maps(feature_maps, feature_names)
    y = np.asarray([_safe_float(item) for item in targets], dtype=np.float64)
    if X.shape[0] == 0:
        return SemanticRuntimeLinearModel(
            target_name=target_name,
            task="probability",
            feature_names=list(feature_names),
            feature_means=[0.0 for _ in feature_names],
            feature_stds=[1.0 for _ in feature_names],
            weights=[0.0 for _ in feature_names],
            bias=0.0,
            training_metrics={"sample_count": 0},
            metadata=dict(metadata or {}),
        )
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds = np.where(stds <= 1e-6, 1.0, stds)
    normalized = (X - means) / stds
    positive_rate = float(np.mean(y)) if y.size else 0.0
    initial = min(max(positive_rate, 1e-5), 1.0 - 1e-5)
    bias = float(np.log(initial / (1.0 - initial)))
    weights = np.zeros(normalized.shape[1], dtype=np.float64)
    if np.unique(y).size > 1:
        step_size = float(learning_rate / max(1.0, np.sqrt(normalized.shape[1])))
        for _ in range(max(int(iterations), 1)):
            logits = normalized @ weights + bias
            probs = _sigmoid(logits)
            error = probs - y
            weights -= step_size * ((normalized.T @ error) / float(len(y)) + l2 * weights)
            bias -= step_size * float(np.mean(error))
    predictions = _sigmoid(normalized @ weights + bias)
    calibration = summarize_calibration(
        confidences=predictions.tolist(),
        outcomes=y.tolist(),
        predictions=predictions.tolist(),
        targets=y.tolist(),
        monotonic_inputs=[_safe_float(item.get("quality_score", 0.0)) for item in feature_maps],
        monotonic_outputs=predictions.tolist(),
        reference_vectors=normalized.tolist(),
        current_vectors=normalized.tolist(),
        metadata={"target_name": target_name},
    ).to_dict()
    metrics = {
        "sample_count": int(len(y)),
        "positive_rate": positive_rate,
        "prediction_mean": float(np.mean(predictions)),
        "accuracy": float(np.mean((predictions >= 0.5) == (y >= 0.5))),
        "calibration": calibration,
    }
    return SemanticRuntimeLinearModel(
        target_name=target_name,
        task="probability",
        feature_names=list(feature_names),
        feature_means=means.astype(np.float64).tolist(),
        feature_stds=stds.astype(np.float64).tolist(),
        weights=weights.astype(np.float64).tolist(),
        bias=float(bias),
        training_metrics=metrics,
        metadata=dict(metadata or {}),
    )


def _fit_regression_model(
    feature_maps: Sequence[Mapping[str, Any]],
    targets: Sequence[float],
    *,
    feature_names: Sequence[str],
    target_name: str,
    metadata: Optional[Mapping[str, Any]] = None,
    l2: float = 0.05,
) -> SemanticRuntimeLinearModel:
    X = _matrix_from_feature_maps(feature_maps, feature_names)
    y = np.asarray([_safe_float(item) for item in targets], dtype=np.float64)
    if X.shape[0] == 0:
        return SemanticRuntimeLinearModel(
            target_name=target_name,
            task="regression",
            feature_names=list(feature_names),
            feature_means=[0.0 for _ in feature_names],
            feature_stds=[1.0 for _ in feature_names],
            weights=[0.0 for _ in feature_names],
            bias=0.0,
            training_metrics={"sample_count": 0},
            metadata=dict(metadata or {}),
        )
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds = np.where(stds <= 1e-6, 1.0, stds)
    normalized = (X - means) / stds
    if normalized.shape[0] == 1:
        weights = np.zeros(normalized.shape[1], dtype=np.float64)
        bias = float(y[0])
        predictions = np.asarray([bias], dtype=np.float64)
    else:
        augmented = np.concatenate([normalized, np.ones((normalized.shape[0], 1), dtype=np.float64)], axis=1)
        regularizer = np.eye(augmented.shape[1], dtype=np.float64) * float(l2)
        regularizer[-1, -1] = 0.0
        solution = np.linalg.solve(augmented.T @ augmented + regularizer, augmented.T @ y)
        weights = solution[:-1]
        bias = float(solution[-1])
        predictions = normalized @ weights + bias
    metrics = {
        "sample_count": int(len(y)),
        "target_mean": float(np.mean(y)) if y.size else 0.0,
        "prediction_mean": float(np.mean(predictions)) if predictions.size else 0.0,
        "mae": _mean_abs_error(predictions.tolist(), y.tolist()),
        "rmse": float(np.sqrt(np.mean((predictions - y) ** 2))) if predictions.size else 0.0,
    }
    return SemanticRuntimeLinearModel(
        target_name=target_name,
        task="regression",
        feature_names=list(feature_names),
        feature_means=means.astype(np.float64).tolist(),
        feature_stds=stds.astype(np.float64).tolist(),
        weights=weights.astype(np.float64).tolist(),
        bias=float(bias),
        training_metrics=metrics,
        metadata=dict(metadata or {}),
    )


def train_semantic_runtime_scorer_package(
    corpus: SemanticRuntimeLearningCorpus | Sequence[SemanticRuntimeLearningRow],
) -> SemanticRuntimeScorerPackage:
    rows = list(corpus.rows if isinstance(corpus, SemanticRuntimeLearningCorpus) else corpus)
    meta_feature_maps = [_meta_route_feature_map(row) for row in rows]
    orchestration_feature_maps = [_orchestration_route_feature_map(row) for row in rows]
    authority_feature_maps = [_authority_feature_map(row) for row in rows]
    counterfactual_feature_maps: List[Dict[str, float]] = []
    counterfactual_targets: List[float] = []
    for row in rows:
        for counterfactual in row.counterfactuals:
            counterfactual_feature_maps.append(_counterfactual_feature_map(row, counterfactual))
            counterfactual_targets.append(float(counterfactual.predicted_outcome_score))
    meta_route_targets = [_safe_float(row.inferential_summary.get("route_success_label", 0.0)) for row in rows]
    orchestration_targets = [
        1.0
        if (
            _bool(row.inferential_summary.get("route_success_label", False))
            and bool(row.orchestration_transformer_target.get("tool_sequence", []))
        )
        else 0.0
        for row in rows
    ]
    authority_targets = [1.0 if derive_authority_success_label(row) else 0.0 for row in rows]
    regret_targets = [_safe_float(row.inferential_summary.get("estimated_regret", 0.0)) for row in rows]
    package = SemanticRuntimeScorerPackage(
        version="semantic_runtime_scorers_v1",
        meta_route_success_model=_fit_probability_model(
            meta_feature_maps,
            meta_route_targets,
            feature_names=META_ROUTE_FEATURE_NAMES,
            target_name="meta_route_success_probability",
            metadata={"lane": "meta_transformer"},
        ),
        orchestration_route_success_model=_fit_probability_model(
            orchestration_feature_maps,
            orchestration_targets,
            feature_names=ORCHESTRATION_ROUTE_FEATURE_NAMES,
            target_name="orchestration_route_success_probability",
            metadata={"lane": "orchestration_transformer"},
        ),
        authority_calibration_model=_fit_probability_model(
            authority_feature_maps,
            authority_targets,
            feature_names=AUTHORITY_FEATURE_NAMES,
            target_name="authority_success_probability",
            metadata={"lane": "authority"},
        ),
        counterfactual_value_model=_fit_regression_model(
            counterfactual_feature_maps,
            counterfactual_targets,
            feature_names=COUNTERFACTUAL_FEATURE_NAMES,
            target_name="counterfactual_value",
            metadata={"lane": "counterfactual"},
        ),
        regret_model=_fit_regression_model(
            meta_feature_maps,
            regret_targets,
            feature_names=META_ROUTE_FEATURE_NAMES,
            target_name="route_regret",
            metadata={"lane": "meta_transformer"},
        ),
        summary={
            "row_count": len(rows),
            "counterfactual_count": len(counterfactual_targets),
            "meta_route_success_rate": float(sum(meta_route_targets) / float(max(len(meta_route_targets), 1))),
            "authority_success_rate": float(sum(authority_targets) / float(max(len(authority_targets), 1))),
            "mean_regret_target": float(sum(regret_targets) / float(max(len(regret_targets), 1))),
            "meta_route_metrics": {},
            "orchestration_route_metrics": {},
            "authority_metrics": {},
            "counterfactual_metrics": {},
            "regret_metrics": {},
        },
        metadata={
            "corpus_summary": _mapping(getattr(corpus, "summary", {})),
        },
    )
    summary = dict(package.summary)
    summary["meta_route_metrics"] = dict(package.meta_route_success_model.training_metrics)
    summary["orchestration_route_metrics"] = dict(package.orchestration_route_success_model.training_metrics)
    summary["authority_metrics"] = dict(package.authority_calibration_model.training_metrics)
    summary["counterfactual_metrics"] = dict(package.counterfactual_value_model.training_metrics)
    summary["regret_metrics"] = dict(package.regret_model.training_metrics)
    return SemanticRuntimeScorerPackage(
        version=package.version,
        meta_route_success_model=package.meta_route_success_model,
        orchestration_route_success_model=package.orchestration_route_success_model,
        authority_calibration_model=package.authority_calibration_model,
        counterfactual_value_model=package.counterfactual_value_model,
        regret_model=package.regret_model,
        summary=summary,
        metadata=package.metadata,
    )


def score_semantic_runtime_learning_row(
    package: SemanticRuntimeScorerPackage | Mapping[str, Any],
    row: SemanticRuntimeLearningRow,
) -> SemanticRuntimeScoreResult:
    scorer_package = coerce_semantic_runtime_scorer_package(package)
    if scorer_package is None:
        raise ValueError("semantic runtime scorer package is required")
    preferred_authority = str(row.meta_transformer_target.get("authority_gt", "dino"))
    meta_route_probability = scorer_package.meta_route_success_model.predict(_meta_route_feature_map(row))
    orchestration_probability = scorer_package.orchestration_route_success_model.predict(
        _orchestration_route_feature_map(row)
    )
    chosen_authority_confidence = scorer_package.authority_calibration_model.predict(
        _authority_feature_map(row, preferred_authority)
    )
    alternate_authority = "vla" if preferred_authority == "dino" else "dino"
    alternate_authority_confidence = scorer_package.authority_calibration_model.predict(
        _authority_feature_map(row, alternate_authority)
    )
    counterfactual_scores = sorted(
        [
            SemanticRuntimeCounterfactualScore(
                counterfactual_id=item.counterfactual_id,
                rationale=item.rationale,
                rescored_value=scorer_package.counterfactual_value_model.predict(_counterfactual_feature_map(row, item)),
                baseline_score=float(item.predicted_outcome_score),
                executable=item.executable,
                candidate=dict(item.candidate or {}),
            )
            for item in row.counterfactuals
        ],
        key=lambda item: item.rescored_value,
        reverse=True,
    )
    return SemanticRuntimeScoreResult(
        score_id=f"semantic_runtime_score_{sha256_json({'sample_id': row.sample_id, 'wm': row.semantic_world_model_summary.get('world_model_id')})[:16]}",
        semantic_world_model_id=str(row.semantic_world_model_summary.get("world_model_id", "")),
        meta_route_success_probability=float(meta_route_probability),
        orchestration_route_success_probability=float(orchestration_probability),
        predicted_regret=float(max(scorer_package.regret_model.predict(_meta_route_feature_map(row)), 0.0)),
        preferred_authority=preferred_authority,
        calibrated_authority=alternate_authority
        if alternate_authority_confidence > chosen_authority_confidence + 0.05
        else preferred_authority,
        chosen_authority_confidence=float(chosen_authority_confidence),
        alternate_authority_confidence=float(alternate_authority_confidence),
        authority_switch_recommended=bool(alternate_authority_confidence > chosen_authority_confidence + 0.05),
        counterfactual_scores=counterfactual_scores,
        feedback_summary=_mapping(row.feedback_summary),
        metadata={
            "sample_id": row.sample_id,
            "task_id": row.task_id,
            "env_id": row.env_id,
            "source_domain": row.source_domain,
        },
    )


def _runtime_feedback_summary(
    semantic_summary: Mapping[str, Any],
    vla_summary: Mapping[str, Any],
    dino_summary: Mapping[str, Any],
    fusion_summary: Mapping[str, Any],
    outcome_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "annotation_to_world_model": {
            "openvla_available": bool(vla_summary.get("vla_available", False)),
            "teacher_trace_available": bool(vla_summary.get("teacher_trace_available", False)),
            "dino_proxy_available": bool(dino_summary.get("dino_proxy_available", False)),
            "annotation_agreement_score": float(fusion_summary.get("annotation_agreement_score", 0.0)),
            "semantic_grounding_ready": bool(outcome_summary.get("semantic_grounded", False)),
        },
        "world_model_to_transformers": {
            "top_meta_nodes": list(semantic_summary.get("top_meta_nodes", []) or []),
            "active_capabilities": list(semantic_summary.get("active_capabilities", []) or []),
            "object_count": int(semantic_summary.get("object_count", 0) or 0),
            "affordance_density": float(semantic_summary.get("affordance_density", 0.0)),
        },
        "transformers_to_runtime": {
            "can_execute": bool(outcome_summary.get("work_order_ready", False)),
            "readiness_score": float(outcome_summary.get("readiness_score", 0.0)),
            "quality_score": float(outcome_summary.get("quality_score", 0.0)),
        },
        "runtime_to_world_model": {
            "reward_signal": float(outcome_summary.get("reward_signal", 0.0)),
            "fusion_quality": float(outcome_summary.get("semantic_fusion_confidence_mean", 0.0)),
        },
    }


def _coerce_live_semantic_world_model(
    semantic_world_model: Any = None,
    *,
    semantic_snapshot: Any = None,
    orchestrator_context: Any = None,
) -> Optional[SemanticWorldModelState]:
    candidates = [
        semantic_world_model,
        getattr(semantic_snapshot, "semantic_world_model", None),
        getattr(orchestrator_context, "semantic_world_model", None),
        getattr(orchestrator_context, "semantic_snapshot", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, SemanticWorldModelState):
            return candidate
        if isinstance(candidate, SemanticSnapshot):
            if isinstance(candidate.semantic_world_model, SemanticWorldModelState):
                return candidate.semantic_world_model
        if isinstance(candidate, Mapping):
            payload = dict(candidate)
            nested = payload.get("semantic_world_model") if "semantic_world_model" in payload else payload
            if isinstance(nested, Mapping):
                try:
                    return SemanticWorldModelState.from_dict(nested)
                except Exception:
                    continue
    return None


def _live_lane_summaries(
    *,
    semantic_world_model: Any = None,
    semantic_snapshot: Any = None,
    orchestrator_context: Any = None,
    meta_output: Optional[Any] = None,
    orchestration_result: Optional[Any] = None,
    vla_summary: Optional[Mapping[str, Any]] = None,
    dino_summary: Optional[Mapping[str, Any]] = None,
    fusion_summary: Optional[Mapping[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    context_metadata = dict(getattr(orchestrator_context, "semantic_metadata", {}) or {})
    semantic_state = _coerce_live_semantic_world_model(
        semantic_world_model,
        semantic_snapshot=semantic_snapshot,
        orchestrator_context=orchestrator_context,
    )
    semantic_summary = build_semantic_world_model_summary(
        semantic_state,
        semantic_snapshot=semantic_snapshot,
        context=orchestrator_context,
    )
    vla_payload = dict(vla_summary or context_metadata.get("vla_summary", {}) or {})
    dino_payload = dict(dino_summary or context_metadata.get("dino_summary", {}) or {})
    fusion_payload = dict(fusion_summary or context_metadata.get("fusion_summary", {}) or {})
    if not vla_payload:
        vla_payload = {
            "vla_available": _safe_float(getattr(meta_output, "metadata", {}).get("semantic_world_model_summary", {}).get("affordance_grounding", 0.0))
            >= 0.45,
            "vla_confidence_mean": _safe_float(getattr(meta_output, "metadata", {}).get("semantic_world_model_summary", {}).get("affordance_grounding", 0.0)),
            "teacher_trace_available": bool(context_metadata.get("teacher_trace_available", False)),
            "teacher_confidence_mean": _safe_float(context_metadata.get("teacher_confidence_mean", 0.0)),
            "teacher_object_refs": list(context_metadata.get("teacher_object_refs", []) or []),
            "teacher_affordance_hints": list(context_metadata.get("teacher_affordance_hints", []) or []),
            "teacher_risk_hints": list(context_metadata.get("teacher_risk_hints", []) or []),
        }
    if not dino_payload:
        dino_payload = {
            "dino_proxy_available": _safe_float(semantic_summary.get("object_memory", 0.0)) > 0.0,
            "dino_proxy_confidence_mean": _safe_float(semantic_summary.get("object_memory", 0.0)),
            "scene_tracks_available": _safe_float(semantic_summary.get("grounded_track_object_count", 0.0)) > 0.0,
            "scene_track_count": int(semantic_summary.get("grounded_track_object_count", 0) or 0),
            "scene_track_label_confidence_mean": _safe_float(semantic_summary.get("object_memory", 0.0)),
            "map_first_available": False,
            "map_first_confidence_mean": 0.0,
            "scene_tracks_backend": str(context_metadata.get("scene_tracks_backend", "")),
        }
    if not fusion_payload:
        fusion_confidence = max(
            min(
                _safe_float(vla_payload.get("vla_confidence_mean", 0.0)),
                _safe_float(dino_payload.get("dino_proxy_confidence_mean", 0.0)),
            ),
            _safe_float(semantic_summary.get("fusion_bridge", 0.0)),
        )
        confidence_gap = abs(
            _safe_float(vla_payload.get("vla_confidence_mean", 0.0))
            - _safe_float(dino_payload.get("dino_proxy_confidence_mean", 0.0))
        )
        fusion_payload = {
            "fusion_available": bool(vla_payload.get("vla_available", False) and dino_payload.get("dino_proxy_available", False)),
            "semantic_fusion_confidence_mean": float(fusion_confidence),
            "annotation_agreement_score": float(max(0.0, min(1.0, 1.0 - confidence_gap))),
            "source_confidence_gap": float(confidence_gap),
            "fusion_advantage_score": float(max(fusion_confidence - max(
                _safe_float(vla_payload.get("vla_confidence_mean", 0.0)),
                _safe_float(dino_payload.get("dino_proxy_confidence_mean", 0.0)),
            ), 0.0)),
        }
    meta_target = {
        "authority_gt": str(getattr(meta_output, "authority", context_metadata.get("authority_gt", "dino"))),
        "confidence_vla": _safe_float(
            context_metadata.get("confidence_vla", _safe_float(vla_payload.get("vla_confidence_mean", 0.0)))
        ),
        "confidence_dino": _safe_float(
            context_metadata.get("confidence_dino", _safe_float(dino_payload.get("dino_proxy_confidence_mean", 0.0)))
        ),
        "objective_preset": str(getattr(meta_output, "objective_preset", context_metadata.get("objective_preset", "balanced"))),
        "energy_profile_weights": dict(getattr(meta_output, "energy_profile_weights", {}) or context_metadata.get("energy_profile_weights", {})),
        "data_mix_weights": dict(getattr(meta_output, "data_mix_weights", {}) or context_metadata.get("data_mix_weights", {})),
        "chosen_backend": str(
            getattr(meta_output, "chosen_backend", context_metadata.get("chosen_backend", getattr(orchestrator_context, "engine_type", "pybullet")))
        ),
        "expected_deltas": {
            "expected_delta_mpl": _safe_float(getattr(meta_output, "expected_delta_mpl", context_metadata.get("expected_delta_mpl", 0.0))),
            "expected_delta_error": _safe_float(getattr(meta_output, "expected_delta_error", context_metadata.get("expected_delta_error", 0.0))),
            "expected_delta_energy_Wh": _safe_float(
                getattr(meta_output, "expected_delta_energy_Wh", context_metadata.get("expected_delta_energy_Wh", 0.0))
            ),
        },
        "bounded_actions": list(getattr(meta_output, "bounded_actions", []) or context_metadata.get("bounded_actions", [])),
        "plan": list(getattr(meta_output, "orchestration_plan", []) or context_metadata.get("plan", [])),
    }
    execution_preconditions = dict(getattr(meta_output, "execution_preconditions", {}) or {})
    meta_work_order = dict(getattr(meta_output, "execution_work_order", {}) or {})
    orchestration_metadata = dict(getattr(orchestration_result, "metadata", {}) or {})
    orchestration_target = {
        "tool_sequence": [
            {
                "name": str(step.tool_call.name),
                "args": dict(step.tool_call.args or {}),
            }
            for step in list(getattr(orchestration_result, "steps", []) or [])
            if getattr(step, "tool_call", None) is not None
        ],
        "chosen_backend": str(getattr(orchestration_result, "chosen_backend", meta_target.get("chosen_backend", "pybullet"))),
        "objective_preset": str(getattr(orchestration_result, "objective_preset", meta_target.get("objective_preset", "balanced"))),
        "energy_profile_weights": dict(getattr(orchestration_result, "energy_profile_weights", {}) or {}),
        "data_mix_weights": dict(getattr(orchestration_result, "data_mix_weights", {}) or {}),
        "execution_mode": str(getattr(orchestration_result, "execution_mode", "advisory")),
        "activation_plan": dict(getattr(orchestration_result, "activation_plan", {}) or {}),
    }
    orchestration_preconditions = dict(orchestration_metadata.get("execution_preconditions", {}) or {})
    orchestration_work_order = dict(getattr(orchestration_result, "activation_work_order", {}) or {})
    readiness_score = max(
        _safe_float(execution_preconditions.get("readiness_score", 0.0)),
        _safe_float(orchestration_preconditions.get("readiness_score", 0.0)),
    )
    work_order_ready = bool(
        meta_work_order.get("ready", execution_preconditions.get("ready", False))
        or orchestration_work_order.get("ready", orchestration_preconditions.get("ready", False))
    )
    semantic_grounded = bool(
        context_metadata.get("semantic_grounded", False)
        or _safe_float(semantic_summary.get("grounded_track_object_count", 0.0)) > 0.0
    )
    reward_signal = 1.0 / (1.0 + np.exp(-_safe_float(getattr(orchestrator_context, "mean_w_econ", 0.0))))
    quality_score = max(
        0.25 * (1.0 if work_order_ready else 0.0)
        + 0.25 * readiness_score
        + 0.2 * (1.0 if semantic_grounded else 0.0)
        + 0.15 * _safe_float(fusion_payload.get("semantic_fusion_confidence_mean", 0.0))
        + 0.15 * reward_signal,
        0.0,
    )
    outcome_summary = {
        "execution_ready": bool(
            execution_preconditions.get("ready", False) or orchestration_preconditions.get("ready", False)
        ),
        "work_order_ready": work_order_ready,
        "readiness_score": float(readiness_score),
        "semantic_grounded": semantic_grounded,
        "teacher_runtime_live": bool(
            context_metadata.get("teacher_runtime_live", False) or vla_payload.get("teacher_trace_available", False)
        ),
        "scene_tracks_non_stub": bool(
            context_metadata.get("scene_tracks_non_stub", False)
            or dino_payload.get("scene_tracks_backend") in {"real", "passthrough", "auto"}
        ),
        "semantic_fusion_confidence_mean": float(fusion_payload.get("semantic_fusion_confidence_mean", 0.0)),
        "quality_score": float(quality_score),
        "reward_signal": float(reward_signal),
    }
    feedback_summary = _runtime_feedback_summary(
        semantic_summary,
        vla_payload,
        dino_payload,
        fusion_payload,
        outcome_summary,
    )
    return semantic_summary, vla_payload, dino_payload, fusion_payload, meta_target, orchestration_target | {
        "_outcome_summary": outcome_summary,
        "_feedback_summary": feedback_summary,
    }


def _live_counterfactuals(
    meta_target: Mapping[str, Any],
    outcome_summary: Mapping[str, Any],
) -> List[SemanticRuntimeCounterfactual]:
    chosen_preset = str(meta_target.get("objective_preset", "balanced"))
    chosen_authority = str(meta_target.get("authority_gt", "dino"))
    chosen_score = max(
        _safe_float(meta_target.get("confidence_vla", 0.0)),
        _safe_float(meta_target.get("confidence_dino", 0.0)),
    )
    counterfactuals: List[SemanticRuntimeCounterfactual] = []
    for preset in ["balanced", "safety", "energy_saver", "throughput"]:
        if preset == chosen_preset:
            continue
        baseline_score = chosen_score
        if preset == "safety":
            baseline_score = max(chosen_score, 0.55)
        elif preset == "energy_saver":
            baseline_score = max(chosen_score * 0.9, 0.45)
        elif preset == "throughput":
            baseline_score = max(chosen_score * 0.92, 0.48)
        counterfactuals.append(
            SemanticRuntimeCounterfactual(
                counterfactual_id=f"runtime_cf_{sha256_json({'preset': preset, 'authority': chosen_authority})[:12]}",
                lane="meta_transformer",
                candidate={"objective_preset": preset},
                predicted_outcome_score=float(baseline_score),
                predicted_regret=float(max(chosen_score - baseline_score, 0.0)),
                executable=bool(outcome_summary.get("execution_ready", False)),
                rationale=f"runtime_counterfactual_objective_preset:{preset}",
            )
        )
    alternate_authority = "vla" if chosen_authority == "dino" else "dino"
    alternate_score = _safe_float(meta_target.get(f"confidence_{alternate_authority}", 0.0))
    counterfactuals.append(
        SemanticRuntimeCounterfactual(
            counterfactual_id=f"runtime_cf_{sha256_json({'authority': alternate_authority})[:12]}",
            lane="meta_transformer",
            candidate={"authority_gt": alternate_authority},
            predicted_outcome_score=float(alternate_score),
            predicted_regret=float(max(chosen_score - alternate_score, 0.0)),
            executable=True,
            rationale=f"runtime_counterfactual_authority:{alternate_authority}",
        )
    )
    return counterfactuals


def score_live_semantic_runtime_stack(
    package: SemanticRuntimeScorerPackage | Mapping[str, Any],
    *,
    semantic_world_model: Any = None,
    semantic_snapshot: Any = None,
    orchestrator_context: Any = None,
    meta_output: Optional[Any] = None,
    orchestration_result: Optional[Any] = None,
    vla_summary: Optional[Mapping[str, Any]] = None,
    dino_summary: Optional[Mapping[str, Any]] = None,
    fusion_summary: Optional[Mapping[str, Any]] = None,
) -> SemanticRuntimeScoreResult:
    scorer_package = coerce_semantic_runtime_scorer_package(package)
    if scorer_package is None:
        raise ValueError("semantic runtime scorer package is required")
    (
        semantic_summary,
        vla_payload,
        dino_payload,
        fusion_payload,
        meta_target,
        orchestration_target_payload,
    ) = _live_lane_summaries(
        semantic_world_model=semantic_world_model,
        semantic_snapshot=semantic_snapshot,
        orchestrator_context=orchestrator_context,
        meta_output=meta_output,
        orchestration_result=orchestration_result,
        vla_summary=vla_summary,
        dino_summary=dino_summary,
        fusion_summary=fusion_summary,
    )
    outcome_summary = dict(orchestration_target_payload.pop("_outcome_summary", {}) or {})
    feedback_summary = dict(orchestration_target_payload.pop("_feedback_summary", {}) or {})
    base_payload = SemanticRuntimeLearningRow(
        sample_id="runtime_live",
        run_id="runtime_live",
        episode_id="runtime_live",
        task_id=str(semantic_summary.get("task_id", getattr(orchestrator_context, "task_type", "runtime_live"))),
        env_id=str(getattr(orchestrator_context, "env_name", "runtime")),
        source_domain="runtime_live",
        semantic_world_model_summary=semantic_summary,
        semantic_tokens=[],
        vla_summary=vla_payload,
        dino_summary=dino_payload,
        fusion_summary=fusion_payload,
        feedback_summary=feedback_summary,
        meta_transformer_target=meta_target,
        orchestration_transformer_target=orchestration_target_payload,
        outcome_summary=outcome_summary,
        inferential_summary={},
        counterfactuals=_live_counterfactuals(meta_target, outcome_summary),
        artifact_refs={},
        metadata={},
    )
    return score_semantic_runtime_learning_row(scorer_package, base_payload)


def write_semantic_runtime_scorer_package(
    output_path: str | Path,
    package: SemanticRuntimeScorerPackage,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(package.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def load_semantic_runtime_scorer_package(path: str | Path) -> SemanticRuntimeScorerPackage:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid semantic runtime scorer package: {path}")
    return SemanticRuntimeScorerPackage.from_dict(payload)


__all__ = [
    "SemanticRuntimeCounterfactualScore",
    "SemanticRuntimeLinearModel",
    "SemanticRuntimeScoreResult",
    "SemanticRuntimeScorerPackage",
    "coerce_semantic_runtime_scorer_package",
    "derive_authority_success_label",
    "load_semantic_runtime_scorer_package",
    "score_live_semantic_runtime_stack",
    "score_semantic_runtime_learning_row",
    "train_semantic_runtime_scorer_package",
    "write_semantic_runtime_scorer_package",
]

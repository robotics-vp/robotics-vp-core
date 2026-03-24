"""Replay-backed semantic runtime learning corpus and shadow counterfactuals."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.config.objective_profile import ObjectiveVector
from src.evidence.teacher_trace import TeacherTrace
from src.orchestrator.context import OrchestratorContext
from src.orchestrator.meta_transformer_training import MetaTransformerSample
from src.orchestrator.orchestration_transformer import _encode_ctx
from src.orchestrator.semantic_transformer_bridge import (
    build_semantic_orchestration_plan,
    build_semantic_world_model_summary,
    build_tool_biases,
    derive_backend,
    derive_data_mix_weights,
    derive_energy_profile_mix,
    derive_objective_preset,
    encode_semantic_world_model_features,
    estimate_expected_deltas,
    semantic_tokens,
)
from src.orchestrator.toolspecs import ToolCall
from src.orchestrator.training_dataset import OrchestrationSample
from src.replay.dataset import ReplayDatasetBundle
from src.replay.schema import ReplayEpisodeRecord, ReplayStepRecord, ReplayWindowRecord
from src.utils.config_digest import sha256_json
from src.utils.json_safe import to_json_safe
from src.vision.map_first_supervision.semantics import parse_vla_semantic_evidence


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _rows(values: Optional[Sequence[Any]]) -> List[Any]:
    return list(values or [])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_mean(value: Any, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return float(default)
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "ready", "success"}
    return bool(value)


def _string_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        return [str(value) for value in values if str(value)]
    return [str(values)] if str(values) else []


def _normalize_weights(payload: Mapping[str, Any]) -> Dict[str, float]:
    weights = {
        str(key): max(0.0, _safe_float(value))
        for key, value in dict(payload or {}).items()
    }
    total = sum(weights.values())
    if total <= 0.0:
        return {key: 0.0 for key in weights}
    return {
        key: value / total
        for key, value in weights.items()
    }


def _extract_artifact_refs(*payloads: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    refs: Dict[str, Any] = {}
    for payload in payloads:
        for key, value in dict(payload or {}).items():
            if value in (None, "", [], {}):
                continue
            normalized = str(key)
            if normalized.endswith(("_path", "_paths", "_ref", "_refs", "_id", "_ids")):
                refs[normalized] = value
                if normalized.endswith("_path"):
                    refs[f"{normalized[:-5]}_ref"] = value
    return dict(sorted(refs.items()))


def _resolve_artifact_path(root_dir: Optional[str], ref: Any) -> Optional[Path]:
    if ref in (None, "", [], {}):
        return None
    path = Path(str(ref))
    if path.exists():
        return path
    if root_dir:
        candidate = Path(root_dir) / path
        if candidate.exists():
            return candidate
    return None


def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _load_npz(path: Optional[Path]) -> Dict[str, np.ndarray]:
    if path is None or not path.exists():
        return {}
    try:
        return dict(np.load(path, allow_pickle=True))
    except Exception:
        return {}


def _objective_vector_from_preset(preset: str) -> List[float]:
    try:
        return ObjectiveVector.from_preset(str(preset or "balanced")).to_list()
    except Exception:
        return [0.6, 0.2, 0.15, 0.05, 0.0]


@dataclass(frozen=True)
class SemanticRuntimeCounterfactual:
    counterfactual_id: str
    lane: str
    candidate: Dict[str, Any]
    predicted_outcome_score: float
    predicted_regret: float
    executable: bool
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counterfactual_id": self.counterfactual_id,
            "lane": self.lane,
            "candidate": _mapping(self.candidate),
            "predicted_outcome_score": float(self.predicted_outcome_score),
            "predicted_regret": float(self.predicted_regret),
            "executable": bool(self.executable),
            "rationale": self.rationale,
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class SemanticRuntimeLearningRow:
    sample_id: str
    run_id: str
    episode_id: str
    task_id: str
    env_id: str
    source_domain: str
    semantic_world_model_summary: Dict[str, Any]
    semantic_tokens: List[str]
    vla_summary: Dict[str, Any]
    dino_summary: Dict[str, Any]
    fusion_summary: Dict[str, Any]
    feedback_summary: Dict[str, Any]
    meta_transformer_target: Dict[str, Any]
    orchestration_transformer_target: Dict[str, Any]
    outcome_summary: Dict[str, Any]
    inferential_summary: Dict[str, Any]
    counterfactuals: List[SemanticRuntimeCounterfactual] = field(default_factory=list)
    artifact_refs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "env_id": self.env_id,
            "source_domain": self.source_domain,
            "semantic_world_model_summary": _mapping(self.semantic_world_model_summary),
            "semantic_tokens": list(self.semantic_tokens),
            "vla_summary": _mapping(self.vla_summary),
            "dino_summary": _mapping(self.dino_summary),
            "fusion_summary": _mapping(self.fusion_summary),
            "feedback_summary": _mapping(self.feedback_summary),
            "meta_transformer_target": _mapping(self.meta_transformer_target),
            "orchestration_transformer_target": _mapping(self.orchestration_transformer_target),
            "outcome_summary": _mapping(self.outcome_summary),
            "inferential_summary": _mapping(self.inferential_summary),
            "counterfactuals": [item.to_dict() for item in self.counterfactuals],
            "artifact_refs": _mapping(self.artifact_refs),
            "metadata": _mapping(self.metadata),
        }


@dataclass(frozen=True)
class SemanticRuntimeLearningCorpus:
    rows: List[SemanticRuntimeLearningRow]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": [row.to_dict() for row in self.rows],
            "summary": _mapping(self.summary),
        }


def _summarize_vla_lane(artifact_refs: Mapping[str, Any], root_dir: Optional[str]) -> Dict[str, Any]:
    teacher_trace_path = _resolve_artifact_path(
        root_dir,
        artifact_refs.get("teacher_trace_ref") or artifact_refs.get("teacher_trace_path"),
    )
    teacher_trace_payload = _load_json(teacher_trace_path)
    teacher_trace = None
    if teacher_trace_payload:
        try:
            teacher_trace = TeacherTrace.from_dict(teacher_trace_payload)
        except Exception:
            teacher_trace = None
    vla_path = _resolve_artifact_path(
        root_dir,
        artifact_refs.get("vla_semantic_evidence_ref")
        or artifact_refs.get("vla_semantic_evidence_path")
        or artifact_refs.get("semantic_evidence_ref")
        or artifact_refs.get("semantic_evidence_path"),
    )
    vla_evidence = parse_vla_semantic_evidence(_load_npz(vla_path)) if vla_path is not None else None
    provenance = dict(getattr(vla_evidence, "provenance", {}) or {})
    confidence_mean = _safe_mean(getattr(vla_evidence, "confidence", None), 0.0)
    track_ids = getattr(vla_evidence, "track_ids", None) if vla_evidence is not None else None
    teacher_confidence_mean = _safe_float(
        (teacher_trace.summary or {}).get("teacher_confidence_mean", 0.0),
        _safe_mean([step.confidence for step in teacher_trace.steps], 0.0) if teacher_trace is not None else 0.0,
    )
    return {
        "teacher_trace_available": teacher_trace is not None,
        "teacher_confidence_mean": float(teacher_confidence_mean),
        "teacher_semantic_tags": list((teacher_trace.metadata or {}).get("semantic_tags", []) or []) if teacher_trace is not None else [],
        "teacher_object_refs": list((teacher_trace.metadata or {}).get("object_refs", []) or []) if teacher_trace is not None else [],
        "teacher_affordance_hints": list((teacher_trace.metadata or {}).get("affordance_hints", []) or []) if teacher_trace is not None else [],
        "teacher_risk_hints": list((teacher_trace.metadata or {}).get("risk_hints", []) or []) if teacher_trace is not None else [],
        "vla_available": bool(provenance.get("vla_available", False) or confidence_mean > 0.1),
        "vla_confidence_mean": float(confidence_mean),
        "vla_source": str(provenance.get("source", "")),
        "vla_fallback_mode": str(provenance.get("fallback_mode", "")),
        "vla_object_refs": _string_list(provenance.get("object_refs")),
        "vla_affordance_hints": _string_list(provenance.get("affordance_hints")),
        "vla_risk_hints": _string_list(provenance.get("risk_hints")),
        "vla_semantic_tags": _string_list(provenance.get("semantic_tags")),
        "vla_track_count": int(len(track_ids)) if track_ids is not None else 0,
        "instruction": str(provenance.get("instruction", teacher_trace.instruction if teacher_trace is not None else "")),
    }


def _summarize_dino_lane(
    artifact_refs: Mapping[str, Any],
    root_dir: Optional[str],
    semantic_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    scene_tracks_path = _resolve_artifact_path(
        root_dir,
        artifact_refs.get("scene_tracks_ref") or artifact_refs.get("scene_tracks_path"),
    )
    scene_tracks = _load_npz(scene_tracks_path)
    map_first_path = _resolve_artifact_path(
        root_dir,
        artifact_refs.get("map_first_ref")
        or artifact_refs.get("map_first_supervision_ref")
        or artifact_refs.get("map_first_supervision_path"),
    )
    map_first = _load_npz(map_first_path)
    track_ids = scene_tracks.get("scene_tracks_v1/track_ids")
    label_confidence = scene_tracks.get("scene_tracks_v1/track_label_confidence")
    motion_score = scene_tracks.get("scene_tracks_v1/track_motion_score")
    map_confidence = map_first.get("map_first_v1/confidence")
    map_vla_confidence = map_first.get("map_first_v1/vla_confidence")
    track_backend = ""
    summary_json_raw = scene_tracks.get("scene_tracks_v1/summary_json")
    if isinstance(summary_json_raw, np.ndarray) and summary_json_raw.size > 0:
        try:
            summary_json = json.loads(str(summary_json_raw.flat[0]))
            track_backend = str(summary_json.get("backend_selected", summary_json.get("adapter_status", {}).get("overall_mode", "")))
        except Exception:
            track_backend = ""
    if not track_backend:
        track_backend = str(semantic_summary.get("grounding_mode", ""))
    return {
        "scene_tracks_available": bool(scene_tracks),
        "scene_track_count": int(len(track_ids) if track_ids is not None else 0),
        "scene_track_label_confidence_mean": float(_safe_mean(label_confidence, semantic_summary.get("object_memory", 0.0))),
        "scene_track_motion_mean": float(_safe_mean(motion_score, 0.0)),
        "map_first_available": bool(map_first),
        "map_first_confidence_mean": float(_safe_mean(map_confidence, semantic_summary.get("fusion_bridge", 0.0))),
        "map_first_vla_confidence_mean": float(_safe_mean(map_vla_confidence, 0.0)),
        "dino_proxy_available": bool(scene_tracks or map_first),
        "dino_proxy_confidence_mean": float(
            max(
                _safe_mean(label_confidence, 0.0),
                _safe_mean(map_confidence, 0.0),
                _safe_float(semantic_summary.get("object_memory", 0.0)),
            )
        ),
        "scene_tracks_backend": track_backend,
        "grounded_track_object_count": int(semantic_summary.get("grounded_track_object_count", 0) or 0),
    }


def _summarize_fusion_lane(
    episode: ReplayEpisodeRecord,
    vla_summary: Mapping[str, Any],
    dino_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = dict(episode.metadata or {})
    semantic_fusion_confidence = _safe_float(
        metadata.get("semantic_fusion_confidence_mean", metadata.get("semantic_fusion_quality_score", 0.0))
    )
    fused_available = semantic_fusion_confidence > 0.0
    if not fused_available:
        fused_available = bool(vla_summary.get("vla_available") and dino_summary.get("dino_proxy_available"))
    source_gap = abs(
        _safe_float(vla_summary.get("vla_confidence_mean", 0.0))
        - _safe_float(dino_summary.get("dino_proxy_confidence_mean", 0.0))
    )
    agreement_score = max(0.0, min(1.0, 1.0 - source_gap))
    return {
        "fusion_available": fused_available,
        "semantic_fusion_confidence_mean": float(semantic_fusion_confidence),
        "annotation_agreement_score": float(agreement_score),
        "source_confidence_gap": float(source_gap),
        "fusion_advantage_score": float(
            max(
                semantic_fusion_confidence - max(
                    _safe_float(vla_summary.get("vla_confidence_mean", 0.0)),
                    _safe_float(dino_summary.get("dino_proxy_confidence_mean", 0.0)),
                ),
                0.0,
            )
        ),
    }


def _summarize_outcome(
    episode: ReplayEpisodeRecord,
    steps: Sequence[ReplayStepRecord],
    windows: Sequence[ReplayWindowRecord],
    semantic_summary: Mapping[str, Any],
    fusion_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = dict(episode.metadata or {})
    execution_preconditions = dict(metadata.get("execution_preconditions", {}) or {})
    source_work_order = dict(
        metadata.get("source_execution_work_order")
        or metadata.get("execution_work_order")
        or {}
    )
    future_signals = dict(metadata.get("future_training_signals", {}) or {})
    total_reward = float(episode.total_reward)
    reward_signal = 1.0 / (1.0 + np.exp(-total_reward / 10.0))
    success = episode.status.lower() in {"success", "completed", "done"}
    readiness_score = _safe_float(execution_preconditions.get("readiness_score", 0.0))
    semantic_grounded = bool(
        future_signals.get("semantic_memory_grounded", False)
        or _safe_float(semantic_summary.get("grounded_track_object_count", 0.0)) > 0.0
    )
    work_order_ready = bool(source_work_order.get("ready", execution_preconditions.get("ready", False)))
    fusion_quality = _safe_float(fusion_summary.get("semantic_fusion_confidence_mean", 0.0))
    quality_score = (
        0.25 * (1.0 if success else 0.0)
        + 0.25 * readiness_score
        + 0.2 * (1.0 if semantic_grounded else 0.0)
        + 0.15 * fusion_quality
        + 0.15 * reward_signal
    )
    return {
        "success": bool(success),
        "execution_ready": bool(execution_preconditions.get("ready", False)),
        "work_order_ready": work_order_ready,
        "readiness_score": float(readiness_score),
        "semantic_grounded": semantic_grounded,
        "teacher_runtime_live": bool(future_signals.get("teacher_runtime_live", False)),
        "scene_tracks_non_stub": bool(future_signals.get("scene_tracks_non_stub", False)),
        "promotion_trace_complete": bool(future_signals.get("promotion_trace_complete", False)),
        "total_reward": float(total_reward),
        "reward_signal": float(reward_signal),
        "step_count": int(len(steps)),
        "window_count": int(len(windows)),
        "semantic_fusion_confidence_mean": float(fusion_quality),
        "quality_score": float(quality_score),
    }


def _feedback_summary(
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
            "promotion_trace_complete": bool(outcome_summary.get("promotion_trace_complete", False)),
        },
    }


def _authority_success_label(
    semantic_summary: Mapping[str, Any],
    vla_summary: Mapping[str, Any],
    dino_summary: Mapping[str, Any],
    fusion_summary: Mapping[str, Any],
    meta_target: Mapping[str, Any],
    outcome_summary: Mapping[str, Any],
) -> bool:
    authority = str(meta_target.get("authority_gt", "dino"))
    route_success = bool(outcome_summary.get("success", False) and outcome_summary.get("work_order_ready", False))
    if authority == "vla":
        return bool(
            route_success
            and vla_summary.get("vla_available", False)
            and (
                _safe_float(fusion_summary.get("annotation_agreement_score", 0.0)) >= 0.35
                or _safe_float(semantic_summary.get("affordance_grounding", 0.0)) >= 0.45
            )
        )
    return bool(
        route_success
        and dino_summary.get("dino_proxy_available", False)
        and outcome_summary.get("semantic_grounded", False)
    )


def _build_meta_transformer_target(
    semantic_summary: Mapping[str, Any],
    vla_summary: Mapping[str, Any],
    dino_summary: Mapping[str, Any],
    fusion_summary: Mapping[str, Any],
    outcome_summary: Mapping[str, Any],
    instruction: str,
) -> Dict[str, Any]:
    econ_signals = {
        "mpl_urgency": max(0.0, 1.0 - float(outcome_summary.get("reward_signal", 0.0))),
        "error_urgency": max(0.0, 1.0 - float(outcome_summary.get("quality_score", 0.0))),
        "energy_urgency": float(semantic_summary.get("efficiency_router_score", 0.0)),
    }
    datapack_signals = {
        "data_coverage_score": float(outcome_summary.get("quality_score", 0.0)),
        "embedding_diversity": float(dino_summary.get("scene_track_motion_mean", 0.0)),
        "vla_annotation_fraction": 1.0 if vla_summary.get("vla_available", False) else 0.0,
        "guidance_annotation_fraction": 1.0 if vla_summary.get("teacher_trace_available", False) else 0.0,
    }
    objective_preset = derive_objective_preset(
        semantic_summary,
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
        instruction=instruction,
    )
    energy_profile_weights = derive_energy_profile_mix(
        semantic_summary,
        econ_signals=econ_signals,
        objective_preset=objective_preset,
    )
    data_mix_weights = derive_data_mix_weights(
        semantic_summary,
        datapack_signals=datapack_signals,
    )
    chosen_backend = derive_backend(
        semantic_summary,
        econ_signals=econ_signals,
        current_backend="pybullet",
    )
    expected_deltas = estimate_expected_deltas(
        semantic_summary,
        econ_signals=econ_signals,
        datapack_signals=datapack_signals,
    )
    dino_conf = max(
        _safe_float(dino_summary.get("dino_proxy_confidence_mean", 0.0)),
        _safe_float(semantic_summary.get("object_memory", 0.0)),
    )
    vla_conf = max(
        _safe_float(vla_summary.get("vla_confidence_mean", 0.0)),
        _safe_float(fusion_summary.get("semantic_fusion_confidence_mean", 0.0)),
    )
    authority_gt = "dino"
    if vla_conf > dino_conf and _safe_float(semantic_summary.get("affordance_grounding", 0.0)) >= 0.45:
        authority_gt = "vla"
    plan = build_semantic_orchestration_plan(
        semantic_summary,
        objective_preset=objective_preset,
        data_mix_weights=data_mix_weights,
        energy_profile_weights=energy_profile_weights,
        datapack_signals=datapack_signals,
    )
    return {
        "authority_gt": authority_gt,
        "authority_score": float(max(vla_conf, dino_conf)),
        "confidence_vla": float(vla_conf),
        "confidence_dino": float(dino_conf),
        "objective_preset": objective_preset,
        "energy_profile_weights": dict(energy_profile_weights),
        "data_mix_weights": dict(data_mix_weights),
        "chosen_backend": chosen_backend,
        "expected_deltas": dict(expected_deltas),
        "execution_mode": "bounded_execution" if outcome_summary.get("execution_ready", False) else "advisory",
        "bounded_actions": [
            "set_objective_preset",
            "set_energy_profile",
            "set_data_mix",
            "set_backend",
        ],
        "plan": plan,
    }


def _tool_sequence_from_plan(
    semantic_summary: Mapping[str, Any],
    meta_target: Mapping[str, Any],
    instruction: str,
) -> List[Dict[str, Any]]:
    tool_biases = build_tool_biases(
        semantic_summary,
        econ_signals={
            "mpl_urgency": _safe_float(meta_target.get("expected_deltas", {}).get("expected_delta_mpl", 0.0)) / 10.0,
            "energy_urgency": _safe_float(meta_target.get("expected_deltas", {}).get("expected_delta_energy_Wh", 0.0)) / 10.0,
            "error_urgency": min(abs(_safe_float(meta_target.get("expected_deltas", {}).get("expected_delta_error", 0.0))), 1.0),
        },
        datapack_signals={"data_coverage_score": _safe_float(meta_target.get("confidence_dino", 0.0))},
        instruction=instruction,
    )
    ranked = sorted(tool_biases.items(), key=lambda item: item[1], reverse=True)
    selected_names = [name for name, _ in ranked[:4]]
    sequence: List[Dict[str, Any]] = []
    for name in selected_names:
        args: Dict[str, Any] = {}
        if name == "SET_OBJECTIVE_PRESET":
            args["preset"] = str(meta_target.get("objective_preset", "balanced"))
        elif name == "SET_ENERGY_PROFILE":
            args["profile_mix"] = dict(meta_target.get("energy_profile_weights", {}))
        elif name == "SET_DATA_MIX":
            args["data_mix"] = dict(meta_target.get("data_mix_weights", {}))
        elif name == "SET_BACKEND":
            args["backend"] = str(meta_target.get("chosen_backend", "pybullet"))
        elif name == "QUERY_DATAPACKS":
            args["filter"] = {"focus": list(semantic_summary.get("top_meta_nodes", []) or [])}
        elif name == "QUERY_ENERGY_SURFACE":
            args["profile_query"] = True
        elif name == "CALL_VLA_FOR_DATAPACK_CLASS":
            args["class_filter"] = list(semantic_summary.get("top_object_labels", []) or [])
        elif name == "CALL_VLA_SINGLE_STEP":
            args["focus_meta_node"] = next(iter(semantic_summary.get("top_meta_nodes", []) or []), "")
        sequence.append({"name": name, "args": args, "score": float(tool_biases.get(name, 0.0))})
    return sequence


def _build_orchestration_target(
    semantic_summary: Mapping[str, Any],
    meta_target: Mapping[str, Any],
    outcome_summary: Mapping[str, Any],
    instruction: str,
) -> Dict[str, Any]:
    tool_sequence = _tool_sequence_from_plan(semantic_summary, meta_target, instruction)
    activation_plan = {
        "mode": "bounded_execution" if outcome_summary.get("execution_ready", False) else "advisory",
        "bounded_actions": [
            "set_objective_preset",
            "set_energy_profile",
            "set_data_mix",
            "set_backend",
            "query_context",
        ],
        "semantic_plan": list(meta_target.get("plan", []) or []),
        "tool_sequence": [row["name"] for row in tool_sequence],
    }
    return {
        "tool_sequence": tool_sequence,
        "chosen_backend": str(meta_target.get("chosen_backend", "pybullet")),
        "objective_preset": str(meta_target.get("objective_preset", "balanced")),
        "energy_profile_weights": dict(meta_target.get("energy_profile_weights", {})),
        "data_mix_weights": dict(meta_target.get("data_mix_weights", {})),
        "execution_mode": activation_plan["mode"],
        "activation_plan": activation_plan,
    }


def _preset_alignment_score(
    preset: str,
    semantic_summary: Mapping[str, Any],
    meta_target: Mapping[str, Any],
    outcome_summary: Mapping[str, Any],
) -> float:
    if preset == "safety":
        return (
            0.45 * _safe_float(semantic_summary.get("risk_triage_score", 0.0))
            + 0.25 * _safe_float(semantic_summary.get("risk_object_fraction", 0.0))
            + 0.15 * (1.0 if outcome_summary.get("execution_ready", False) else 0.0)
            + 0.15 * _safe_float(meta_target.get("confidence_dino", 0.0))
        )
    if preset == "energy_saver":
        return (
            0.45 * _safe_float(semantic_summary.get("efficiency_router_score", 0.0))
            + 0.25 * _safe_float(outcome_summary.get("quality_score", 0.0))
            + 0.15 * _safe_float(meta_target.get("expected_deltas", {}).get("expected_delta_energy_Wh", 0.0)) / 5.0
            + 0.15 * _safe_float(meta_target.get("confidence_vla", 0.0))
        )
    if preset == "throughput":
        return (
            0.35 * max(_safe_float(meta_target.get("expected_deltas", {}).get("expected_delta_mpl", 0.0)), 0.0) / 5.0
            + 0.25 * _safe_float(outcome_summary.get("reward_signal", 0.0))
            + 0.2 * (1.0 - _safe_float(semantic_summary.get("risk_object_fraction", 0.0)))
            + 0.2 * _safe_float(meta_target.get("confidence_vla", 0.0))
        )
    return (
        0.35 * _safe_float(outcome_summary.get("quality_score", 0.0))
        + 0.25 * _safe_float(semantic_summary.get("capability_mean", 0.0))
        + 0.2 * _safe_float(meta_target.get("confidence_dino", 0.0))
        + 0.2 * _safe_float(meta_target.get("confidence_vla", 0.0))
    )


def _build_counterfactuals(
    sample_id: str,
    semantic_summary: Mapping[str, Any],
    meta_target: Mapping[str, Any],
    outcome_summary: Mapping[str, Any],
    *,
    max_count: int,
) -> List[SemanticRuntimeCounterfactual]:
    chosen_preset = str(meta_target.get("objective_preset", "balanced"))
    chosen_score = _preset_alignment_score(chosen_preset, semantic_summary, meta_target, outcome_summary)
    candidates: List[SemanticRuntimeCounterfactual] = []
    for preset in ["balanced", "safety", "energy_saver", "throughput"]:
        if preset == chosen_preset:
            continue
        candidate_score = _preset_alignment_score(preset, semantic_summary, meta_target, outcome_summary)
        candidate_payload = dict(meta_target)
        candidate_payload["objective_preset"] = preset
        candidate_payload["energy_profile_weights"] = derive_energy_profile_mix(
            semantic_summary,
            econ_signals={
                "energy_urgency": float(semantic_summary.get("efficiency_router_score", 0.0)),
            },
            objective_preset=preset,
        )
        counterfactual_id = f"cf_{sha256_json({'sample_id': sample_id, 'lane': 'meta_transformer', 'preset': preset})[:16]}"
        candidates.append(
            SemanticRuntimeCounterfactual(
                counterfactual_id=counterfactual_id,
                lane="meta_transformer",
                candidate={
                    "objective_preset": preset,
                    "energy_profile_weights": candidate_payload["energy_profile_weights"],
                },
                predicted_outcome_score=float(candidate_score),
                predicted_regret=float(chosen_score - candidate_score),
                executable=bool(outcome_summary.get("execution_ready", False)),
                rationale=f"counterfactual_objective_preset:{preset}",
            )
        )
    alternate_authority = "vla" if str(meta_target.get("authority_gt", "dino")) == "dino" else "dino"
    authority_score = _safe_float(meta_target.get(f"confidence_{alternate_authority}", 0.0))
    candidates.append(
        SemanticRuntimeCounterfactual(
            counterfactual_id=f"cf_{sha256_json({'sample_id': sample_id, 'lane': 'authority', 'authority': alternate_authority})[:16]}",
            lane="meta_transformer",
            candidate={"authority_gt": alternate_authority},
            predicted_outcome_score=float(authority_score),
            predicted_regret=float(chosen_score - authority_score),
            executable=True,
            rationale=f"counterfactual_authority:{alternate_authority}",
        )
    )
    ranked = sorted(candidates, key=lambda item: item.predicted_outcome_score, reverse=True)
    return ranked[: max(max_count, 1)]


def build_semantic_runtime_learning_row(
    episode: ReplayEpisodeRecord,
    *,
    steps: Optional[Sequence[ReplayStepRecord]] = None,
    windows: Optional[Sequence[ReplayWindowRecord]] = None,
    root_dir: Optional[str] = None,
    max_counterfactuals: int = 3,
) -> SemanticRuntimeLearningRow:
    artifact_refs = _extract_artifact_refs(episode.metadata, episode.provenance, episode.metadata.get("future_training_artifacts", {}))
    semantic_summary = dict(episode.metadata.get("semantic_world_model_summary", {}) or {})
    if not semantic_summary:
        semantic_world_model_path = _resolve_artifact_path(
            root_dir,
            artifact_refs.get("semantic_world_model_ref") or artifact_refs.get("semantic_world_model_path"),
        )
        semantic_summary = build_semantic_world_model_summary(_load_json(semantic_world_model_path))
    semantic_summary.setdefault("task_id", episode.task_id)
    vla_summary = _summarize_vla_lane(artifact_refs, root_dir)
    dino_summary = _summarize_dino_lane(artifact_refs, root_dir, semantic_summary)
    fusion_summary = _summarize_fusion_lane(episode, vla_summary, dino_summary)
    outcome_summary = _summarize_outcome(
        episode,
        steps or [],
        windows or [],
        semantic_summary,
        fusion_summary,
    )
    instruction = str(vla_summary.get("instruction", ""))
    meta_target = _build_meta_transformer_target(
        semantic_summary,
        vla_summary,
        dino_summary,
        fusion_summary,
        outcome_summary,
        instruction,
    )
    orchestration_target = _build_orchestration_target(
        semantic_summary,
        meta_target,
        outcome_summary,
        instruction,
    )
    counterfactuals = _build_counterfactuals(
        f"{episode.run_id}:{episode.episode_id}",
        semantic_summary,
        meta_target,
        outcome_summary,
        max_count=max_counterfactuals,
    )
    best_counterfactual = counterfactuals[0].predicted_outcome_score if counterfactuals else 0.0
    chosen_score = _preset_alignment_score(
        str(meta_target.get("objective_preset", "balanced")),
        semantic_summary,
        meta_target,
        outcome_summary,
    )
    feedback_summary = _feedback_summary(
        semantic_summary,
        vla_summary,
        dino_summary,
        fusion_summary,
        outcome_summary,
    )
    inferential_summary = {
        "preferred_authority": str(meta_target.get("authority_gt", "dino")),
        "chosen_route_score": float(chosen_score),
        "best_counterfactual_score": float(best_counterfactual),
        "estimated_regret": float(max(best_counterfactual - chosen_score, 0.0)),
        "route_success_label": bool(outcome_summary.get("success", False) and outcome_summary.get("work_order_ready", False)),
        "orchestration_route_success_label": bool(
            outcome_summary.get("success", False)
            and outcome_summary.get("work_order_ready", False)
            and bool(orchestration_target.get("tool_sequence", []))
        ),
        "authority_success_label": _authority_success_label(
            semantic_summary,
            vla_summary,
            dino_summary,
            fusion_summary,
            meta_target,
            outcome_summary,
        ),
        "semantic_gain_label": bool(outcome_summary.get("semantic_grounded", False)),
        "fusion_gain_label": bool(fusion_summary.get("fusion_advantage_score", 0.0) > 0.0),
        "feedback_edges": feedback_summary,
    }
    semantic_token_list = semantic_tokens(semantic_summary)
    sample_id = f"semantic_runtime_{sha256_json({'run_id': episode.run_id, 'episode_id': episode.episode_id, 'task_id': episode.task_id})[:16]}"
    return SemanticRuntimeLearningRow(
        sample_id=sample_id,
        run_id=episode.run_id,
        episode_id=episode.episode_id,
        task_id=episode.task_id,
        env_id=episode.env_id,
        source_domain=episode.source_domain,
        semantic_world_model_summary=semantic_summary,
        semantic_tokens=semantic_token_list,
        vla_summary=vla_summary,
        dino_summary=dino_summary,
        fusion_summary=fusion_summary,
        feedback_summary=feedback_summary,
        meta_transformer_target=meta_target,
        orchestration_transformer_target=orchestration_target,
        outcome_summary=outcome_summary,
        inferential_summary=inferential_summary,
        counterfactuals=counterfactuals,
        artifact_refs=artifact_refs,
        metadata={
            "episode_status": episode.status,
            "objective_tensor_summary": dict(episode.objective_tensor_summary),
            "econ_tensor_summary": dict(episode.econ_tensor_summary),
            "skill_mode": episode.skill_mode,
        },
    )


def build_semantic_runtime_learning_corpus(
    bundle: ReplayDatasetBundle,
    *,
    max_counterfactuals: int = 3,
) -> SemanticRuntimeLearningCorpus:
    steps_by_episode: Dict[str, List[ReplayStepRecord]] = {}
    for step in bundle.steps:
        steps_by_episode.setdefault(step.episode_id, []).append(step)
    windows_by_episode: Dict[str, List[ReplayWindowRecord]] = {}
    for window in bundle.windows:
        windows_by_episode.setdefault(window.episode_id, []).append(window)
    rows = [
        build_semantic_runtime_learning_row(
            episode,
            steps=steps_by_episode.get(episode.episode_id, []),
            windows=windows_by_episode.get(episode.episode_id, []),
            root_dir=bundle.root_dir,
            max_counterfactuals=max_counterfactuals,
        )
        for episode in bundle.episodes
    ]
    summary = {
        "row_count": len(rows),
        "source_domains": sorted({row.source_domain for row in rows}),
        "authority_distribution": {
            "dino": sum(1 for row in rows if row.meta_transformer_target.get("authority_gt") == "dino"),
            "vla": sum(1 for row in rows if row.meta_transformer_target.get("authority_gt") == "vla"),
        },
        "bounded_ready_count": sum(
            1 for row in rows if row.outcome_summary.get("execution_ready", False)
        ),
        "semantic_grounded_count": sum(
            1 for row in rows if row.outcome_summary.get("semantic_grounded", False)
        ),
        "route_success_count": sum(
            1 for row in rows if row.inferential_summary.get("route_success_label", False)
        ),
        "authority_success_count": sum(
            1 for row in rows if row.inferential_summary.get("authority_success_label", False)
        ),
        "mean_quality_score": float(
            sum(_safe_float(row.outcome_summary.get("quality_score", 0.0)) for row in rows) / float(max(len(rows), 1))
        ),
        "mean_estimated_regret": float(
            sum(_safe_float(row.inferential_summary.get("estimated_regret", 0.0)) for row in rows) / float(max(len(rows), 1))
        ),
        "manifest_summary": dict(bundle.manifest.metadata),
    }
    return SemanticRuntimeLearningCorpus(rows=rows, summary=summary)


def build_meta_transformer_runtime_dataset(
    rows: Sequence[SemanticRuntimeLearningRow],
) -> List[MetaTransformerSample]:
    samples: List[MetaTransformerSample] = []
    for row in rows:
        semantic_feature_vec = encode_semantic_world_model_features(row.semantic_world_model_summary)
        vla_embedding = np.array(
            [
                _safe_float(row.vla_summary.get("vla_confidence_mean", 0.0)),
                1.0 if row.vla_summary.get("vla_available", False) else 0.0,
                _safe_float(row.vla_summary.get("teacher_confidence_mean", 0.0)),
                float(len(row.vla_summary.get("vla_object_refs", []) or [])) / 8.0,
                float(len(row.vla_summary.get("vla_affordance_hints", []) or [])) / 8.0,
                float(len(row.vla_summary.get("vla_risk_hints", []) or [])) / 8.0,
                _safe_float(row.fusion_summary.get("semantic_fusion_confidence_mean", 0.0)),
                _safe_float(row.fusion_summary.get("annotation_agreement_score", 0.0)),
            ],
            dtype=np.float32,
        )
        dino_embedding = np.concatenate(
            [
                semantic_feature_vec.astype(np.float32),
                np.array(
                    [
                        _safe_float(row.dino_summary.get("dino_proxy_confidence_mean", 0.0)),
                        _safe_float(row.dino_summary.get("scene_track_label_confidence_mean", 0.0)),
                        _safe_float(row.dino_summary.get("map_first_confidence_mean", 0.0)),
                        float(row.dino_summary.get("scene_track_count", 0)) / 8.0,
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        samples.append(
            MetaTransformerSample(
                sample_id=row.sample_id,
                vla_embedding=vla_embedding,
                dino_embedding=dino_embedding,
                semantic_tokens=list(row.semantic_tokens),
                authority_gt=str(row.meta_transformer_target.get("authority_gt", "dino")),
                confidence_vla=float(row.meta_transformer_target.get("confidence_vla", 0.0)),
                confidence_dino=float(row.meta_transformer_target.get("confidence_dino", 0.0)),
                task_context={
                    "task_id": row.task_id,
                    "env_id": row.env_id,
                    "source_domain": row.source_domain,
                    "objective_preset": row.meta_transformer_target.get("objective_preset", "balanced"),
                    "quality_score": row.outcome_summary.get("quality_score", 0.0),
                    "feedback_summary": row.feedback_summary,
                },
            )
        )
    return samples


def _profile_summaries_from_energy_mix(weights: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    normalized = _normalize_weights(weights)
    base_profiles = {
        "BASE": {"mpl": 60.0, "error": 0.03, "energy_Wh": 12.0, "risk": 0.2},
        "BOOST": {"mpl": 72.0, "error": 0.08, "energy_Wh": 20.0, "risk": 0.35},
        "SAVER": {"mpl": 50.0, "error": 0.04, "energy_Wh": 8.0, "risk": 0.18},
        "SAFE": {"mpl": 46.0, "error": 0.015, "energy_Wh": 14.0, "risk": 0.08},
    }
    for name, profile in base_profiles.items():
        profile["weight"] = normalized.get(name, 0.0)
    return base_profiles


def build_orchestration_runtime_dataset(
    rows: Sequence[SemanticRuntimeLearningRow],
) -> List[OrchestrationSample]:
    samples: List[OrchestrationSample] = []
    for row in rows:
        objective_preset = str(row.meta_transformer_target.get("objective_preset", "balanced"))
        context = OrchestratorContext(
            env_name=row.env_id,
            engine_type=str(row.orchestration_transformer_target.get("chosen_backend", "pybullet")),
            task_type=row.task_id,
            customer_segment="semantic_runtime",
            market_region=str(row.metadata.get("market_region", "US")),
            objective_vector=_objective_vector_from_preset(objective_preset),
            wage_human=float(row.metadata.get("wage_human", 20.0)),
            energy_price_kWh=float(row.metadata.get("energy_price_kWh", 0.12)),
            mean_delta_mpl=float(row.meta_transformer_target.get("expected_deltas", {}).get("expected_delta_mpl", 0.0)),
            mean_delta_error=float(row.meta_transformer_target.get("expected_deltas", {}).get("expected_delta_error", 0.0)),
            mean_delta_j=float(row.meta_transformer_target.get("expected_deltas", {}).get("expected_delta_energy_Wh", 0.0)),
            mean_trust=float(row.outcome_summary.get("quality_score", 0.0)),
            mean_w_econ=float(row.outcome_summary.get("reward_signal", 0.0)),
            profile_summaries=_profile_summaries_from_energy_mix(
                row.orchestration_transformer_target.get("energy_profile_weights", {})
            ),
            semantic_metadata={
                "semantic_world_model_summary": row.semantic_world_model_summary,
                "data_gaps": list(row.metadata.get("data_gaps", []) or []),
            },
        )
        context_features = np.asarray(_encode_ctx(context), dtype=np.float32)
        target_tool_sequence = [
            ToolCall(name=str(item.get("name")), args=dict(item.get("args", {})))
            for item in list(row.orchestration_transformer_target.get("tool_sequence", []) or [])
        ]
        samples.append(
            OrchestrationSample(
                context=context,
                context_features=np.asarray(context_features, dtype=np.float32),
                target_tool_sequence=target_tool_sequence,
                heuristic_rationale=[
                    f"score={_safe_float(item.get('score', 0.0)):.3f}"
                    for item in list(row.orchestration_transformer_target.get("tool_sequence", []) or [])
                ],
                metadata={
                    "sample_id": row.sample_id,
                    "source_domain": row.source_domain,
                    "objective_preset": objective_preset,
                    "quality_score": row.outcome_summary.get("quality_score", 0.0),
                    "feedback_summary": row.feedback_summary,
                },
                source_type="semantic_runtime_corpus",
            )
        )
    return samples


def write_semantic_runtime_learning_corpus(
    output_dir: str | Path,
    corpus: SemanticRuntimeLearningCorpus,
) -> Dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows_path = root / "semantic_runtime_learning_rows.jsonl"
    summary_path = root / "semantic_runtime_learning_summary.json"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in corpus.rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")
    summary_path.write_text(json.dumps(corpus.summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "rows_path": str(rows_path),
        "summary_path": str(summary_path),
    }


__all__ = [
    "SemanticRuntimeCounterfactual",
    "SemanticRuntimeLearningCorpus",
    "SemanticRuntimeLearningRow",
    "build_meta_transformer_runtime_dataset",
    "build_orchestration_runtime_dataset",
    "build_semantic_runtime_learning_corpus",
    "build_semantic_runtime_learning_row",
    "write_semantic_runtime_learning_corpus",
]

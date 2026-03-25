"""Benchmark-gating helpers for real-or-unavailable semantic/VLA lanes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.evidence.preconditions import ExecutionPreconditionsReport, build_execution_preconditions


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(payload or {})


def collect_benchmark_gating_signals(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    payload = _mapping(metadata)
    semantic_summary = _mapping(payload.get("semantic_world_model_summary"))
    topology = _mapping(semantic_summary.get("topology"))
    grounded_count = int(
        payload.get("grounded_track_object_count")
        or semantic_summary.get("grounded_track_object_count")
        or topology.get("grounded_track_object_count")
        or 0
    )
    scene_tracks_backend = str(payload.get("scene_tracks_backend", "") or "")
    teacher_backend = str(
        payload.get("openvla_backend_selected")
        or payload.get("teacher_runtime_backend_selected")
        or payload.get("teacher_backend_selected")
        or ""
    )
    vision_backend = str(
        payload.get("vision_backbone_selected")
        or payload.get("openvla_vision_backbone_selected")
        or payload.get("teacher_runtime_vision_backbone_selected")
        or ""
    )
    grounding_mode = str(
        payload.get("semantic_grounding_mode")
        or payload.get("grounding_mode")
        or payload.get("scene_tracks_backend")
        or ""
    )
    heuristic = bool(payload.get("semantic_grounding_heuristic", False)) or grounding_mode in {
        "heuristic",
        "heuristic_fallback",
        "keyword_tags",
    }
    semantic_memory_grounded = bool(payload.get("semantic_memory_grounded", False) or grounded_count > 0)
    semantic_grounding_non_heuristic = semantic_memory_grounded and not heuristic
    benchmark_eligible = (
        semantic_grounding_non_heuristic
        and scene_tracks_backend == "real"
        and teacher_backend != "stub"
        and vision_backend != "stub"
    )
    return {
        "scene_tracks_backend": scene_tracks_backend,
        "teacher_backend_selected": teacher_backend,
        "vision_backbone_selected": vision_backend,
        "grounded_track_object_count": grounded_count,
        "semantic_memory_grounded": semantic_memory_grounded,
        "semantic_grounding_non_heuristic": semantic_grounding_non_heuristic,
        "scene_tracks_backend_real": scene_tracks_backend == "real",
        "teacher_runtime_real": teacher_backend == "real",
        "vision_backbone_real": vision_backend == "real",
        "benchmark_eligible": benchmark_eligible,
    }


def build_benchmark_gate_report(
    *,
    subject_id: str,
    subject_kind: str,
    metadata: Optional[Mapping[str, Any]] = None,
    require_real_scene_tracks: bool = True,
    require_teacher_runtime: bool = False,
    require_vision_backbone: bool = False,
) -> ExecutionPreconditionsReport:
    payload = _mapping(metadata)
    signals = collect_benchmark_gating_signals(payload)
    blocked_reasons: list[str] = []

    scene_tracks_backend = str(signals.get("scene_tracks_backend", ""))
    if scene_tracks_backend == "stub":
        blocked_reasons.append("scene_tracks_stub_selected")
    if scene_tracks_backend == "passthrough" and require_real_scene_tracks:
        blocked_reasons.append("scene_tracks_passthrough_selected")

    teacher_backend = str(signals.get("teacher_backend_selected", ""))
    if teacher_backend == "stub":
        blocked_reasons.append("teacher_runtime_stub_selected")

    vision_backend = str(signals.get("vision_backbone_selected", ""))
    if vision_backend == "stub" and require_vision_backbone:
        blocked_reasons.append("vision_backbone_stub_selected")

    if not bool(signals.get("semantic_grounding_non_heuristic", False)):
        blocked_reasons.append("semantic_grounding_heuristic")

    required_boolean_signals = {
        "semantic_grounding_non_heuristic": True,
    }
    if require_real_scene_tracks:
        required_boolean_signals["scene_tracks_backend_real"] = True
    if require_teacher_runtime:
        required_boolean_signals["teacher_runtime_real"] = True
    if require_vision_backbone:
        required_boolean_signals["vision_backbone_real"] = True

    return build_execution_preconditions(
        subject_id=subject_id,
        subject_kind=subject_kind,
        artifact_refs={
            "semantic_world_model_summary": semantic_summary if (semantic_summary := _mapping(payload.get("semantic_world_model_summary"))) else {},
        },
        signal_values=signals,
        required_boolean_signals=required_boolean_signals,
        blocked_reasons=blocked_reasons,
        metadata={
            "require_real_scene_tracks": bool(require_real_scene_tracks),
            "require_teacher_runtime": bool(require_teacher_runtime),
            "require_vision_backbone": bool(require_vision_backbone),
        },
    )


__all__ = [
    "build_benchmark_gate_report",
    "collect_benchmark_gating_signals",
]

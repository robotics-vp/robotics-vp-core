from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.evidence import (
    EvidenceBus,
    EvidenceRecord,
    TeacherTrace,
    belief_state_from_evidence_bus,
    build_execution_preconditions,
    build_execution_work_order,
)
from src.motor_backend.rollout_capture import RolloutBundle
from src.orchestrator.semantic_fusion import SEMANTIC_FUSION_PREFIX, fuse_semantic_evidence_mvp
from src.semantic.runtime_backbone import SemanticRuntimeBackbone
from src.vision.map_first_supervision.artifacts import MAP_FIRST_PREFIX
from src.vision.map_first_supervision.semantics import parse_vla_semantic_evidence
from src.world_model import SemanticWorldModelBuilder

logger = logging.getLogger(__name__)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_trajectory_payload(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if "trajectory" not in data:
        return None
    payload = data["trajectory"]
    if hasattr(payload, "item") and payload.shape == ():
        payload = payload.item()
    if isinstance(payload, dict):
        return payload
    return None


def _get_scene_tracks_array(scene_tracks: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    if key in scene_tracks:
        return np.asarray(scene_tracks[key])
    prefixed = f"scene_tracks_v1/{key}"
    if prefixed in scene_tracks:
        return np.asarray(scene_tracks[prefixed])
    return None


def _extract_scene_track_ids(payload: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not payload:
        return None
    scene_tracks = payload.get("scene_tracks_v1") or payload.get("scene_tracks")
    if isinstance(scene_tracks, dict):
        return _get_scene_tracks_array(scene_tracks, "track_ids")
    scene_tracks_path = payload.get("scene_tracks_path") or payload.get("scene_tracks_npz")
    if scene_tracks_path:
        try:
            data = _load_npz(Path(scene_tracks_path))
        except Exception:
            return None
        return _get_scene_tracks_array(data, "track_ids")
    return None


def _find_map_first_path(trajectory_path: Path) -> Optional[Path]:
    candidates = [
        trajectory_path.with_name("map_first_supervision_v1.npz"),
        trajectory_path.with_name(f"{trajectory_path.stem}_map_first_v1.npz"),
        trajectory_path.with_name(f"{trajectory_path.stem}_map_first_supervision_v1.npz"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _find_vla_path(trajectory_path: Path) -> Optional[Path]:
    candidate = trajectory_path.with_name(f"{trajectory_path.stem}_vla_semantic_evidence_v1.npz")
    if candidate.exists():
        return candidate
    return None


def _find_teacher_trace_path(trajectory_path: Path) -> Optional[Path]:
    candidate = trajectory_path.with_name(f"{trajectory_path.stem}_teacher_trace_v1.json")
    if candidate.exists():
        return candidate
    return None


def _get_map_first_array(data: Optional[Dict[str, np.ndarray]], key: str) -> Optional[np.ndarray]:
    if data is None:
        return None
    return data.get(f"{MAP_FIRST_PREFIX}{key}")


def _to_track_list(track_ids: Optional[np.ndarray]) -> Optional[list[str]]:
    if track_ids is None:
        return None
    return [str(tid) for tid in list(track_ids)]


def _write_summary_jsonl(summary_path: Path, rows: list[Dict[str, Any]]) -> None:
    if not rows:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _update_episode_metadata(
    episode_dir: Path,
    metrics: Dict[str, float],
    fusion_path: Optional[Path],
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    meta_path = episode_dir / "metadata.json"
    if not meta_path.exists():
        return
    try:
        payload = json.loads(meta_path.read_text())
    except Exception:
        return
    existing_metrics = payload.get("metrics", {})
    if not isinstance(existing_metrics, dict):
        existing_metrics = {}
    existing_metrics.update(metrics)
    payload["metrics"] = existing_metrics
    if fusion_path is not None:
        payload["semantic_fusion_path"] = str(fusion_path)
    for key, value in dict(extra_fields or {}).items():
        payload[str(key)] = value
    meta_path.write_text(json.dumps(payload, indent=2))


def _safe_episode_id(episode_id: str) -> str:
    safe = episode_id.replace(os.sep, "_")
    if os.altsep:
        safe = safe.replace(os.altsep, "_")
    return safe or "episode"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _extract_teacher_trace(payload: Optional[Dict[str, Any]]) -> Optional[TeacherTrace]:
    if not isinstance(payload, dict):
        return None
    try:
        return TeacherTrace.from_dict(payload)
    except Exception:
        return None


def _write_degraded_fusion_artifact(
    *,
    trajectory_path: Path,
    episode_id: str,
    reason: str,
    artifact_refs: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    failure_path = trajectory_path.with_name(f"{_safe_episode_id(episode_id)}_semantic_degraded_v1.json")
    readiness = build_execution_preconditions(
        subject_id=episode_id,
        subject_kind="semantic_fusion_episode",
        artifact_refs=artifact_refs,
        required_artifact_refs=["trajectory_path"],
        blocked_reasons=[reason],
        metadata=metadata,
    )
    work_order = build_execution_work_order(
        order_type="semantic_fusion_repair",
        subject_id=episode_id,
        subject_kind="semantic_fusion_episode",
        decision="capture_negative_supervision",
        priority=1.0,
        recommended_mode="negative_counterexample",
        readiness=readiness,
        reasons=[reason],
        artifact_refs=artifact_refs,
        metadata=metadata,
    )
    _write_json(
        failure_path,
        {
            "episode_id": episode_id,
            "failure_reason": reason,
            "artifact_refs": dict(artifact_refs),
            "execution_preconditions": readiness.to_dict(),
            "execution_work_order": work_order.to_dict(),
            "metadata": dict(metadata or {}),
            "version": "semantic_degraded_v1",
        },
    )
    _update_episode_metadata(
        trajectory_path.parent,
        {"semantic_fusion_quality_score": 0.0},
        None,
        extra_fields={
            "semantic_fusion_failure_path": str(failure_path),
            "semantic_fusion_status": "blocked",
        },
    )
    return failure_path


def run_semantic_fusion_for_rollouts(
    rollout_bundle: RolloutBundle,
    summary_path: Optional[Path] = None,
    emit_semantic_fusion: bool = True,
    emit_evidence_bus: bool = True,
    emit_belief_state: bool = True,
    emit_semantic_world_model: bool = True,
    emit_semantic_snapshot: bool = True,
    emit_orchestrator_advisory: bool = True,
) -> list[Dict[str, Any]]:
    """Run semantic fusion for each episode in a rollout bundle."""
    summaries: list[Dict[str, Any]] = []
    semantic_world_model_builder = SemanticWorldModelBuilder()
    semantic_backbone = SemanticRuntimeBackbone({"write_to_file": False})
    for episode in rollout_bundle.episodes:
        trajectory_path = episode.trajectory_path
        map_first_path = _find_map_first_path(trajectory_path)
        vla_path = _find_vla_path(trajectory_path)
        teacher_trace_path = _find_teacher_trace_path(trajectory_path)

        map_first_data = _load_npz(map_first_path) if map_first_path else None
        vla_payload = _load_npz(vla_path) if vla_path else None
        teacher_trace = _extract_teacher_trace(_load_json(teacher_trace_path)) if teacher_trace_path else None

        scene_payload = _load_trajectory_payload(trajectory_path)
        scene_track_ids = _extract_scene_track_ids(scene_payload)

        vla_evidence = parse_vla_semantic_evidence(vla_payload)
        vla_class_probs = vla_evidence.class_probs if vla_evidence is not None else None
        vla_confidence = vla_evidence.confidence if vla_evidence is not None else None
        vla_track_ids = vla_evidence.track_ids if vla_evidence is not None else None

        if vla_class_probs is None:
            vla_class_probs = _get_map_first_array(map_first_data, "vla_class_probs")
        if vla_confidence is None:
            vla_confidence = _get_map_first_array(map_first_data, "vla_confidence")

        map_semantics = _get_map_first_array(map_first_data, "evidence_map_semantics")
        if map_semantics is None:
            map_semantics = _get_map_first_array(map_first_data, "semantics_stable")
        map_stability = _get_map_first_array(map_first_data, "evidence_map_stability")
        if map_stability is None:
            map_stability = _get_map_first_array(map_first_data, "meta_semantics_stability")

        geom_residual = _get_map_first_array(map_first_data, "evidence_geom_residual")
        occlusion = _get_map_first_array(map_first_data, "evidence_occlusion")
        dynamic_evidence = _get_map_first_array(map_first_data, "evidence_dynamics_score")

        artifact_refs = {
            "trajectory_path": str(trajectory_path),
            "map_first_path": str(map_first_path) if map_first_path is not None else None,
            "vla_semantic_evidence_path": str(vla_path) if vla_path is not None else None,
            "teacher_trace_path": str(teacher_trace_path) if teacher_trace_path is not None else None,
        }

        if map_semantics is None and vla_class_probs is None:
            failure_path = _write_degraded_fusion_artifact(
                trajectory_path=trajectory_path,
                episode_id=episode.metadata.episode_id,
                reason="missing_semantic_inputs",
                artifact_refs=artifact_refs,
            )
            summaries.append(
                {
                    "episode_id": episode.metadata.episode_id,
                    "semantic_fusion_status": "blocked",
                    "semantic_fusion_failure_reason": "missing_semantic_inputs",
                    "semantic_fusion_failure_path": str(failure_path),
                    "semantic_fusion_quality_score": 0.0,
                }
            )
            continue

        scene_ids = _to_track_list(scene_track_ids)
        vla_ids = _to_track_list(vla_track_ids)
        if scene_ids is not None and vla_ids is not None and scene_ids != vla_ids:
            logger.warning("Semantic fusion skipped: track_ids mismatch for episode %s", episode.metadata.episode_id)
            failure_path = _write_degraded_fusion_artifact(
                trajectory_path=trajectory_path,
                episode_id=episode.metadata.episode_id,
                reason="track_ids_mismatch",
                artifact_refs=artifact_refs,
                metadata={"scene_ids": scene_ids, "vla_ids": vla_ids},
            )
            summaries.append(
                {
                    "episode_id": episode.metadata.episode_id,
                    "semantic_fusion_status": "blocked",
                    "semantic_fusion_failure_reason": "track_ids_mismatch",
                    "semantic_fusion_failure_path": str(failure_path),
                    "semantic_fusion_quality_score": 0.0,
                }
            )
            continue
        if scene_ids is not None and map_semantics is not None and len(scene_ids) != map_semantics.shape[1]:
            logger.warning("Semantic fusion skipped: SceneTracks size mismatch for episode %s", episode.metadata.episode_id)
            failure_path = _write_degraded_fusion_artifact(
                trajectory_path=trajectory_path,
                episode_id=episode.metadata.episode_id,
                reason="scene_tracks_size_mismatch",
                artifact_refs=artifact_refs,
            )
            summaries.append(
                {
                    "episode_id": episode.metadata.episode_id,
                    "semantic_fusion_status": "blocked",
                    "semantic_fusion_failure_reason": "scene_tracks_size_mismatch",
                    "semantic_fusion_failure_path": str(failure_path),
                    "semantic_fusion_quality_score": 0.0,
                }
            )
            continue
        if vla_ids is not None and map_semantics is not None and len(vla_ids) != map_semantics.shape[1]:
            logger.warning("Semantic fusion skipped: VLA size mismatch for episode %s", episode.metadata.episode_id)
            failure_path = _write_degraded_fusion_artifact(
                trajectory_path=trajectory_path,
                episode_id=episode.metadata.episode_id,
                reason="vla_size_mismatch",
                artifact_refs=artifact_refs,
            )
            summaries.append(
                {
                    "episode_id": episode.metadata.episode_id,
                    "semantic_fusion_status": "blocked",
                    "semantic_fusion_failure_reason": "vla_size_mismatch",
                    "semantic_fusion_failure_path": str(failure_path),
                    "semantic_fusion_quality_score": 0.0,
                }
            )
            continue
        if vla_class_probs is not None and map_semantics is not None:
            if vla_class_probs.shape != map_semantics.shape:
                logger.warning("Semantic fusion skipped: shape mismatch for episode %s", episode.metadata.episode_id)
                failure_path = _write_degraded_fusion_artifact(
                    trajectory_path=trajectory_path,
                    episode_id=episode.metadata.episode_id,
                    reason="semantic_shape_mismatch",
                    artifact_refs=artifact_refs,
                )
                summaries.append(
                    {
                        "episode_id": episode.metadata.episode_id,
                        "semantic_fusion_status": "blocked",
                        "semantic_fusion_failure_reason": "semantic_shape_mismatch",
                        "semantic_fusion_failure_path": str(failure_path),
                        "semantic_fusion_quality_score": 0.0,
                    }
                )
                continue

        if scene_ids is None and vla_ids is None:
            logger.warning("Semantic fusion track_ids unavailable for episode %s; proceeding without alignment checks", episode.metadata.episode_id)

        num_classes = 1
        if map_semantics is not None:
            num_classes = int(map_semantics.shape[-1])
        elif vla_class_probs is not None:
            num_classes = int(vla_class_probs.shape[-1])

        fusion_result = fuse_semantic_evidence_mvp(
            vla_class_probs=vla_class_probs,
            vla_confidence=vla_confidence,
            map_semantics=map_semantics,
            map_stability=map_stability,
            geom_residual=geom_residual,
            occlusion=occlusion,
            dynamic_evidence=dynamic_evidence,
            num_classes=num_classes,
        )

        confidence_mean = float(np.mean(fusion_result.fused_confidence))
        disagreement_mean = float(fusion_result.diagnostics.get("disagreement_mean", 0.0)) if fusion_result.diagnostics else 0.0
        summary = {
            "episode_id": episode.metadata.episode_id,
            "semantic_fusion_status": "ready",
            "semantic_fusion_confidence_mean": confidence_mean,
            "semantic_disagreement_vla_vs_map": disagreement_mean,
            "semantic_fusion_quality_score": confidence_mean,
        }

        metrics = {
            "semantic_fusion_confidence_mean": confidence_mean,
            "semantic_disagreement_vla_vs_map": disagreement_mean,
            "semantic_fusion_quality_score": confidence_mean,
        }
        fusion_path: Optional[Path] = None
        if emit_semantic_fusion:
            episode_id = getattr(episode.metadata, "episode_id", "") or ""
            fusion_basename = f"{_safe_episode_id(episode_id)}_semantic_fusion_v1.npz"
            fusion_path = trajectory_path.with_name(fusion_basename)
            fusion_payload = fusion_result.to_npz()
            np.savez_compressed(fusion_path, **fusion_payload)
            summary.update(
                {
                    "semantic_fusion_path": str(fusion_path),
                    "semantic_fusion_keys": list(fusion_payload.keys()),
                    "semantic_fusion_prefix": SEMANTIC_FUSION_PREFIX,
                }
            )
        evidence_bus_path: Optional[Path] = None
        belief_state_path: Optional[Path] = None
        semantic_world_model_path: Optional[Path] = None
        semantic_snapshot_path: Optional[Path] = None
        orchestrator_advisory_path: Optional[Path] = None
        if emit_semantic_fusion and (emit_evidence_bus or emit_belief_state):
            episode_id = getattr(episode.metadata, "episode_id", "") or ""
            evidence_records: list[EvidenceRecord] = []
            if map_first_path is not None:
                evidence_records.append(
                    EvidenceRecord.from_components(
                        episode_id=episode_id,
                        timestamp=f"{episode_id}:semantic_fusion",
                        source="map_first",
                        kind="map_first_semantics",
                        confidence=float(np.mean(map_stability)) if map_stability is not None else 0.0,
                        disagreement=float(np.mean(geom_residual)) if geom_residual is not None else 0.0,
                        validity={"frame_count": int(map_semantics.shape[0]) if map_semantics is not None else 0},
                        metrics={
                            "map_first_quality_score": float(np.mean(map_stability)) if map_stability is not None else 0.0,
                        },
                        artifact_refs={"map_first_path": str(map_first_path)},
                    )
                )
            if vla_path is not None:
                vla_source = "vla_semantic_evidence"
                if vla_evidence is not None and isinstance(vla_evidence.provenance, dict):
                    vla_source = str(vla_evidence.provenance.get("source", vla_source))
                evidence_records.append(
                    EvidenceRecord.from_components(
                        episode_id=episode_id,
                        timestamp=f"{episode_id}:semantic_fusion",
                        source=vla_source,
                        kind="teacher_semantics",
                        confidence=float(np.mean(vla_confidence)) if vla_confidence is not None else 0.0,
                        disagreement=disagreement_mean,
                        validity={"frame_count": int(vla_class_probs.shape[0]) if vla_class_probs is not None else 0},
                        metrics={
                            "vla_confidence_mean": float(np.mean(vla_confidence)) if vla_confidence is not None else 0.0,
                        },
                        artifact_refs={"vla_semantic_evidence_path": str(vla_path)},
                        provenance=vla_evidence.provenance if vla_evidence is not None else {},
                    )
                )
            if teacher_trace is not None and teacher_trace_path is not None:
                teacher_confidence = float(
                    teacher_trace.summary.get("teacher_confidence_mean", 0.0)
                    or np.mean([step.confidence for step in teacher_trace.steps] or [0.0])
                )
                evidence_records.append(
                    EvidenceRecord.from_components(
                        episode_id=episode_id,
                        timestamp=f"{episode_id}:semantic_fusion",
                        source=teacher_trace.teacher_id,
                        kind="teacher_trace",
                        confidence=teacher_confidence,
                        disagreement=0.0,
                        validity={"step_count": len(teacher_trace.steps)},
                        metrics={"teacher_confidence_mean": teacher_confidence},
                        artifact_refs={"teacher_trace_path": str(teacher_trace_path)},
                        provenance=teacher_trace.provenance,
                    )
                )
            if fusion_path is not None:
                evidence_records.append(
                    EvidenceRecord.from_components(
                        episode_id=episode_id,
                        timestamp=f"{episode_id}:semantic_fusion",
                        source="semantic_fusion_runner",
                        kind="semantic_fusion",
                        confidence=confidence_mean,
                        disagreement=disagreement_mean,
                        validity={"frame_count": int(fusion_result.fused_confidence.shape[0])},
                        metrics=metrics,
                        artifact_refs={"semantic_fusion_path": str(fusion_path)},
                    )
                )

            evidence_bus = EvidenceBus(evidence_records)
            if emit_evidence_bus:
                evidence_bus_path = trajectory_path.with_name(f"{_safe_episode_id(episode_id)}_evidence_bus_v1.json")
                _write_json(evidence_bus_path, evidence_bus.to_dict())
                summary["evidence_bus_path"] = str(evidence_bus_path)

            if emit_belief_state:
                semantic_tags: list[str] = []
                if teacher_trace is not None and isinstance(teacher_trace.metadata, dict):
                    semantic_tags.extend(teacher_trace.metadata.get("semantic_tags", []) or [])
                if vla_evidence is not None and isinstance(vla_evidence.provenance, dict):
                    semantic_tags.extend(vla_evidence.provenance.get("semantic_tags", []) or [])
                belief_state = belief_state_from_evidence_bus(
                    evidence_bus=evidence_bus,
                    episode_id=episode_id,
                    timestamp=f"{episode_id}:semantic_fusion",
                    semantic_tags=semantic_tags,
                    artifact_refs={
                        "semantic_fusion_path": str(fusion_path) if fusion_path is not None else "",
                        "teacher_trace_path": str(teacher_trace_path) if teacher_trace_path is not None else "",
                    },
                    extra_state={
                        "scene_ir_quality": float(episode.metrics.get("scene_ir_quality", 0.0)),
                        "semantic_fusion_confidence_mean": confidence_mean,
                    },
                )
                belief_state_path = trajectory_path.with_name(
                    f"{_safe_episode_id(episode_id)}_belief_state_v1.json"
                )
                _write_json(belief_state_path, belief_state.to_dict())
                summary["belief_state_path"] = str(belief_state_path)
                if emit_semantic_world_model or emit_semantic_snapshot or emit_orchestrator_advisory:
                    objective_preset = (
                        getattr(episode.metadata, "objective_preset", None)
                        or episode.metrics.get("objective_preset")
                        or "balanced"
                    )
                    task_id = (
                        getattr(episode.metadata, "task_id", None)
                        or getattr(episode.metadata, "task_name", None)
                        or getattr(episode.metadata, "env_name", None)
                        or episode_id
                    )
                    semantic_world_model = semantic_world_model_builder.build_from_runtime_fusion(
                        episode_id=episode_id,
                        task_id=str(task_id),
                        objective_preset=str(objective_preset),
                        belief_state=belief_state,
                        semantic_tags=semantic_tags,
                        scene_tracks_payload=scene_payload,
                        teacher_trace=teacher_trace,
                        vla_semantic_evidence=vla_evidence if vla_evidence is not None else vla_payload,
                        semantic_fusion_summary={
                            "track_ids": scene_ids or vla_ids or [],
                            "semantic_fusion_path": str(fusion_path) if fusion_path is not None else "",
                            "confidence_mean": confidence_mean,
                            "disagreement_mean": disagreement_mean,
                            "scene_ir_quality": float(episode.metrics.get("scene_ir_quality", 0.0) or 0.0),
                        },
                        artifact_refs={
                            "semantic_fusion_path": str(fusion_path) if fusion_path is not None else "",
                            "belief_state_path": str(belief_state_path),
                            "evidence_bus_path": str(evidence_bus_path) if evidence_bus_path is not None else "",
                        },
                        metadata={
                            "source_stage": "semantic_fusion_runner",
                            "track_count": len(scene_ids or vla_ids or []),
                        },
                    )
                    backbone_result = semantic_backbone.build(
                        task_id=str(task_id),
                        objective_preset=str(objective_preset),
                        semantic_world_model=semantic_world_model,
                        runtime_metrics={
                            "avg_mpl_units_per_hour": confidence_mean * 100.0,
                            "avg_wage_parity": float(episode.metrics.get("wage_parity", 1.0) or 1.0),
                            "avg_energy_cost": float(episode.metrics.get("energy_cost", 0.0) or 0.0),
                            "avg_error_rate": disagreement_mean,
                            "mobility_drift_rate": float(episode.metrics.get("mobility_drift_rate", 0.0) or 0.0),
                            "recovery_segment_fraction": 1.0
                            if {"error_recovery", "mode:recovery"} & set(semantic_tags)
                            else 0.0,
                        },
                        frontier_episodes=[episode_id],
                        metadata={
                            "source_stage": "semantic_fusion_runner",
                            "semantic_fusion_path": str(fusion_path) if fusion_path is not None else "",
                            "belief_state_path": str(belief_state_path),
                        },
                        backends=[str(getattr(episode.metadata, "backend", "semantic_fusion"))],
                    )
                    backbone_paths = semantic_backbone.write_sidecars(
                        output_dir=trajectory_path.parent,
                        file_stem=_safe_episode_id(episode_id),
                        result=backbone_result,
                    )
                    semantic_world_model_path = Path(backbone_paths["semantic_world_model_path"])
                    semantic_snapshot_path = Path(backbone_paths["semantic_snapshot_path"])
                    orchestrator_advisory_path = Path(backbone_paths["orchestrator_advisory_path"])
                    if emit_semantic_world_model:
                        summary["semantic_world_model_path"] = str(semantic_world_model_path)
                    if emit_semantic_snapshot:
                        summary["semantic_snapshot_path"] = str(semantic_snapshot_path)
                    if emit_orchestrator_advisory:
                        summary["orchestrator_advisory_path"] = str(orchestrator_advisory_path)

        if emit_semantic_fusion:
            _update_episode_metadata(
                trajectory_path.parent,
                metrics,
                fusion_path,
                extra_fields={
                    "evidence_bus_path": str(evidence_bus_path) if evidence_bus_path is not None else None,
                    "belief_state_path": str(belief_state_path) if belief_state_path is not None else None,
                    "semantic_world_model_path": str(semantic_world_model_path) if semantic_world_model_path is not None else None,
                    "semantic_snapshot_path": str(semantic_snapshot_path) if semantic_snapshot_path is not None else None,
                    "orchestrator_advisory_path": str(orchestrator_advisory_path) if orchestrator_advisory_path is not None else None,
                },
            )

        summaries.append(summary)
        episode.metrics = dict(episode.metrics, **metrics)

    if summary_path is not None:
        _write_summary_jsonl(summary_path, summaries)

    return summaries

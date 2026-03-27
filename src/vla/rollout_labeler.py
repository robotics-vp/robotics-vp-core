from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.evidence.benchmark_gating import collect_benchmark_gating_signals
from src.evidence.preconditions import build_execution_preconditions
from src.evidence.scene_tracks_truth import (
    build_scene_tracks_provider_truth,
    scene_tracks_truth_from_metadata,
)
from src.evidence.teacher_trace import (
    TeacherStep,
    TeacherTrace,
    build_teacher_provider_truth,
    save_teacher_trace_json,
)
from src.motor_backend.datapacks import DatapackConfig, MotionClipSpec
from src.motor_backend.rollout_capture import EpisodeRollout, RolloutBundle
from src.sima2.semantic_primitive_extractor import extract_primitives_from_rollout
from src.vla.semantic_evidence import (
    build_vla_semantic_evidence_payload,
    save_vla_semantic_evidence_npz,
)
from src.vla.teacher_runtime import (
    OpenVLATeacherRuntime,
    TeacherActionEnvelope,
    TeacherAdapterContract,
    save_teacher_action_envelope_json,
    save_teacher_adapter_contract_json,
)

logger = logging.getLogger(__name__)


def label_rollouts_with_vla(
    rollouts: RolloutBundle,
    base_datapack: DatapackConfig,
) -> list[DatapackConfig]:
    """
    Call into the VLA/vision stack to label rollouts and produce new datapacks.
    """
    if not rollouts.episodes:
        return []

    derived_tags = set(base_datapack.tags)
    derived_tags.update({"auto_labeled", "vla_labeled"})
    derived_task_tags = set(base_datapack.task_tags)
    derived_robot_families = set(base_datapack.robot_families)
    derived_motion_clips: list[MotionClipSpec] = []
    derived_objective_hint = base_datapack.objective_hint
    primitive_tags: set[str] = set()
    risk_levels: set[str] = set()
    vla_tags: set[str] = set()
    episode_labeling_rows: list[dict[str, Any]] = []

    openvla_policy = _openvla_backend_policy()
    openvla_enabled = openvla_policy != "disabled"
    teacher_runtime = None
    teacher_contract = _fallback_teacher_contract(
        enabled=openvla_enabled,
        availability_reason="openvla_disabled" if not openvla_enabled else "",
        backend_policy=openvla_policy,
    )
    vla_error_reason = "openvla_disabled" if not openvla_enabled else None
    if openvla_enabled:
        try:
            teacher_runtime, vla_error_reason = _get_openvla_teacher_runtime()
            if teacher_runtime is not None:
                teacher_contract = teacher_runtime.describe_contract()
            elif teacher_contract is not None and vla_error_reason:
                teacher_contract = _fallback_teacher_contract(
                    enabled=True,
                    availability_reason=vla_error_reason,
                    backend_policy=openvla_policy,
                )
        except Exception as exc:
            vla_error_reason = str(exc)
            teacher_contract = _fallback_teacher_contract(
                enabled=True,
                availability_reason=vla_error_reason,
                backend_policy=openvla_policy,
            )
            logger.warning("OpenVLA initialization failed; teacher remains unavailable: %s", exc)

    for episode in rollouts.episodes:
        derived_motion_clips.append(MotionClipSpec(path=str(episode.trajectory_path), weight=1.0))
        if episode.metadata.robot_family:
            derived_robot_families.add(episode.metadata.robot_family)

        trajectory_payload = _load_trajectory_payload(episode.trajectory_path)
        rollout_dict = _build_rollout_dict(episode, base_datapack, trajectory_payload=trajectory_payload)
        primitives = extract_primitives_from_rollout(rollout_dict)
        episode_tags: set[str] = set()
        for prim in primitives:
            primitive_tags.update(prim.tags)
            episode_tags.update(prim.tags)
            risk_levels.add(prim.risk_level)
            derived_task_tags.update(_select_task_tags(prim.tags))

        teacher_envelope = None
        vla_action = None
        if teacher_contract is not None:
            teacher_envelope, vla_action_error = _try_openvla_action(
                teacher_runtime,
                teacher_contract,
                episode,
                base_datapack,
            )
            if teacher_envelope is not None:
                vla_action = teacher_envelope.to_vla_payload()
                vla_action_tags = _tags_from_vla_action(vla_action)
                vla_tags.update(vla_action_tags)
                episode_tags.update(vla_action_tags)
            if vla_action_error:
                vla_error_reason = vla_action_error

        artifact_refs = _write_vla_semantic_evidence_sidecar(
            episode=episode,
            semantic_tags=sorted(episode_tags),
            vla_action=vla_action,
            teacher_contract=teacher_contract,
            teacher_envelope=teacher_envelope,
            instruction=base_datapack.objective_hint or base_datapack.description or "",
            vla_error_reason=vla_error_reason,
        )
        episode_labeling_rows.append(
            _build_episode_labeling_row(
                episode=episode,
                base_datapack=base_datapack,
                trajectory_payload=trajectory_payload,
                semantic_tags=sorted(episode_tags),
                teacher_contract=teacher_contract,
                teacher_envelope=teacher_envelope,
                vla_error_reason=vla_error_reason,
                artifact_refs=artifact_refs,
            )
        )

        if derived_objective_hint is None:
            derived_objective_hint = _derive_objective_hint(primitives, episode.metrics)

    derived_tags.update(primitive_tags)
    derived_tags.update(vla_tags)
    if vla_error_reason:
        derived_tags.add("vla_error")

    description = base_datapack.description or ""
    if primitive_tags:
        summary = ", ".join(sorted(primitive_tags)[:3])
        description = f"{description} (VLA tags: {summary})".strip()
    elif not description:
        description = "Auto-labeled rollout datapack"

    derived_metadata, quality_score, novelty_score = _aggregate_labeling_metadata(
        rollouts=rollouts,
        base_datapack=base_datapack,
        semantic_tags=sorted(derived_tags),
        primitive_tags=sorted(primitive_tags),
        vla_tags=sorted(vla_tags),
        risk_levels=sorted(risk_levels),
        episode_rows=episode_labeling_rows,
        openvla_policy=openvla_policy,
    )
    derived = DatapackConfig(
        id=f"{base_datapack.id}_vla",
        description=description,
        motion_clips=derived_motion_clips or list(base_datapack.motion_clips),
        quality_score=quality_score,
        novelty_score=novelty_score,
        domain_randomization=dict(base_datapack.domain_randomization),
        curriculum=dict(base_datapack.curriculum),
        tags=sorted(derived_tags),
        task_tags=sorted(derived_task_tags),
        robot_families=sorted(derived_robot_families),
        objective_hint=derived_objective_hint or "auto-labeled",
        metadata=derived_metadata,
    )

    return [derived]


def _build_rollout_dict(
    episode: EpisodeRollout,
    base_datapack: DatapackConfig,
    *,
    trajectory_payload: Any | None = None,
) -> dict[str, Any]:
    rollout: dict[str, Any] = {
        "episode_id": episode.metadata.episode_id,
        "task": episode.metadata.task_id,
        "task_type": episode.metadata.task_id,
        "tags": list(base_datapack.tags) + list(base_datapack.task_tags),
        "metrics": dict(episode.metrics),
        "metadata": {"robot_family": episode.metadata.robot_family, "seed": episode.metadata.seed},
    }
    if trajectory_payload is None:
        trajectory_payload = _load_trajectory_payload(episode.trajectory_path)
    if isinstance(trajectory_payload, dict):
        for key in ("events", "segments", "primitive_events", "semantic_primitives", "primitives"):
            if key in trajectory_payload:
                rollout[key] = trajectory_payload[key]
    return rollout


def _load_trajectory_payload(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        import numpy as np

        data = np.load(path, allow_pickle=True)
        if "trajectory" not in data:
            return None
        payload = data["trajectory"]
        if hasattr(payload, "item") and payload.shape == ():
            return payload.item()
        return payload
    except Exception:
        return None


def _extract_scene_tracks_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    scene_tracks = payload.get("scene_tracks_v1") or payload.get("scene_tracks")
    if isinstance(scene_tracks, dict):
        return scene_tracks
    scene_tracks_path = payload.get("scene_tracks_path") or payload.get("scene_tracks_npz")
    if scene_tracks_path:
        try:
            import numpy as np

            data = dict(np.load(scene_tracks_path, allow_pickle=False))
            return data
        except Exception:
            return None
    return None


def _write_vla_semantic_evidence_sidecar(
    *,
    episode: EpisodeRollout,
    semantic_tags: list[str],
    vla_action: Optional[Mapping[str, Any]],
    teacher_contract: Optional[TeacherAdapterContract],
    teacher_envelope: Optional[TeacherActionEnvelope],
    instruction: str,
    vla_error_reason: Optional[str],
) -> dict[str, Any]:
    try:
        trajectory_payload = _load_trajectory_payload(episode.trajectory_path)
        scene_tracks = _extract_scene_tracks_payload(trajectory_payload)
        effective_semantic_tags = sorted(
            {
                str(tag)
                for tag in list(semantic_tags)
                + list(getattr(teacher_envelope, "semantic_tags", []) or [])
                if str(tag).strip()
            }
        )
        object_refs = list(getattr(teacher_envelope, "object_refs", []) or [])
        affordance_hints = list(getattr(teacher_envelope, "affordance_hints", []) or [])
        risk_hints = list(getattr(teacher_envelope, "risk_hints", []) or [])
        teacher_contract_ref = ""
        teacher_action_ref = ""
        if teacher_contract is not None:
            teacher_contract_path = episode.trajectory_path.with_name(
                f"{episode.trajectory_path.stem}_teacher_contract_v1.json"
            )
            save_teacher_adapter_contract_json(teacher_contract_path, teacher_contract)
            teacher_contract_ref = str(teacher_contract_path)
        if teacher_envelope is not None:
            teacher_action_path = episode.trajectory_path.with_name(
                f"{episode.trajectory_path.stem}_teacher_action_envelope_v1.json"
            )
            save_teacher_action_envelope_json(teacher_action_path, teacher_envelope)
            teacher_action_ref = str(teacher_action_path)

        teacher_trace = TeacherTrace.from_components(
            episode_id=episode.metadata.episode_id,
            teacher_id=teacher_contract.teacher_id if teacher_contract is not None else "openvla",
            modality="action_semantics",
            advisory_only=True,
            instruction=instruction,
            steps=[
                TeacherStep(
                    step_idx=0,
                    instruction=instruction,
                    action=dict(vla_action or {}),
                    confidence=float(vla_action.get("confidence", 0.0)) if isinstance(vla_action, Mapping) else 0.0,
                    semantic_tags=effective_semantic_tags,
                    artifact_refs={
                        "teacher_contract_ref": teacher_contract_ref,
                        "teacher_action_ref": teacher_action_ref,
                    },
                    metadata={
                        "availability_reason": str(vla_error_reason or ""),
                        "vla_available": bool(vla_action.get("vla_available", False)) if isinstance(vla_action, Mapping) else False,
                        "object_refs": object_refs,
                        "affordance_hints": affordance_hints,
                        "risk_hints": risk_hints,
                    },
                )
            ],
            summary={
                "teacher_confidence_mean": float(vla_action.get("confidence", 0.0)) if isinstance(vla_action, Mapping) else 0.0,
                "step_count": 1.0,
                "semantic_tag_count": float(len(effective_semantic_tags)),
                "object_ref_count": float(len(object_refs)),
            },
            provenance={
                "source": teacher_contract.teacher_id if teacher_contract is not None else "openvla",
                "contract_id": teacher_contract.contract_id if teacher_contract is not None else "",
                "availability_reason": str(vla_error_reason or ""),
            },
            metadata={
                "semantic_tags": effective_semantic_tags,
                "object_refs": object_refs,
                "affordance_hints": affordance_hints,
                "risk_hints": risk_hints,
                "teacher_contract_ref": teacher_contract_ref,
                "teacher_action_ref": teacher_action_ref,
            },
            provider_truth=(
                dict(getattr(teacher_envelope, "provider_truth", {}) or {})
                or dict(getattr(teacher_contract, "provider_truth", {}) or {})
                or build_teacher_provider_truth(
                    provider_id=teacher_contract.teacher_id if teacher_contract is not None else "openvla",
                    provider_name=teacher_contract.model_name if teacher_contract is not None else "openvla",
                    available=bool(vla_action.get("vla_available", False)) if isinstance(vla_action, Mapping) else False,
                    backend_selected=_teacher_backend_selected(teacher_contract),
                    fallback_mode=str(vla_error_reason or ""),
                    confidence=float(vla_action.get("confidence", 0.0)) if isinstance(vla_action, Mapping) else 0.0,
                )
            ),
        )
        teacher_trace_path = episode.trajectory_path.with_name(
            f"{episode.trajectory_path.stem}_teacher_trace_v1.json"
        )
        save_teacher_trace_json(teacher_trace_path, teacher_trace)
        evidence = build_vla_semantic_evidence_payload(
            scene_tracks=scene_tracks,
            vla_payload=vla_action,
            semantic_tags=effective_semantic_tags,
            instruction=instruction,
            teacher_trace_ref=str(teacher_trace_path),
            teacher_contract_ref=teacher_contract_ref,
            teacher_action_ref=teacher_action_ref,
        )
        evidence_path = episode.trajectory_path.with_name(
            f"{episode.trajectory_path.stem}_vla_semantic_evidence_v1.npz"
        )
        save_vla_semantic_evidence_npz(evidence_path, evidence)
        return {
            "teacher_contract_ref": teacher_contract_ref,
            "teacher_action_ref": teacher_action_ref,
            "teacher_trace_ref": str(teacher_trace_path),
            "vla_semantic_evidence_ref": str(evidence_path),
            "scene_tracks_ref": str(episode.trajectory_path),
        }
    except Exception as exc:
        logger.warning("Failed to write VLA semantic evidence sidecar: %s", exc)
    return {}


def _build_episode_labeling_row(
    *,
    episode: EpisodeRollout,
    base_datapack: DatapackConfig,
    trajectory_payload: Any,
    semantic_tags: Sequence[str],
    teacher_contract: Optional[TeacherAdapterContract],
    teacher_envelope: Optional[TeacherActionEnvelope],
    vla_error_reason: Optional[str],
    artifact_refs: Mapping[str, Any],
) -> dict[str, Any]:
    episode_metadata_payload = _load_episode_metadata_payload(episode)
    scene_tracks_payload = _extract_scene_tracks_payload(trajectory_payload)
    scene_tracks_metadata = _scene_tracks_metadata(trajectory_payload, episode_metadata_payload)
    scene_tracks_truth = scene_tracks_truth_from_metadata(scene_tracks_metadata)
    scene_tracks_provider_truth = build_scene_tracks_provider_truth(scene_tracks_metadata)
    teacher_backend = _teacher_backend_selected(teacher_contract)
    vision_backend = _teacher_vision_backend_selected(teacher_contract)
    teacher_provider_truth = (
        dict(getattr(teacher_envelope, "provider_truth", {}) or {})
        or dict(getattr(teacher_contract, "provider_truth", {}) or {})
        or build_teacher_provider_truth(
            provider_id=teacher_contract.teacher_id if teacher_contract is not None else "openvla",
            provider_name=teacher_contract.model_name if teacher_contract is not None else "openvla",
            available=bool(teacher_envelope is not None and teacher_envelope.available),
            backend_selected=teacher_backend,
            fallback_mode=str(vla_error_reason or ""),
            confidence=float(getattr(teacher_envelope, "confidence", 0.0) or 0.0),
            metadata={"vision_backbone_selected": vision_backend},
        )
    )
    grounded_track_object_count = _grounded_track_object_count(
        trajectory_payload=trajectory_payload,
        scene_tracks_payload=scene_tracks_payload,
        episode_metadata_payload=episode_metadata_payload,
    )
    semantic_memory_grounded = bool(
        grounded_track_object_count > 0
        or scene_tracks_truth.get("semantic_grounding_non_heuristic", False)
        or _bool_from_payload(
            trajectory_payload,
            episode_metadata_payload,
            key="semantic_memory_grounded",
        )
    )
    semantic_grounding_mode = (
        "non_heuristic"
        if scene_tracks_truth.get("semantic_grounding_non_heuristic", False)
        else "heuristic_fallback"
    )
    teacher_available = bool(teacher_envelope is not None and teacher_envelope.available)
    teacher_confidence = float(getattr(teacher_envelope, "confidence", 0.0) or 0.0)
    artifact_map = {
        key: value
        for key, value in dict(artifact_refs or {}).items()
        if value not in (None, "", [], {})
    }
    artifact_map["trajectory_path"] = str(episode.trajectory_path)
    if scene_tracks_payload is not None and not artifact_map.get("scene_tracks_ref"):
        artifact_map["scene_tracks_ref"] = str(episode.trajectory_path)
    return {
        "episode_id": episode.metadata.episode_id,
        "task_id": episode.metadata.task_id,
        "robot_family": episode.metadata.robot_family,
        "instruction": base_datapack.objective_hint or base_datapack.description or "",
        "semantic_tags": [str(tag) for tag in semantic_tags if str(tag).strip()],
        "semantic_tag_count": len([tag for tag in semantic_tags if str(tag).strip()]),
        "teacher_runtime_backend_selected": teacher_backend,
        "teacher_runtime_live": teacher_available,
        "teacher_confidence": teacher_confidence,
        "teacher_provider_truth": teacher_provider_truth,
        "vision_backbone_selected": vision_backend,
        "scene_tracks_backend": str(scene_tracks_truth.get("scene_tracks_backend", "") or ""),
        "scene_tracks_non_stub": bool(scene_tracks_truth.get("scene_tracks_non_stub", False)),
        "scene_tracks_provider_truth": scene_tracks_provider_truth,
        "semantic_grounding_non_heuristic": bool(
            scene_tracks_truth.get("semantic_grounding_non_heuristic", False)
        ),
        "semantic_grounding_mode": semantic_grounding_mode,
        "semantic_memory_grounded": semantic_memory_grounded,
        "grounded_track_object_count": grounded_track_object_count,
        "vla_error_reason": str(vla_error_reason or ""),
        "artifact_refs": artifact_map,
    }


def _aggregate_labeling_metadata(
    *,
    rollouts: RolloutBundle,
    base_datapack: DatapackConfig,
    semantic_tags: Sequence[str],
    primitive_tags: Sequence[str],
    vla_tags: Sequence[str],
    risk_levels: Sequence[str],
    episode_rows: Sequence[Mapping[str, Any]],
    openvla_policy: str,
) -> tuple[dict[str, Any], float, float]:
    rows = [dict(row) for row in episode_rows]
    scene_tracks_backend = _prefer_backend(rows, "scene_tracks_backend")
    teacher_backend = _prefer_backend(rows, "teacher_runtime_backend_selected")
    vision_backend = _prefer_backend(rows, "vision_backbone_selected")
    grounded_track_object_count = sum(
        int(_safe_float(row.get("grounded_track_object_count", 0.0), 0.0)) for row in rows
    )
    teacher_confidence_mean = _mean([row.get("teacher_confidence", 0.0) for row in rows])
    teacher_live_fraction = _fraction_true(rows, "teacher_runtime_live")
    scene_tracks_real_fraction = _fraction_true(rows, "scene_tracks_non_stub")
    semantic_grounding_fraction = _fraction_true(rows, "semantic_grounding_non_heuristic")
    semantic_tag_count_mean = _mean([row.get("semantic_tag_count", 0.0) for row in rows])
    artifact_refs = _aggregate_artifact_refs(rows)
    teacher_provider_truth = _prefer_provider_truth(rows, "teacher_provider_truth")
    scene_tracks_provider_truth = _prefer_provider_truth(rows, "scene_tracks_provider_truth")
    benchmark_payload = {
        "scene_tracks_backend": scene_tracks_backend,
        "teacher_runtime_backend_selected": teacher_backend,
        "vision_backbone_selected": vision_backend,
        "semantic_grounding_mode": (
            "non_heuristic" if semantic_grounding_fraction > 0.0 else "heuristic_fallback"
        ),
        "semantic_memory_grounded": grounded_track_object_count > 0 or semantic_grounding_fraction > 0.0,
        "grounded_track_object_count": grounded_track_object_count,
    }
    benchmark_signals = collect_benchmark_gating_signals(benchmark_payload)
    readiness = build_execution_preconditions(
        subject_id=f"{base_datapack.id}_vla",
        subject_kind="vla_labeled_datapack",
        artifact_refs=artifact_refs,
        required_artifact_refs=["teacher_trace_ref", "vla_semantic_evidence_ref"],
        soft_required_artifact_refs=["teacher_contract_ref", "teacher_action_ref", "scene_tracks_ref"],
        signal_values={
            **benchmark_signals,
            "teacher_runtime_live": teacher_live_fraction > 0.0,
            "teacher_confidence_mean": teacher_confidence_mean,
            "scene_tracks_non_stub_fraction": scene_tracks_real_fraction,
            "semantic_grounding_non_heuristic_fraction": semantic_grounding_fraction,
        },
        required_boolean_signals={
            "semantic_grounding_non_heuristic": True,
            "teacher_runtime_real": True,
            "vision_backbone_real": True,
        },
        soft_boolean_signals={"teacher_runtime_live": True},
        metadata={
            "openvla_backend_policy": openvla_policy,
            "episode_count": len(rows),
            "selection_contract": "vla_rollout_labeler_v2",
        },
    )
    prior_tags = {str(tag).strip().lower() for tag in base_datapack.tags if str(tag).strip()}
    new_tags = {str(tag).strip().lower() for tag in semantic_tags if str(tag).strip()} - prior_tags
    novelty_score = min(1.0, float(len(new_tags)) / float(max(len(prior_tags) + len(new_tags), 1)))
    artifact_completeness = _artifact_completeness(artifact_refs)
    semantic_tag_density = min(1.0, semantic_tag_count_mean / 8.0)
    quality_score = min(
        1.0,
        (0.4 * artifact_completeness)
        + (0.35 * teacher_confidence_mean)
        + (0.25 * semantic_tag_density),
    )
    metadata = {
        "labeler": {
            "source": "vla_rollout_labeler",
            "version": "v2",
            "openvla_backend_policy": openvla_policy,
        },
        "derived_from_datapack_id": base_datapack.id,
        "rollout_scenario_id": rollouts.scenario_id,
        "episode_count": len(rows),
        "quality_score_kind": "labeling_contract_proxy_v1",
        "novelty_score_kind": "semantic_tag_delta_v1",
        "quality_score_components": {
            "artifact_completeness": artifact_completeness,
            "teacher_confidence_mean": teacher_confidence_mean,
            "semantic_tag_density": semantic_tag_density,
        },
        "quality_score": quality_score,
        "novelty_score": novelty_score,
        "semantic_tags": list(semantic_tags),
        "primitive_tags": list(primitive_tags),
        "vla_tags": list(vla_tags),
        "risk_levels": list(risk_levels),
        "teacher_runtime_backend_selected": teacher_backend,
        "teacher_provider_truth": teacher_provider_truth,
        "vision_backbone_selected": vision_backend,
        "scene_tracks_backend": scene_tracks_backend,
        "scene_tracks_provider_truth": scene_tracks_provider_truth,
        "semantic_grounding_mode": (
            "non_heuristic" if semantic_grounding_fraction > 0.0 else "heuristic_fallback"
        ),
        "semantic_memory_grounded": bool(
            grounded_track_object_count > 0 or semantic_grounding_fraction > 0.0
        ),
        "grounded_track_object_count": grounded_track_object_count,
        "benchmark_signals": benchmark_signals,
        "execution_preconditions": readiness.to_dict(),
        "future_training_signals": {
            "teacher_runtime_live": teacher_live_fraction > 0.0,
            "scene_tracks_non_stub": scene_tracks_real_fraction > 0.0,
            "semantic_grounding_non_heuristic": semantic_grounding_fraction > 0.0,
            "benchmark_eligible": bool(benchmark_signals.get("benchmark_eligible", False)),
        },
        "future_training_artifacts": artifact_refs,
        "artifacts": _artifact_catalog(artifact_refs),
        "episodes": rows,
    }
    return metadata, quality_score, novelty_score


def _load_episode_metadata_payload(episode: EpisodeRollout) -> dict[str, Any]:
    metadata_path = episode.trajectory_path.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _scene_tracks_metadata(trajectory_payload: Any, episode_metadata_payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {}
    if isinstance(trajectory_payload, Mapping):
        metadata.update(dict(trajectory_payload))
    episode_metadata = episode_metadata_payload.get("metadata")
    if isinstance(episode_metadata, Mapping):
        metadata.update(dict(episode_metadata))
    for key in ("scene_tracks_backend", "scene_tracks_path", "scene_tracks_npz", "scene_tracks_v1", "scene_tracks"):
        if key in episode_metadata_payload and key not in metadata:
            metadata[key] = episode_metadata_payload.get(key)
    return metadata


def _grounded_track_object_count(
    *,
    trajectory_payload: Any,
    scene_tracks_payload: Optional[Dict[str, Any]],
    episode_metadata_payload: Mapping[str, Any],
) -> int:
    for payload in (trajectory_payload, episode_metadata_payload):
        if isinstance(payload, Mapping):
            direct = payload.get("grounded_track_object_count")
            if direct is not None:
                return int(_safe_float(direct, 0.0))
            summary = payload.get("semantic_world_model_summary")
            if isinstance(summary, Mapping):
                topology = summary.get("topology")
                if isinstance(topology, Mapping) and topology.get("grounded_track_object_count") is not None:
                    return int(_safe_float(topology.get("grounded_track_object_count"), 0.0))
    summary_payload = _scene_tracks_summary_payload(scene_tracks_payload)
    if isinstance(summary_payload, Mapping):
        direct = summary_payload.get("grounded_track_object_count")
        if direct is not None:
            return int(_safe_float(direct, 0.0))
        topology = summary_payload.get("topology")
        if isinstance(topology, Mapping) and topology.get("grounded_track_object_count") is not None:
            return int(_safe_float(topology.get("grounded_track_object_count"), 0.0))
    return 0


def _scene_tracks_summary_payload(scene_tracks_payload: Optional[Dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(scene_tracks_payload, Mapping):
        return {}
    for key in (
        "summary_json",
        "scene_tracks_v1/summary_json",
        "semantic_summary_json",
        "scene_tracks_v1/semantic_summary_json",
    ):
        if key not in scene_tracks_payload:
            continue
        value = scene_tracks_payload.get(key)
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except Exception:
                continue
            if isinstance(payload, Mapping):
                return dict(payload)
    return {}


def _teacher_backend_selected(teacher_contract: Optional[TeacherAdapterContract]) -> str:
    if teacher_contract is None:
        return ""
    provider_truth = dict(getattr(teacher_contract, "provider_truth", {}) or {})
    backend = str(provider_truth.get("backend_selected", "") or "").strip()
    if backend:
        return backend
    metadata = dict(getattr(teacher_contract, "metadata", {}) or {})
    backend_status = metadata.get("backend_status")
    if isinstance(backend_status, Mapping):
        backend = str(backend_status.get("backend_selected", "") or "").strip()
        if backend:
            return backend
    backend = str(metadata.get("backend_selected", "") or "").strip()
    if backend:
        return backend
    return "real" if teacher_contract.available else "unavailable"


def _teacher_vision_backend_selected(teacher_contract: Optional[TeacherAdapterContract]) -> str:
    if teacher_contract is None:
        return ""
    provider_truth = dict(getattr(teacher_contract, "provider_truth", {}) or {})
    provider_meta = provider_truth.get("metadata")
    if isinstance(provider_meta, Mapping):
        backend = str(provider_meta.get("vision_backbone_selected", "") or "").strip()
        if backend:
            return backend
    metadata = dict(getattr(teacher_contract, "metadata", {}) or {})
    backend_status = metadata.get("backend_status")
    if isinstance(backend_status, Mapping):
        backend = str(backend_status.get("vision_backbone_selected", "") or "").strip()
        if backend:
            return backend
    for key in (
        "vision_backbone_selected",
        "openvla_vision_backbone_selected",
        "teacher_runtime_vision_backbone_selected",
    ):
        backend = str(metadata.get(key, "") or "").strip()
        if backend:
            return backend
    return "real" if teacher_contract.available else "unavailable"


def _bool_from_payload(*payloads: Any, key: str) -> bool:
    for payload in payloads:
        if isinstance(payload, Mapping) and key in payload:
            return bool(payload.get(key))
    return False


def _aggregate_artifact_refs(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for row in rows:
        artifact_refs = row.get("artifact_refs")
        if not isinstance(artifact_refs, Mapping):
            continue
        for key, value in artifact_refs.items():
            if value in (None, "", [], {}):
                continue
            refs.setdefault(str(key), [])
            if str(value) not in refs[str(key)]:
                refs[str(key)].append(str(value))
    return refs


def _artifact_catalog(artifact_refs: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    return {
        "teacher_contracts": list(artifact_refs.get("teacher_contract_ref", []) or []),
        "teacher_actions": list(artifact_refs.get("teacher_action_ref", []) or []),
        "teacher_traces": list(artifact_refs.get("teacher_trace_ref", []) or []),
        "vla_semantic_evidence": list(artifact_refs.get("vla_semantic_evidence_ref", []) or []),
        "scene_tracks": list(artifact_refs.get("scene_tracks_ref", []) or []),
    }


def _artifact_completeness(artifact_refs: Mapping[str, Sequence[str]]) -> float:
    required = ("teacher_contract_ref", "teacher_trace_ref", "vla_semantic_evidence_ref")
    satisfied = sum(1 for key in required if artifact_refs.get(key))
    return float(satisfied) / float(len(required))


def _prefer_provider_truth(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    for row in rows:
        payload = row.get(key)
        if isinstance(payload, Mapping) and payload:
            if str(payload.get("backend_selected", "") or "") == "real":
                return dict(payload)
    for row in rows:
        payload = row.get(key)
        if isinstance(payload, Mapping) and payload:
            return dict(payload)
    return {}


def _fraction_true(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    true_count = sum(1 for row in rows if bool(row.get(key, False)))
    return float(true_count) / float(len(rows))


def _mean(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    total = sum(_safe_float(value, 0.0) for value in values)
    return total / float(len(values))


def _prefer_backend(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    values = [str(row.get(key, "") or "").strip() for row in rows if str(row.get(key, "") or "").strip()]
    if not values:
        return ""
    for preferred in ("real", "passthrough", "artifact_present_unknown", "unavailable", "disabled", "stub"):
        if preferred in values:
            return preferred
    return values[0]


def _select_task_tags(tags: list[str]) -> set[str]:
    allowlist = {
        "reach",
        "grasp",
        "lift",
        "place",
        "locomotion",
        "navigate",
        "carry",
        "inspect",
        "align",
        "pull",
        "push",
        "release",
        "approach",
    }
    return {tag for tag in tags if tag in allowlist}


def _derive_objective_hint(primitives: Sequence[Any], metrics: Mapping[str, Any]) -> str:
    if any(getattr(prim, "risk_level", "") == "high" for prim in primitives):
        return "reduce risk exposure"
    success_rate = _safe_float(metrics.get("success_rate"), default=1.0)
    error_rate = _safe_float(metrics.get("error_rate"), default=0.0)
    energy_kwh = _safe_float(metrics.get("energy_kwh_mean") or metrics.get("energy_kwh"), default=0.0)
    if error_rate >= 0.2 or success_rate <= 0.5:
        return "reduce errors"
    if energy_kwh >= 1.0:
        return "reduce energy usage"
    return "auto-labeled"


def _tags_from_vla_action(action: Mapping[str, Any]) -> set[str]:
    tags: set[str] = set()
    if action.get("vla_available"):
        tags.add("vla:available")
        if abs(_safe_float(action.get("gripper"), 0.0)) > 0.2:
            tags.add("vla:gripper_motion")
        if any(abs(_safe_float(action.get(axis), 0.0)) > 0.2 for axis in ("dx", "dy", "dz")):
            tags.add("vla:translation_motion")
    else:
        tags.add("vla:unavailable")
    return tags


def _openvla_backend_policy() -> str:
    explicit = os.getenv("OPENVLA_BACKEND_POLICY", "").strip().lower()
    if explicit in {"auto", "real", "disabled", "stub"}:
        return explicit
    for key in ("OPENVLA_ENABLE", "VLA_ENABLE"):
        raw = os.getenv(key, "")
        if raw.strip().lower() in {"1", "true", "yes"}:
            return "auto"
    return "disabled"


def _fallback_teacher_contract(
    *,
    enabled: bool,
    availability_reason: str,
    backend_policy: str,
) -> TeacherAdapterContract:
    return TeacherAdapterContract(
        teacher_id="openvla",
        model_name=os.getenv("OPENVLA_MODEL_NAME") or os.getenv("OPENVLA_MODEL") or "openvla/openvla-7b",
        modality="action_semantics",
        advisory_only=True,
        available=False,
        metadata={
            "enabled": bool(enabled),
            "availability_reason": str(availability_reason),
            "backend_policy": str(backend_policy),
            "backend_selected": "disabled" if not enabled else "unavailable",
        },
        provider_truth=build_teacher_provider_truth(
            provider_id="openvla",
            provider_name=os.getenv("OPENVLA_MODEL_NAME") or os.getenv("OPENVLA_MODEL") or "openvla/openvla-7b",
            available=False,
            backend_selected="disabled" if not enabled else "unavailable",
            fallback_mode=str(availability_reason or ("disabled" if not enabled else "unavailable")),
            confidence=0.0,
            metadata={
                "backend_policy": str(backend_policy),
                "failure_reason": str(availability_reason),
            },
        ),
    )


_OPENVLA_RUNTIME = None
_OPENVLA_INITIALIZED = False
_OPENVLA_ERROR: str | None = None


def _get_openvla_teacher_runtime() -> Tuple[OpenVLATeacherRuntime | None, str | None]:
    global _OPENVLA_RUNTIME, _OPENVLA_INITIALIZED, _OPENVLA_ERROR
    if _OPENVLA_INITIALIZED:
        return _OPENVLA_RUNTIME, _OPENVLA_ERROR
    _OPENVLA_INITIALIZED = True
    try:
        from src.vla.openvla_controller import OpenVLAConfig, OpenVLAController
    except Exception as exc:
        logger.warning("OpenVLA import failed; teacher remains unavailable: %s", exc)
        _OPENVLA_ERROR = str(exc)
        return None, _OPENVLA_ERROR
    model_name = os.getenv("OPENVLA_MODEL_NAME") or os.getenv("OPENVLA_MODEL") or "openvla/openvla-7b"
    backend_policy = _openvla_backend_policy()
    cfg = OpenVLAConfig(
        model_name=model_name,
        device=os.getenv("OPENVLA_DEVICE", "cuda:0"),
        dtype=os.getenv("OPENVLA_DTYPE", "bfloat16"),
        backend_policy=backend_policy,
        use_vision_backbone=os.getenv("OPENVLA_USE_VISION_BACKBONE", "").strip().lower() in {"1", "true", "yes"},
        vision_backbone_type=os.getenv("OPENVLA_VISION_BACKBONE_TYPE", "dino"),
        vision_backbone_model=os.getenv("OPENVLA_VISION_BACKBONE_MODEL", "facebook/dinov2-small"),
        vision_backbone_policy=os.getenv("OPENVLA_VISION_BACKBONE_POLICY", "auto"),
    )
    controller = OpenVLAController(cfg)
    controller.load_model()
    runtime = OpenVLATeacherRuntime(controller)
    _OPENVLA_RUNTIME = runtime
    if not controller.available:
        logger.warning(
            "OpenVLA unavailable; backend_selected=%s reason=%s",
            controller.backend_selected,
            controller.failure_reason,
        )
        _OPENVLA_ERROR = controller.failure_reason or "OpenVLA unavailable"
        return runtime, _OPENVLA_ERROR
    _OPENVLA_ERROR = None
    return runtime, None


def _try_openvla_action(
    teacher_runtime: OpenVLATeacherRuntime | None,
    teacher_contract: TeacherAdapterContract,
    episode: EpisodeRollout,
    base_datapack: DatapackConfig,
) -> Tuple[TeacherActionEnvelope | None, str | None]:
    instruction = base_datapack.objective_hint or base_datapack.description or "Execute the task safely."
    if teacher_runtime is None:
        unavailable = TeacherActionEnvelope.unavailable(
            teacher_id=teacher_contract.teacher_id,
            model_name=teacher_contract.model_name,
            instruction=instruction,
            failure_mode=str(teacher_contract.metadata.get("availability_reason", "teacher_unavailable")),
            metadata={
                "contract_id": teacher_contract.contract_id,
                "backend_selected": _teacher_backend_selected(teacher_contract),
                "backend_policy": str(teacher_contract.metadata.get("backend_policy", "")),
                "vision_backbone_selected": _teacher_vision_backend_selected(teacher_contract),
            },
        )
        return unavailable, unavailable.failure_mode
    frame = _load_first_frame(episode)
    if frame is None:
        missing_frame = TeacherActionEnvelope.unavailable(
            teacher_id=teacher_contract.teacher_id,
            model_name=teacher_contract.model_name,
            instruction=instruction,
            failure_mode="missing_frame",
            metadata={
                "contract_id": teacher_contract.contract_id,
                "backend_selected": _teacher_backend_selected(teacher_contract),
                "backend_policy": str(teacher_contract.metadata.get("backend_policy", "")),
                "vision_backbone_selected": _teacher_vision_backend_selected(teacher_contract),
            },
        )
        return missing_frame, "missing_frame"
    try:
        envelope = teacher_runtime.predict_action(frame, instruction)
        error = envelope.failure_mode if not envelope.available and envelope.failure_mode else None
        return envelope, error
    except Exception as exc:
        logger.warning("OpenVLA inference failed; teacher remains unavailable: %s", exc)
        unavailable = TeacherActionEnvelope.unavailable(
            teacher_id=teacher_contract.teacher_id,
            model_name=teacher_contract.model_name,
            instruction=instruction,
            failure_mode=str(exc),
            metadata={
                "contract_id": teacher_contract.contract_id,
                "backend_selected": _teacher_backend_selected(teacher_contract),
                "backend_policy": str(teacher_contract.metadata.get("backend_policy", "")),
                "vision_backbone_selected": _teacher_vision_backend_selected(teacher_contract),
            },
        )
        return unavailable, str(exc)


def _load_first_frame(episode: EpisodeRollout):
    if not episode.rgb_video_path:
        return None
    path = Path(episode.rgb_video_path)
    if not path.exists():
        return None
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        try:
            from PIL import Image

            return Image.open(path)
        except Exception:
            return None
    try:
        import imageio.v2 as imageio  # type: ignore[import-not-found]

        reader = imageio.get_reader(str(path))
        frame = reader.get_data(0)
        reader.close()
    except Exception:
        return None
    try:
        from PIL import Image

        return Image.fromarray(frame)
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

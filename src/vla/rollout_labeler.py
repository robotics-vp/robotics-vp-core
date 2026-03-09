from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.evidence.teacher_trace import TeacherStep, TeacherTrace, save_teacher_trace_json
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

    openvla_enabled = _openvla_enabled()
    teacher_runtime = None
    teacher_contract = _fallback_teacher_contract(
        enabled=openvla_enabled,
        availability_reason="openvla_disabled" if not openvla_enabled else "",
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
                )
        except Exception as exc:
            vla_error_reason = str(exc)
            teacher_contract = _fallback_teacher_contract(
                enabled=True,
                availability_reason=vla_error_reason,
            )
            logger.warning("OpenVLA initialization failed; falling back to stub labels: %s", exc)

    for episode in rollouts.episodes:
        derived_motion_clips.append(MotionClipSpec(path=str(episode.trajectory_path), weight=1.0))
        if episode.metadata.robot_family:
            derived_robot_families.add(episode.metadata.robot_family)

        rollout_dict = _build_rollout_dict(episode, base_datapack)
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

        _write_vla_semantic_evidence_sidecar(
            episode=episode,
            semantic_tags=sorted(episode_tags),
            vla_action=vla_action,
            teacher_contract=teacher_contract,
            teacher_envelope=teacher_envelope,
            instruction=base_datapack.objective_hint or base_datapack.description or "",
            vla_error_reason=vla_error_reason,
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

    derived = DatapackConfig(
        id=f"{base_datapack.id}_vla",
        description=description,
        motion_clips=derived_motion_clips or list(base_datapack.motion_clips),
        domain_randomization=dict(base_datapack.domain_randomization),
        curriculum=dict(base_datapack.curriculum),
        tags=sorted(derived_tags),
        task_tags=sorted(derived_task_tags),
        robot_families=sorted(derived_robot_families),
        objective_hint=derived_objective_hint or "auto-labeled",
    )

    return [derived]


def _build_rollout_dict(episode: EpisodeRollout, base_datapack: DatapackConfig) -> dict[str, Any]:
    rollout: dict[str, Any] = {
        "episode_id": episode.metadata.episode_id,
        "task": episode.metadata.task_id,
        "task_type": episode.metadata.task_id,
        "tags": list(base_datapack.tags) + list(base_datapack.task_tags),
        "metrics": dict(episode.metrics),
        "metadata": {"robot_family": episode.metadata.robot_family, "seed": episode.metadata.seed},
    }
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
) -> None:
    try:
        trajectory_payload = _load_trajectory_payload(episode.trajectory_path)
        scene_tracks = _extract_scene_tracks_payload(trajectory_payload)
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
                    semantic_tags=semantic_tags,
                    artifact_refs={
                        "teacher_contract_ref": teacher_contract_ref,
                        "teacher_action_ref": teacher_action_ref,
                    },
                    metadata={
                        "availability_reason": str(vla_error_reason or ""),
                        "vla_available": bool(vla_action.get("vla_available", False)) if isinstance(vla_action, Mapping) else False,
                    },
                )
            ],
            summary={
                "teacher_confidence_mean": float(vla_action.get("confidence", 0.0)) if isinstance(vla_action, Mapping) else 0.0,
                "step_count": 1.0,
            },
            provenance={
                "source": teacher_contract.teacher_id if teacher_contract is not None else "openvla",
                "contract_id": teacher_contract.contract_id if teacher_contract is not None else "",
                "availability_reason": str(vla_error_reason or ""),
            },
            metadata={
                "semantic_tags": semantic_tags,
                "teacher_contract_ref": teacher_contract_ref,
                "teacher_action_ref": teacher_action_ref,
            },
        )
        teacher_trace_path = episode.trajectory_path.with_name(
            f"{episode.trajectory_path.stem}_teacher_trace_v1.json"
        )
        save_teacher_trace_json(teacher_trace_path, teacher_trace)
        evidence = build_vla_semantic_evidence_payload(
            scene_tracks=scene_tracks,
            vla_payload=vla_action,
            semantic_tags=semantic_tags,
            instruction=instruction,
            teacher_trace_ref=str(teacher_trace_path),
            teacher_contract_ref=teacher_contract_ref,
            teacher_action_ref=teacher_action_ref,
        )
        evidence_path = episode.trajectory_path.with_name(
            f"{episode.trajectory_path.stem}_vla_semantic_evidence_v1.npz"
        )
        save_vla_semantic_evidence_npz(evidence_path, evidence)
    except Exception as exc:
        logger.warning("Failed to write VLA semantic evidence sidecar: %s", exc)


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


def _openvla_enabled() -> bool:
    for key in ("OPENVLA_ENABLE", "VLA_ENABLE"):
        raw = os.getenv(key, "")
        if raw.strip().lower() in {"1", "true", "yes"}:
            return True
    return False


def _fallback_teacher_contract(
    *,
    enabled: bool,
    availability_reason: str,
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
        },
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
        logger.warning("OpenVLA import failed; falling back to stub labels: %s", exc)
        _OPENVLA_ERROR = str(exc)
        return None, _OPENVLA_ERROR
    model_name = os.getenv("OPENVLA_MODEL_NAME") or os.getenv("OPENVLA_MODEL") or "openvla/openvla-7b"
    cfg = OpenVLAConfig(
        model_name=model_name,
        device=os.getenv("OPENVLA_DEVICE", "cuda:0"),
        dtype=os.getenv("OPENVLA_DTYPE", "bfloat16"),
    )
    controller = OpenVLAController(cfg)
    controller.load_model()
    runtime = OpenVLATeacherRuntime(controller)
    _OPENVLA_RUNTIME = runtime
    if not controller.available:
        logger.warning("OpenVLA unavailable; falling back to stub labels.")
        _OPENVLA_ERROR = "OpenVLA unavailable"
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
            metadata={"contract_id": teacher_contract.contract_id},
        )
        return unavailable, unavailable.failure_mode
    frame = _load_first_frame(episode)
    if frame is None:
        missing_frame = TeacherActionEnvelope.unavailable(
            teacher_id=teacher_contract.teacher_id,
            model_name=teacher_contract.model_name,
            instruction=instruction,
            failure_mode="missing_frame",
            metadata={"contract_id": teacher_contract.contract_id},
        )
        return missing_frame, "missing_frame"
    try:
        envelope = teacher_runtime.predict_action(frame, instruction)
        error = envelope.failure_mode if not envelope.available and envelope.failure_mode else None
        return envelope, error
    except Exception as exc:
        logger.warning("OpenVLA inference failed; falling back to stub labels: %s", exc)
        unavailable = TeacherActionEnvelope.unavailable(
            teacher_id=teacher_contract.teacher_id,
            model_name=teacher_contract.model_name,
            instruction=instruction,
            failure_mode=str(exc),
            metadata={"contract_id": teacher_contract.contract_id},
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

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.motor_backend.datapacks import DatapackConfig, MotionClipSpec
from src.motor_backend.rollout_capture import EpisodeRollout, RolloutBundle
from src.sima2.semantic_primitive_extractor import extract_primitives_from_rollout


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
    controller = _get_openvla_controller() if openvla_enabled else None

    for episode in rollouts.episodes:
        derived_motion_clips.append(MotionClipSpec(path=str(episode.trajectory_path), weight=1.0))
        if episode.metadata.robot_family:
            derived_robot_families.add(episode.metadata.robot_family)

        rollout_dict = _build_rollout_dict(episode, base_datapack)
        primitives = extract_primitives_from_rollout(rollout_dict)
        for prim in primitives:
            primitive_tags.update(prim.tags)
            risk_levels.add(prim.risk_level)
            derived_task_tags.update(_select_task_tags(prim.tags))

        if controller is not None:
            vla_action = _try_openvla_action(controller, episode, base_datapack)
            if vla_action:
                vla_tags.update(_tags_from_vla_action(vla_action))

        if derived_objective_hint is None:
            derived_objective_hint = _derive_objective_hint(primitives, episode.metrics)

    derived_tags.update(primitive_tags)
    derived_tags.update(vla_tags)

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


_OPENVLA_CONTROLLER = None
_OPENVLA_INITIALIZED = False


def _get_openvla_controller():
    global _OPENVLA_CONTROLLER, _OPENVLA_INITIALIZED
    if _OPENVLA_INITIALIZED:
        return _OPENVLA_CONTROLLER
    _OPENVLA_INITIALIZED = True
    try:
        from src.vla.openvla_controller import OpenVLAConfig, OpenVLAController
    except Exception:
        return None
    cfg = OpenVLAConfig(
        model_name=os.getenv("OPENVLA_MODEL", "openvla/openvla-7b"),
        device=os.getenv("OPENVLA_DEVICE", "cuda:0"),
        dtype=os.getenv("OPENVLA_DTYPE", "bfloat16"),
    )
    controller = OpenVLAController(cfg)
    controller.load_model()
    if not controller.available:
        return None
    _OPENVLA_CONTROLLER = controller
    return controller


def _try_openvla_action(
    controller: Any,
    episode: EpisodeRollout,
    base_datapack: DatapackConfig,
) -> Mapping[str, Any] | None:
    frame = _load_first_frame(episode)
    if frame is None:
        return None
    instruction = base_datapack.objective_hint or base_datapack.description or "Execute the task safely."
    try:
        return controller.predict_action(frame, instruction)
    except Exception:
        return None


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
        import imageio.v2 as imageio

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

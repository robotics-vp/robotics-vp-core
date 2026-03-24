"""
SceneTracks production runner for datapacks.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.evidence.preconditions import build_execution_preconditions
from src.ontology.datapack_registry import register_scene_tracks_artifact
from src.ontology.store import OntologyStore
from src.vision.scene_ir_tracker import SceneIRTracker, SceneIRTrackerConfig
from src.vision.scene_ir_tracker.io.datapack_frame_reader import (
    compute_datapack_frame_hash,
    read_datapack_frames,
)
from src.vision.scene_ir_tracker.quality.scene_tracks_quality import (
    SceneTracksQuality,
    SceneTracksQualityConfig,
    compute_scene_tracks_quality,
)
from src.vision.scene_ir_tracker.serialization import (
    compute_scene_ir_quality_score,
    deserialize_scene_tracks_v1,
    serialize_scene_tracks_v1,
)


@dataclass(frozen=True)
class SceneTracksRunResult:
    scene_tracks_path: Path
    quality: SceneTracksQuality
    scene_ir_quality: float
    frame_metadata: Dict[str, Any]
    registry_entry: Dict[str, Any]
    adapter_status: Dict[str, Any]


class SceneTracksQualityError(RuntimeError):
    """Raised when SceneTracks quality is below threshold."""


SCENE_TRACKS_RUNNER_VERSION = "scene_tracks_runner_v1"
_BACKEND_POLICIES = {"auto", "real", "passthrough", "stub"}


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        return
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        deterministic = getattr(torch, "use_deterministic_algorithms", None)
        if callable(deterministic):
            deterministic(True, warn_only=True)
    except Exception:
        return


def _normalize_backend_policy(value: Optional[str]) -> str:
    policy = str(value or "auto").strip().lower()
    return policy if policy in _BACKEND_POLICIES else "auto"


def _passthrough_ready(frames_contract: Any) -> bool:
    return any(bool(frame_masks) for frame_masks in list(getattr(frames_contract, "instance_masks", []) or []))


def _make_tracker_config(
    *,
    seed: Optional[int],
    use_stub_adapters: bool,
    zero_inference_passthrough: bool,
) -> SceneIRTrackerConfig:
    return SceneIRTrackerConfig(
        device="cpu",
        use_stub_adapters=use_stub_adapters,
        allow_fallbacks=False,
        sam3d_objects_config={"stub_seed": seed},
        sam3d_body_config={"stub_seed": seed},
        zero_inference_passthrough=zero_inference_passthrough,
    )


def _resolve_tracker(
    *,
    frames_contract: Any,
    seed: Optional[int],
    backend_policy: str,
    use_stub_adapters: Optional[bool],
    zero_inference_passthrough: Optional[bool],
) -> tuple[SceneIRTracker, SceneIRTrackerConfig, str, Optional[str]]:
    explicit_stub = use_stub_adapters is True
    explicit_passthrough = zero_inference_passthrough is True

    if explicit_stub:
        cfg = _make_tracker_config(seed=seed, use_stub_adapters=True, zero_inference_passthrough=False)
        return SceneIRTracker(cfg), cfg, "stub", None
    if explicit_passthrough:
        cfg = _make_tracker_config(seed=seed, use_stub_adapters=False, zero_inference_passthrough=True)
        return SceneIRTracker(cfg), cfg, "passthrough", None

    policy = _normalize_backend_policy(backend_policy)
    last_error: Optional[str] = None

    if policy in {"auto", "real"}:
        cfg = _make_tracker_config(seed=seed, use_stub_adapters=False, zero_inference_passthrough=False)
        try:
            return SceneIRTracker(cfg), cfg, "real", None
        except Exception as exc:
            last_error = str(exc)
            if policy == "real":
                raise RuntimeError(f"Real on-device SAM3D requested but unavailable: {exc}") from exc

    if policy in {"auto", "passthrough"}:
        if policy == "passthrough" or _passthrough_ready(frames_contract):
            cfg = _make_tracker_config(seed=seed, use_stub_adapters=False, zero_inference_passthrough=True)
            return SceneIRTracker(cfg), cfg, "passthrough", last_error
        if policy == "passthrough":
            raise RuntimeError(
                "Zero-inference passthrough requested but no segmentation/object masks are available."
            )

    if policy == "stub":
        cfg = _make_tracker_config(seed=seed, use_stub_adapters=True, zero_inference_passthrough=False)
        return SceneIRTracker(cfg), cfg, "stub", last_error

    detail = (
        f"real backend unavailable ({last_error}) and no segmentation/object masks were available "
        "for zero-inference passthrough."
        if last_error
        else "no eligible SceneTracks backend could be selected."
    )
    raise RuntimeError(f"Automatic SceneTracks backend resolution failed: {detail}")

def run_scene_tracks(
    *,
    datapack_path: str | Path,
    output_path: str | Path,
    seed: Optional[int] = None,
    max_frames: Optional[int] = None,
    camera: Optional[str] = None,
    mode: str = "rgb",
    ontology_root: str | Path = "data/ontology",
    min_quality: float = 0.2,
    allow_low_quality: bool = False,
    quality_config: Optional[SceneTracksQualityConfig] = None,
    use_stub_adapters: Optional[bool] = None,
    zero_inference_passthrough: Optional[bool] = None,
    backend_policy: str = "auto",
) -> SceneTracksRunResult:
    """Run SceneIRTracker on a datapack and persist SceneTracks_v1."""
    if seed is not None:
        _set_deterministic_seed(seed)
    frames_contract = read_datapack_frames(
        datapack_path,
        camera=camera,
        mode=mode,
        max_frames=max_frames,
        seed=seed,
    )

    env_stub_adapters = str(os.environ.get("SCENE_TRACKS_USE_STUB_ADAPTERS", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
    if use_stub_adapters is None:
        use_stub_adapters = True if env_stub_adapters else None
    if zero_inference_passthrough is None and str(
            os.environ.get("SCENE_TRACKS_ZERO_INFERENCE_PASSTHROUGH", "0")
        ).strip().lower() in {"1", "true", "yes"}:
        zero_inference_passthrough = True
    backend_policy = _normalize_backend_policy(os.environ.get("SCENE_TRACKS_BACKEND_POLICY", backend_policy))

    tracker, tracker_config, backend_selected, real_failure_reason = _resolve_tracker(
        frames_contract=frames_contract,
        seed=seed,
        backend_policy=backend_policy,
        use_stub_adapters=use_stub_adapters,
        zero_inference_passthrough=zero_inference_passthrough,
    )
    scene_tracks = tracker.process_episode(
        frames=frames_contract.frames,
        instance_masks=frames_contract.instance_masks,
        camera=frames_contract.camera_params,
        class_labels=frames_contract.class_labels,
        object_refs=frames_contract.object_refs,
        depth_frames=frames_contract.depth_frames,
    )
    adapter_status = tracker.adapter_status()

    scene_ir_quality = compute_scene_ir_quality_score(scene_tracks)
    payload = serialize_scene_tracks_v1(scene_tracks)
    quality = compute_scene_tracks_quality(
        payload,
        config=quality_config,
    )

    if quality.quality_score < min_quality and not allow_low_quality:
        raise SceneTracksQualityError(
            f"SceneTracks quality {quality.quality_score:.3f} below threshold {min_quality:.3f}"
        )

    output_path = _resolve_output_path(output_path, datapack_path, frames_contract.camera_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_summary = _summarize_scene_track_semantics(
        scene_tracks=scene_tracks,
        payload=payload,
        semantic_context=frames_contract.metadata.get("semantic_context"),
        quality=quality,
        scene_ir_quality=scene_ir_quality,
    )
    _apply_scene_track_semantics(payload, semantic_summary)
    np.savez_compressed(output_path, **payload)

    frame_meta = frames_contract.to_metadata()
    if frames_contract.metadata:
        frame_meta.update(frames_contract.metadata)
    frame_meta["datapack_hash"] = compute_datapack_frame_hash(frame_meta)
    runner_meta = {
        "version": SCENE_TRACKS_RUNNER_VERSION,
        "tracker_config": tracker_config.to_dict(),
        "run_config": {
            "seed": seed,
            "max_frames": max_frames,
            "camera": frames_contract.camera_name,
            "mode": mode,
            "backend_policy": backend_policy,
            "backend_selected": backend_selected,
            "use_stub_adapters": bool(tracker_config.use_stub_adapters),
            "zero_inference_passthrough": bool(tracker_config.zero_inference_passthrough),
            "real_backend_failure": real_failure_reason,
        },
    }
    frame_meta["runner"] = runner_meta
    frame_meta["runner_config_hash"] = _hash_payload(runner_meta)
    frame_meta["scene_ir_quality"] = float(scene_ir_quality)
    frame_meta["scene_tracks_quality"] = quality.to_dict()
    frame_meta["adapter_status"] = adapter_status
    frame_meta["semantic_summary"] = semantic_summary["summary"]
    frame_meta["semantic_tags"] = list(semantic_summary["summary"].get("semantic_tags", []))
    frame_meta["semantic_track_catalog"] = semantic_summary["track_catalog"]
    frame_meta["semantic_density_score"] = float(semantic_summary["summary"].get("semantic_density_score", 0.0))
    frame_meta["semantic_grounding_ready"] = bool(semantic_summary["summary"].get("grounding_ready", False))
    execution_preconditions = build_execution_preconditions(
        subject_id=str(_infer_episode_id(Path(datapack_path)) or output_path.stem),
        subject_kind="scene_tracks_run",
        artifact_refs={
            "scene_tracks_path": str(output_path),
            "datapack_path": str(datapack_path),
        },
        required_artifact_refs=["scene_tracks_path", "datapack_path"],
        signal_values={
            "scene_tracks_quality": float(quality.quality_score),
            "scene_ir_quality": float(scene_ir_quality),
            "use_stub_adapters": bool(tracker_config.use_stub_adapters),
            "quality_gate_passed": bool(quality.quality_score >= float(min_quality)),
            "semantic_density_score": float(semantic_summary["summary"].get("semantic_density_score", 0.0)),
            "class_label_coverage": float(semantic_summary["summary"].get("class_label_coverage", 0.0)),
            "semantic_grounding_ready": bool(semantic_summary["summary"].get("grounding_ready", False)),
            "scene_ir_backend_stub_free": bool(adapter_status.get("overall_mode") == "real"),
            "scene_ir_backend_passthrough": bool(adapter_status.get("overall_mode") == "passthrough"),
            "scene_ir_backend_auto_selected_real": bool(backend_policy == "auto" and backend_selected == "real"),
        },
        min_signal_thresholds={
            "scene_tracks_quality": float(min_quality),
            "scene_ir_quality": 0.2,
        },
        required_boolean_signals={
            "quality_gate_passed": True,
            "use_stub_adapters": False,
            "scene_ir_backend_stub_free": True,
        },
        metadata={"camera": frames_contract.camera_name},
    )
    frame_meta["execution_preconditions"] = execution_preconditions.to_dict()
    frame_meta["training_eligible"] = bool(execution_preconditions.ready)

    registry_entry = _register_artifact(
        datapack_path=Path(datapack_path),
        output_path=output_path,
        frame_meta=frame_meta,
        ontology_root=Path(ontology_root),
    )
    _update_datapack_metadata(Path(datapack_path), output_path, frame_meta)

    return SceneTracksRunResult(
        scene_tracks_path=output_path,
        quality=quality,
        scene_ir_quality=scene_ir_quality,
        frame_metadata=frame_meta,
        registry_entry=registry_entry,
        adapter_status=adapter_status,
    )


def _hash_payload(payload: Dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _resolve_output_path(
    output_path: str | Path,
    datapack_path: str | Path,
    camera_name: str,
) -> Path:
    out = Path(output_path)
    if out.suffix.lower() == ".npz":
        return out
    datapack = Path(datapack_path)
    episode_id = _infer_episode_id(datapack)
    if episode_id:
        filename = f"{episode_id}_{camera_name}_scene_tracks_v1.npz"
    else:
        filename = f"{camera_name}_scene_tracks_v1.npz"
    return out / filename


def _infer_episode_id(path: Path) -> Optional[str]:
    metadata = _load_metadata(path)
    meta = metadata.get("metadata") if isinstance(metadata, dict) else None
    if isinstance(meta, dict) and meta.get("episode_id"):
        return str(meta.get("episode_id"))
    if path.is_dir():
        return path.name
    return None


def _load_metadata(path: Path) -> Dict[str, Any]:
    meta_path = path / "metadata.json" if path.is_dir() else path.with_name("metadata.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _register_artifact(
    *,
    datapack_path: Path,
    output_path: Path,
    frame_meta: Dict[str, Any],
    ontology_root: Path,
) -> Dict[str, Any]:
    metadata = _load_metadata(datapack_path)
    meta = metadata.get("metadata") if isinstance(metadata, dict) else {}
    if not isinstance(meta, dict):
        meta = {}

    datapack_id = str(meta.get("episode_id") or datapack_path.name)
    task_id = str(meta.get("task_id") or "unknown_task")
    store = OntologyStore(root_dir=str(ontology_root))
    return register_scene_tracks_artifact(
        store=store,
        datapack_id=datapack_id,
        task_id=task_id,
        artifact_path=str(output_path),
        frame_metadata=frame_meta,
        source_datapack=str(datapack_path),
    )


def _update_datapack_metadata(
    datapack_path: Path,
    output_path: Path,
    frame_meta: Dict[str, Any],
) -> None:
    if not datapack_path.is_dir():
        return
    meta_path = datapack_path / "metadata.json"
    if not meta_path.exists():
        return
    try:
        payload = json.loads(meta_path.read_text())
    except Exception:
        return
    payload["scene_tracks_path"] = str(output_path)
    payload["scene_tracks_quality"] = frame_meta.get("scene_tracks_quality", {})
    payload["scene_ir_quality"] = frame_meta.get("scene_ir_quality", 0.0)
    payload["scene_tracks_semantic_summary"] = frame_meta.get("semantic_summary", {})
    payload["semantic_tags"] = sorted(
        {
            str(tag)
            for tag in list(payload.get("semantic_tags", []) or []) + list(frame_meta.get("semantic_tags", []) or [])
            if str(tag).strip()
        }
    )
    meta_path.write_text(json.dumps(payload, indent=2))


def _summarize_scene_track_semantics(
    *,
    scene_tracks: Any,
    payload: Dict[str, np.ndarray],
    semantic_context: Any,
    quality: SceneTracksQuality,
    scene_ir_quality: float,
) -> Dict[str, Any]:
    tracks = deserialize_scene_tracks_v1(payload)
    track_ids = [str(track_id) for track_id in list(tracks.track_ids)]
    class_ids = np.asarray(tracks.class_ids)
    entity_types = np.asarray(tracks.entity_types)
    poses_t = np.asarray(tracks.poses_t, dtype=np.float32)
    class_names = [str(name) for name in list(tracks.class_names or [])]

    context = dict(semantic_context or {})
    catalog = list(context.get("scene_object_catalog", []) or [])
    catalog_by_class: Dict[str, Dict[str, Any]] = {}
    duplicate_classes: set[str] = set()
    for item in catalog:
        class_name = str(item.get("class_name", ""))
        if not class_name:
            continue
        if class_name in catalog_by_class:
            duplicate_classes.add(class_name)
        else:
            catalog_by_class[class_name] = dict(item)

    track_catalog: list[Dict[str, Any]] = []
    aggregate_tags: set[str] = set(str(tag) for tag in context.get("semantic_tags", []) if str(tag).strip())
    class_histogram: Dict[str, int] = {}
    labeled_track_count = 0
    catalog_aligned_count = 0
    coverage_values: list[float] = []
    source_refs = _track_source_refs(scene_tracks, track_ids)

    for index, track_id in enumerate(track_ids):
        class_name = "unknown_object"
        if 0 <= int(class_ids[index]) < len(class_names):
            class_name = str(class_names[int(class_ids[index])])
        elif index < len(entity_types) and int(entity_types[index]) == 1:
            class_name = "human_body"
        class_name = str(class_name or "unknown_object")

        coverage = float(quality.track_coverage_ratio.get(track_id, 0.0))
        coverage_values.append(coverage)
        motion_score = 0.0
        if poses_t.ndim == 3 and index < poses_t.shape[1] and poses_t.shape[0] > 1:
            diffs = np.diff(poses_t[:, index, :], axis=0)
            motion_score = float(np.mean(np.linalg.norm(diffs, axis=-1)))

        catalog_meta = dict(catalog_by_class.get(class_name, {}))
        category = str(
            catalog_meta.get("category")
            or ("human_body" if index < len(entity_types) and int(entity_types[index]) == 1 else "tracked_object")
        )
        has_known_label = class_name not in {"unknown_object", ""}
        if has_known_label:
            labeled_track_count += 1
            class_histogram[class_name] = int(class_histogram.get(class_name, 0)) + 1
        if class_name in catalog_by_class:
            catalog_aligned_count += 1
        hint_object_id = str(source_refs.get(track_id, {}).get("source_object_id") or "")
        source_instance_id = str(source_refs.get(track_id, {}).get("source_instance_id") or "")
        if not hint_object_id and class_name in catalog_by_class and class_name not in duplicate_classes:
            hint_object_id = str(catalog_meta.get("object_id") or "")

        label_confidence = float(
            np.clip(
                0.1
                + 0.45 * coverage
                + 0.2 * (1.0 if has_known_label else 0.0)
                + 0.15 * (1.0 if class_name in catalog_by_class else 0.0)
                + 0.1 * (1.0 if quality.quality_score >= 0.4 else 0.0),
                0.0,
                1.0,
            )
        )
        semantic_tags = set(str(tag) for tag in catalog_meta.get("semantic_tags", []) if str(tag).strip())
        if has_known_label:
            semantic_tags.add(f"object:{class_name}")
        semantic_tags.add(f"category:{category}")
        semantic_tags.add("dynamic_track" if motion_score > 0.03 else "static_track")
        if coverage < 0.4:
            semantic_tags.add("low_coverage_track")
        affordances = [str(tag) for tag in catalog_meta.get("affordances", []) if str(tag).strip()]

        aggregate_tags.update(semantic_tags)
        track_catalog.append(
            {
                "track_id": track_id,
                "class_name": class_name,
                "category": category,
                "label_source": str(context.get("label_source") or "scene_ir_tracker"),
                "label_confidence": label_confidence,
                "coverage_ratio": coverage,
                "motion_score": motion_score,
                "semantic_tags": sorted(semantic_tags),
                "affordances": sorted(set(affordances)),
                "hint_object_id": hint_object_id,
                "source_instance_id": source_instance_id,
            }
        )

    class_label_coverage = float(labeled_track_count / max(len(track_ids), 1))
    catalog_alignment = float(catalog_aligned_count / max(len(track_ids), 1))
    mean_coverage = float(np.mean(coverage_values)) if coverage_values else 0.0
    semantic_density_score = float(
        np.clip(
            0.45 * class_label_coverage + 0.35 * catalog_alignment + 0.20 * mean_coverage,
            0.0,
            1.0,
        )
    )
    summary = {
        "track_count": int(len(track_ids)),
        "labeled_track_count": int(labeled_track_count),
        "catalog_object_count": int(len(catalog)),
        "class_histogram": class_histogram,
        "class_label_coverage": class_label_coverage,
        "catalog_alignment": catalog_alignment,
        "semantic_density_score": semantic_density_score,
        "scene_ir_quality": float(scene_ir_quality),
        "scene_tracks_quality": float(quality.quality_score),
        "grounding_ready": bool(semantic_density_score >= 0.35 and labeled_track_count > 0),
        "training_candidate": bool(
            semantic_density_score >= 0.5 and quality.quality_score >= 0.25 and scene_ir_quality >= 0.2
        ),
        "semantic_tags": sorted(aggregate_tags),
    }
    return {"summary": summary, "track_catalog": track_catalog}


def _apply_scene_track_semantics(
    payload: Dict[str, np.ndarray],
    semantic_summary: Dict[str, Any],
) -> None:
    track_catalog = list(semantic_summary.get("track_catalog", []) or [])
    payload["scene_tracks_v1/track_label_source"] = np.array(
        [str(item.get("label_source", "")) for item in track_catalog],
        dtype="U64",
    )
    payload["scene_tracks_v1/track_category"] = np.array(
        [str(item.get("category", "")) for item in track_catalog],
        dtype="U64",
    )
    payload["scene_tracks_v1/track_label_confidence"] = np.asarray(
        [float(item.get("label_confidence", 0.0)) for item in track_catalog],
        dtype=np.float32,
    )
    payload["scene_tracks_v1/track_motion_score"] = np.asarray(
        [float(item.get("motion_score", 0.0)) for item in track_catalog],
        dtype=np.float32,
    )
    payload["scene_tracks_v1/track_hint_object_id"] = np.array(
        [str(item.get("hint_object_id", "")) for item in track_catalog],
        dtype="U128",
    )
    payload["scene_tracks_v1/track_source_instance_id"] = np.array(
        [str(item.get("source_instance_id", "")) for item in track_catalog],
        dtype="U128",
    )
    payload["scene_tracks_v1/track_source_object_id"] = np.array(
        [str(item.get("hint_object_id", "")) for item in track_catalog],
        dtype="U128",
    )
    payload["scene_tracks_v1/track_semantic_tags_json"] = np.array(
        [json.dumps(item.get("semantic_tags", []), separators=(",", ":")) for item in track_catalog],
        dtype="U512",
    )
    payload["scene_tracks_v1/track_affordances_json"] = np.array(
        [json.dumps(item.get("affordances", []), separators=(",", ":")) for item in track_catalog],
        dtype="U512",
    )

    merged_summary: Dict[str, Any] = {}
    summary_key = "scene_tracks_v1/summary_json"
    if summary_key in payload and payload[summary_key].size > 0:
        try:
            merged_summary.update(json.loads(str(payload[summary_key][0])))
        except Exception:
            merged_summary = {}
    merged_summary.update(dict(semantic_summary.get("summary", {})))
    payload[summary_key] = np.array(
        [json.dumps(merged_summary, separators=(",", ":"))],
        dtype="U8192",
    )
    payload["scene_tracks_v1/semantic_summary_json"] = np.array(
        [json.dumps(semantic_summary.get("summary", {}), separators=(",", ":"))],
        dtype="U8192",
    )


def _track_source_refs(scene_tracks: Any, track_ids: list[str]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    tracks_dict = getattr(scene_tracks, "tracks", {}) or {}
    for track_id in track_ids:
        history = list(tracks_dict.get(track_id, []) or [])
        source_object_id = ""
        source_instance_id = ""
        for entity in history:
            source_object_id = str(getattr(entity, "source_object_id", "") or source_object_id)
            source_instance_id = str(getattr(entity, "source_instance_id", "") or source_instance_id)
            if source_object_id and source_instance_id:
                break
        mapping[str(track_id)] = {
            "source_object_id": source_object_id,
            "source_instance_id": source_instance_id,
        }
    return mapping

"""
SceneTracks production runner for datapacks.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

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
    serialize_scene_tracks_v1,
)


@dataclass(frozen=True)
class SceneTracksRunResult:
    scene_tracks_path: Path
    quality: SceneTracksQuality
    scene_ir_quality: float
    frame_metadata: Dict[str, Any]
    registry_entry: Dict[str, Any]


class SceneTracksQualityError(RuntimeError):
    """Raised when SceneTracks quality is below threshold."""


SCENE_TRACKS_RUNNER_VERSION = "scene_tracks_runner_v1"


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

    tracker_config = SceneIRTrackerConfig(
        device="cpu",
        use_stub_adapters=True,
        sam3d_objects_config={"stub_seed": seed},
        sam3d_body_config={"stub_seed": seed},
    )
    tracker = SceneIRTracker(tracker_config)
    scene_tracks = tracker.process_episode(
        frames=frames_contract.frames,
        instance_masks=frames_contract.instance_masks,
        camera=frames_contract.camera_params,
    )

    scene_ir_quality = compute_scene_ir_quality_score(scene_tracks)
    quality = compute_scene_tracks_quality(
        serialize_scene_tracks_v1(scene_tracks),
        config=quality_config,
    )

    if quality.quality_score < min_quality and not allow_low_quality:
        raise SceneTracksQualityError(
            f"SceneTracks quality {quality.quality_score:.3f} below threshold {min_quality:.3f}"
        )

    output_path = _resolve_output_path(output_path, datapack_path, frames_contract.camera_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_scene_tracks_v1(scene_tracks)
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
        },
    }
    frame_meta["runner"] = runner_meta
    frame_meta["runner_config_hash"] = _hash_payload(runner_meta)
    frame_meta["scene_ir_quality"] = float(scene_ir_quality)
    frame_meta["scene_tracks_quality"] = quality.to_dict()

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
    meta_path.write_text(json.dumps(payload, indent=2))

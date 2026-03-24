"""
Datapack frame reader and contract validation for SceneTracks.
"""
from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.vision.nag.types import CameraParams


class DatapackFrameError(ValueError):
    """Raised when datapack frames violate the contract."""


@dataclass
class DatapackFramesContract:
    """Validated frames bundle for SceneTracks production."""

    frames: List[np.ndarray]
    timestamps_s: List[float]
    camera_params: CameraParams
    camera_name: str
    instance_masks: List[Dict[str, np.ndarray]]
    class_labels: List[Dict[str, str]] = field(default_factory=list)
    object_refs: List[Dict[str, str]] = field(default_factory=list)
    depth_frames: Optional[List[np.ndarray]] = None
    segmentation_frames: Optional[List[np.ndarray]] = None
    frame_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_depth(self) -> bool:
        return bool(self.depth_frames)

    @property
    def has_segmentation(self) -> bool:
        return bool(self.segmentation_frames)

    @property
    def frame_range(self) -> Tuple[int, int]:
        if not self.frame_indices:
            return (0, max(len(self.frames) - 1, 0))
        return (self.frame_indices[0], self.frame_indices[-1])

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "frame_count": len(self.frames),
            "frame_range": list(self.frame_range),
            "frame_indices": list(self.frame_indices),
            "timestamps_s": list(self.timestamps_s),
            "has_depth": self.has_depth,
            "has_segmentation": self.has_segmentation,
        }


@dataclass
class SensorBundleReadResult:
    frames: List[np.ndarray]
    depth_frames: Optional[List[np.ndarray]]
    seg_frames: Optional[List[np.ndarray]]
    timestamps_s: Optional[List[float]]
    camera_params: Optional[CameraParams]
    metadata: Dict[str, Any] = field(default_factory=dict)


def read_datapack_frames(
    datapack_path: str | Path,
    *,
    camera: Optional[str] = None,
    mode: str = "rgb",
    max_frames: Optional[int] = None,
    seed: Optional[int] = None,
) -> DatapackFramesContract:
    """Read frames from datapack, enforcing the DatapackFramesContract."""
    path = Path(datapack_path)
    camera_name = camera or "front"

    sensor_bundle = _load_sensor_bundle(path, camera_name=camera_name)
    if sensor_bundle:
        frames = sensor_bundle.frames
        depth_frames = sensor_bundle.depth_frames
        seg_frames = sensor_bundle.seg_frames
    else:
        frames, depth_frames, seg_frames = _load_frames_from_path(path, camera_name=camera_name)
    if not frames:
        if mode == "vector_proxy":
            frames, depth_frames, seg_frames, proxy_camera_params = _render_vector_proxy_frames(
                path,
                camera_name=camera_name,
                max_frames=max_frames,
                seed=seed,
            )
            timestamps = _build_timestamps(len(frames))
            instance_masks = _build_instance_masks(seg_frames, len(frames))
            class_labels, object_refs, semantic_context = _build_semantic_context(
                path,
                metadata=_load_metadata(path),
                seg_frames=seg_frames,
                num_frames=len(frames),
            )
            contract = DatapackFramesContract(
                frames=frames,
                timestamps_s=timestamps,
                camera_params=proxy_camera_params,
                camera_name=camera_name,
                instance_masks=instance_masks,
                class_labels=class_labels,
                object_refs=object_refs,
                depth_frames=depth_frames,
                segmentation_frames=seg_frames,
                frame_indices=list(range(len(frames))),
                metadata={
                    "mode": mode,
                    "source": "vector_proxy",
                    "seed": seed,
                    "max_frames": max_frames,
                    "semantic_context": semantic_context,
                },
            )
            _validate_contract(contract)
            return contract
        raise DatapackFrameError(
            "No frames found in datapack. Provide rgb frames (rgb.mp4/rgb.npz) "
            "or run with --mode vector_proxy to synthesize frames."
        )

    timestamps = sensor_bundle.timestamps_s if sensor_bundle and sensor_bundle.timestamps_s else _load_timestamps(path, len(frames))
    indices = list(range(len(frames)))
    frames, depth_frames, seg_frames, timestamps, indices = _downsample(
        frames,
        depth_frames,
        seg_frames,
        timestamps,
        indices,
        max_frames=max_frames,
        seed=seed,
    )

    metadata = _load_metadata(path)
    _validate_camera(metadata, camera_name)
    camera_params: Optional[CameraParams] = None
    if sensor_bundle and sensor_bundle.camera_params is not None:
        camera_params = sensor_bundle.camera_params
    if camera_params is None:
        camera_params = _build_camera_params(
            camera_name=camera_name,
            height=frames[0].shape[0],
            width=frames[0].shape[1],
            num_frames=len(frames),
            metadata=metadata,
        )

    instance_masks = _build_instance_masks(seg_frames, len(frames))
    class_labels, object_refs, semantic_context = _build_semantic_context(
        path,
        metadata=metadata,
        seg_frames=seg_frames,
        num_frames=len(frames),
    )
    contract = DatapackFramesContract(
        frames=frames,
        timestamps_s=timestamps,
        camera_params=camera_params,
        camera_name=camera_name,
        instance_masks=instance_masks,
        class_labels=class_labels,
        object_refs=object_refs,
        depth_frames=depth_frames,
        segmentation_frames=seg_frames,
        frame_indices=indices,
        metadata={
            "mode": mode,
            "source": "sensor_bundle" if sensor_bundle else "datapack",
            "seed": seed,
            "max_frames": max_frames,
            "semantic_context": semantic_context,
        },
    )
    _validate_contract(contract)
    return contract


def _load_frames_from_path(
    path: Path,
    *,
    camera_name: str,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    if path.is_dir():
        return _load_frames_from_dir(path, camera_name=camera_name)
    if path.suffix.lower() == ".npz":
        return _load_frames_from_npz(path)
    return [], None, None


def _load_sensor_bundle(
    path: Path,
    *,
    camera_name: str,
) -> Optional[SensorBundleReadResult]:
    if not path.is_dir():
        return None
    rgb_dir = path / "rgb"
    if not rgb_dir.exists():
        return None
    rgb_path = _find_bundle_file(rgb_dir, camera_name)
    if rgb_path is None:
        return None

    frames = _read_npz_frames(rgb_path) if rgb_path.suffix.lower() == ".npz" else _read_npy_frames(rgb_path)
    if not frames:
        return None

    depth_frames = _read_bundle_frames(path / "depth", camera_name)
    seg_frames = _read_bundle_frames(path / "seg", camera_name)
    timestamps = _load_bundle_timestamps(path)
    camera_params = _load_bundle_camera_params(
        path=path,
        camera_name=camera_name,
        height=frames[0].shape[0],
        width=frames[0].shape[1],
        num_frames=len(frames),
    )
    return SensorBundleReadResult(
        frames=frames,
        depth_frames=depth_frames,
        seg_frames=seg_frames,
        timestamps_s=timestamps,
        camera_params=camera_params,
    )


def _find_bundle_file(root: Path, camera_name: str) -> Optional[Path]:
    for ext in (".npz", ".npy"):
        candidate = root / f"{camera_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def _read_bundle_frames(root: Path, camera_name: str) -> Optional[List[np.ndarray]]:
    if not root.exists():
        return None
    path = _find_bundle_file(root, camera_name)
    if path is None:
        return None
    if path.suffix.lower() == ".npz":
        return _read_npz_frames(path)
    return _read_npy_frames(path)


def _read_npy_frames(path: Path) -> List[np.ndarray]:
    try:
        arr = np.load(path)
    except Exception as exc:
        raise DatapackFrameError(f"Failed to load npy frames from {path}: {exc}") from exc
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = arr[np.newaxis, ...]
    return [np.asarray(frame) for frame in arr]


def _load_bundle_timestamps(path: Path) -> Optional[List[float]]:
    for name in ("timestamps_s.npy", "timestamps.npy"):
        ts_path = path / name
        if ts_path.exists():
            try:
                arr = np.load(ts_path)
                return [float(x) for x in np.asarray(arr).tolist()]
            except Exception:
                return None
    return None


def _load_bundle_camera_params(
    *,
    path: Path,
    camera_name: str,
    height: int,
    width: int,
    num_frames: int,
) -> Optional[CameraParams]:
    intrinsics = _read_bundle_intrinsics(path / "intrinsics", camera_name)
    extrinsics = _read_bundle_extrinsics(path / "extrinsics", camera_name)
    if intrinsics is None or extrinsics is None:
        return None

    fx = intrinsics.get("fx")
    fy = intrinsics.get("fy")
    cx = intrinsics.get("cx")
    cy = intrinsics.get("cy")
    width = int(intrinsics.get("width", width))
    height = int(intrinsics.get("height", height))

    world_from_cam = np.asarray(extrinsics, dtype=np.float32)
    if world_from_cam.ndim == 2:
        world_from_cam = world_from_cam[np.newaxis, ...]
    if world_from_cam.shape[0] == 1 and num_frames > 1:
        world_from_cam = np.repeat(world_from_cam, num_frames, axis=0)

    if None in (fx, fy, cx, cy):
        return None
    fx_value = float(fx) if fx is not None else float(width)
    fy_value = float(fy) if fy is not None else float(height)
    cx_value = float(cx) if cx is not None else float(width) / 2.0
    cy_value = float(cy) if cy is not None else float(height) / 2.0
    return CameraParams(
        fx=fx_value,
        fy=fy_value,
        cx=cx_value,
        cy=cy_value,
        width=width,
        height=height,
        world_from_cam=world_from_cam,
        camera_id=camera_name,
    )


def _read_bundle_intrinsics(root: Path, camera_name: str) -> Optional[Dict[str, Any]]:
    if not root.exists():
        return None
    json_path = root / f"{camera_name}.json"
    if json_path.exists():
        try:
            return json.loads(json_path.read_text())
        except Exception:
            return None
    npz_path = root / f"{camera_name}.npz"
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=False)
            return {k: float(data[k]) for k in data}
        except Exception:
            return None
    return None


def _read_bundle_extrinsics(root: Path, camera_name: str) -> Optional[np.ndarray]:
    if not root.exists():
        return None
    npy_path = root / f"{camera_name}.npy"
    if npy_path.exists():
        try:
            return np.load(npy_path)
        except Exception:
            return None
    npz_path = root / f"{camera_name}.npz"
    if npz_path.exists():
        try:
            data = np.load(npz_path, allow_pickle=False)
            if "world_from_cam" in data:
                return data["world_from_cam"]
        except Exception:
            return None
    return None


def _load_frames_from_dir(
    path: Path,
    *,
    camera_name: str,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    rgb_candidates = [
        path / f"rgb_{camera_name}.mp4",
        path / f"rgb_{camera_name}.npz",
        path / "rgb.mp4",
        path / "rgb.npz",
    ]
    depth_candidates = [
        path / f"depth_{camera_name}.mp4",
        path / f"depth_{camera_name}.npz",
        path / "depth.mp4",
        path / "depth.npz",
    ]
    seg_candidates = [
        path / f"seg_{camera_name}.npz",
        path / f"segmentation_{camera_name}.npz",
        path / "segmentation.npz",
    ]

    rgb_frames = _load_frames_from_candidates(rgb_candidates)
    depth_frames = _load_frames_from_candidates(depth_candidates)
    seg_frames = _load_frames_from_candidates(seg_candidates)

    if not rgb_frames:
        frames_dir = path / "frames"
        if frames_dir.exists():
            rgb_frames = _load_frames_from_dir_frames(frames_dir)
    return rgb_frames or [], depth_frames, seg_frames


def _load_frames_from_candidates(candidates: Iterable[Path]) -> Optional[List[np.ndarray]]:
    for cand in candidates:
        if not cand.exists():
            continue
        if cand.suffix.lower() == ".mp4":
            frames = _read_video_frames(cand)
            if frames:
                return frames
        if cand.suffix.lower() == ".npz":
            frames = _read_npz_frames(cand)
            if frames:
                return frames
    return None


def _load_frames_from_dir_frames(frames_dir: Path) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for npy in sorted(frames_dir.glob("*.npy")):
        try:
            frames.append(np.load(npy))
        except Exception:
            continue
    return frames


def _read_video_frames(path: Path) -> List[np.ndarray]:
    try:
        import imageio.v3 as iio  # type: ignore[import-not-found]
    except Exception as exc:
        raise DatapackFrameError(f"imageio is required to read video frames: {exc}") from exc
    try:
        frames = iio.imread(path)
    except Exception as exc:
        raise DatapackFrameError(f"Failed to read video frames from {path}: {exc}") from exc
    if frames is None:
        return []
    if frames.ndim == 3:
        frames = frames[np.newaxis, ...]
    return [np.asarray(frame) for frame in frames]


def _read_npz_frames(path: Path) -> List[np.ndarray]:
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise DatapackFrameError(f"Failed to load npz frames from {path}: {exc}") from exc
    for key in ("frames", "rgb", "rgb_frames"):
        if key in data:
            arr = np.asarray(data[key])
            if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                arr = arr[np.newaxis, ...]
            return [np.asarray(frame) for frame in arr]
    return []


def _load_frames_from_npz(path: Path) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    rgb = _read_npz_frames(path)
    return rgb, None, None


def _load_metadata(path: Path) -> Dict[str, Any]:
    if path.is_dir():
        meta_path = path / "metadata.json"
    else:
        meta_path = path.with_name("metadata.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _validate_camera(metadata: Dict[str, Any], camera_name: str) -> None:
    if not isinstance(metadata, dict):
        return
    camera_names = metadata.get("camera_names")
    if isinstance(camera_names, list) and camera_names:
        if camera_name not in camera_names:
            raise DatapackFrameError(
                f"Requested camera '{camera_name}' not found. Available: {camera_names}"
            )
    sensor_bundle = metadata.get("sensor_bundle")
    if isinstance(sensor_bundle, dict):
        bundle_cameras = sensor_bundle.get("cameras")
        if isinstance(bundle_cameras, list) and bundle_cameras:
            if camera_name not in bundle_cameras:
                raise DatapackFrameError(
                    f"Requested camera '{camera_name}' not found. Available: {bundle_cameras}"
                )
    default_camera = metadata.get("camera_name")
    if isinstance(default_camera, str) and default_camera and camera_name != default_camera:
        raise DatapackFrameError(
            f"Requested camera '{camera_name}' not found. Available: [{default_camera}]"
        )


def _load_timestamps(path: Path, num_frames: int) -> List[float]:
    bundle_ts = _load_bundle_timestamps(path)
    if bundle_ts and len(bundle_ts) == num_frames:
        return [float(t) for t in bundle_ts]
    metadata = _load_metadata(path)
    timestamps = metadata.get("frame_timestamps") or metadata.get("timestamps_s")
    if isinstance(timestamps, list) and len(timestamps) == num_frames:
        return [float(t) for t in timestamps]
    fps = metadata.get("fps") or metadata.get("frame_rate") or 30
    try:
        fps_val = float(fps)
    except Exception:
        fps_val = 30.0
    return _build_timestamps(num_frames, fps=fps_val)


def _build_timestamps(num_frames: int, fps: float = 30.0) -> List[float]:
    if num_frames <= 0:
        return []
    dt = 1.0 / max(fps, 1e-6)
    return [i * dt for i in range(num_frames)]


def _downsample(
    frames: List[np.ndarray],
    depth_frames: Optional[List[np.ndarray]],
    seg_frames: Optional[List[np.ndarray]],
    timestamps: List[float],
    indices: List[int],
    *,
    max_frames: Optional[int],
    seed: Optional[int],
) -> Tuple[
    List[np.ndarray],
    Optional[List[np.ndarray]],
    Optional[List[np.ndarray]],
    List[float],
    List[int],
]:
    if max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
        return frames, depth_frames, seg_frames, timestamps, indices

    if len(frames) == 0:
        return frames, depth_frames, seg_frames, timestamps, indices

    if max_frames == 1:
        chosen = [0]
    else:
        chosen = np.linspace(0, len(frames) - 1, num=max_frames, dtype=int).tolist()
    chosen = sorted(set(chosen))

    frames = [frames[i] for i in chosen]
    if depth_frames:
        depth_frames = [depth_frames[i] for i in chosen]
    if seg_frames:
        seg_frames = [seg_frames[i] for i in chosen]
    timestamps = [timestamps[i] for i in chosen]
    indices = [indices[i] for i in chosen]
    return frames, depth_frames, seg_frames, timestamps, indices


def _build_camera_params(
    *,
    camera_name: str,
    height: int,
    width: int,
    num_frames: int,
    metadata: Dict[str, Any],
) -> CameraParams:
    camera_params = metadata.get("camera_params")
    if isinstance(camera_params, dict) and "world_from_cam" in camera_params:
        try:
            return CameraParams(**camera_params)
        except Exception:
            pass

    default_pose = _default_camera_pose(camera_name)
    fov_deg = 60.0
    world_from_cam = CameraParams.from_single_pose(
        position=default_pose["position"],
        look_at=default_pose["look_at"],
        up=default_pose["up"],
        fov_deg=fov_deg,
        width=int(width),
        height=int(height),
        camera_id=camera_name,
    ).world_from_cam
    if num_frames > 1 and world_from_cam.shape[0] == 1:
        world_from_cam = np.repeat(world_from_cam, num_frames, axis=0)
    return CameraParams(
        fx=0.5 * width / math.tan(math.radians(fov_deg) / 2.0),
        fy=0.5 * height / math.tan(math.radians(fov_deg) / 2.0),
        cx=width / 2.0,
        cy=height / 2.0,
        height=int(height),
        width=int(width),
        world_from_cam=world_from_cam,
        camera_id=camera_name,
    )


def _default_camera_pose(camera_name: str) -> Dict[str, Tuple[float, float, float]]:
    if camera_name == "top":
        return {"position": (0.0, 0.0, 2.0), "look_at": (0.0, 0.0, 0.0), "up": (0.0, 1.0, 0.0)}
    if camera_name == "wrist":
        return {"position": (0.3, -0.3, 0.6), "look_at": (0.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)}
    return {"position": (0.0, -1.5, 1.0), "look_at": (0.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)}


def _build_instance_masks(
    seg_frames: Optional[List[np.ndarray]],
    num_frames: int,
) -> List[Dict[str, np.ndarray]]:
    if not seg_frames:
        return [{} for _ in range(num_frames)]
    masks: List[Dict[str, np.ndarray]] = []
    for frame in seg_frames:
        frame_masks: Dict[str, np.ndarray] = {}
        if frame.ndim == 3 and frame.shape[-1] in (1, 2, 3, 4):
            frame = frame[..., 0]
        unique_ids = [val for val in np.unique(frame) if val != 0]
        for uid in unique_ids:
            mask = frame == uid
            frame_masks[str(uid)] = mask
        masks.append(frame_masks)
    return masks


def _validate_contract(contract: DatapackFramesContract) -> None:
    if not contract.frames:
        raise DatapackFrameError("DatapackFramesContract requires at least one frame.")
    if len(contract.timestamps_s) != len(contract.frames):
        raise DatapackFrameError("Frame timestamps length must match frames length.")
    if not _is_monotonic(contract.timestamps_s):
        raise DatapackFrameError("Frame timestamps must be strictly increasing.")

    height, width = contract.frames[0].shape[:2]
    for frame in contract.frames:
        if frame.shape[:2] != (height, width):
            raise DatapackFrameError("All frames must share the same resolution.")
    if contract.depth_frames:
        if len(contract.depth_frames) != len(contract.frames):
            raise DatapackFrameError("Depth frames length must match RGB frames.")
        for depth in contract.depth_frames:
            if depth.shape[:2] != (height, width):
                raise DatapackFrameError("Depth frames must match RGB resolution.")
    if contract.class_labels and len(contract.class_labels) != len(contract.frames):
        raise DatapackFrameError("Class labels length must match RGB frames.")
    if contract.object_refs and len(contract.object_refs) != len(contract.frames):
        raise DatapackFrameError("Object refs length must match RGB frames.")


def _is_monotonic(values: List[float]) -> bool:
    if len(values) <= 1:
        return True
    return all(b > a for a, b in zip(values, values[1:]))


def _render_vector_proxy_frames(
    path: Path,
    *,
    camera_name: str,
    max_frames: Optional[int],
    seed: Optional[int],
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]], CameraParams]:
    trajectory = _load_trajectory_payload(path)
    scene_spec = _extract_scene_spec(trajectory)
    if scene_spec is None:
        raise DatapackFrameError(
            "Vector proxy requested but no scene_spec found in datapack. "
            "Include scene_spec in trajectory payload or provide RGB frames."
        )

    states = _extract_states(trajectory)
    try:
        from src.envs.workcell_env.observations.mujoco_render import render_workcell_frames
        return render_workcell_frames(
            scene_spec=scene_spec,
            states=states,
            camera_name=camera_name,
            max_frames=max_frames,
            seed=seed,
        )
    except Exception as exc:
        raise DatapackFrameError(f"Failed to render vector proxy frames: {exc}") from exc


def _load_trajectory_payload(path: Path) -> Optional[Dict[str, Any]]:
    traj_path = path / "trajectory.npz" if path.is_dir() else path
    if not traj_path.exists():
        return None
    if traj_path.suffix.lower() != ".npz":
        return None
    try:
        data = np.load(traj_path, allow_pickle=True)
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


def _extract_scene_spec(trajectory: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not trajectory:
        return None
    scene_spec = trajectory.get("scene_spec")
    if scene_spec is None:
        return None
    try:
        from src.envs.workcell_env.scene.scene_spec import WorkcellSceneSpec
        if isinstance(scene_spec, WorkcellSceneSpec):
            return scene_spec
        if isinstance(scene_spec, dict):
            return WorkcellSceneSpec.from_dict(scene_spec)
    except Exception:
        return None
    return None


def _extract_states(trajectory: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    if not trajectory:
        return None
    states = trajectory.get("states")
    if isinstance(states, list):
        return states
    return None


def _build_semantic_context(
    path: Path,
    *,
    metadata: Dict[str, Any],
    seg_frames: Optional[List[np.ndarray]],
    num_frames: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
    trajectory = _load_trajectory_payload(path)
    scene_spec = _extract_scene_spec(trajectory) or _extract_scene_spec_from_metadata(metadata)
    object_catalog = _build_scene_object_catalog(scene_spec)
    segmentation_map = _extract_segmentation_label_map(metadata)

    if not segmentation_map and object_catalog:
        segmentation_map = {
            str(index): dict(obj)
            for index, obj in enumerate(object_catalog, start=1)
        }

    class_labels: List[Dict[str, str]] = []
    object_refs: List[Dict[str, str]] = []
    known_class_names: set[str] = set()
    observed_object_refs: set[str] = set()

    if seg_frames:
        for frame in seg_frames[:num_frames]:
            labels_for_frame: Dict[str, str] = {}
            refs_for_frame: Dict[str, str] = {}
            frame_arr = frame[..., 0] if frame.ndim == 3 and frame.shape[-1] in (1, 2, 3, 4) else frame
            for raw_uid in [value for value in np.unique(frame_arr) if int(value) != 0]:
                uid = str(int(raw_uid))
                label_meta = segmentation_map.get(uid, {})
                class_name = _normalize_semantic_label(label_meta.get("class_name") or f"instance_{uid}")
                object_id = str(label_meta.get("object_id") or "")
                labels_for_frame[uid] = class_name
                if object_id:
                    refs_for_frame[uid] = object_id
                    observed_object_refs.add(object_id)
                if not class_name.startswith("instance_"):
                    known_class_names.add(class_name)
            class_labels.append(labels_for_frame)
            object_refs.append(refs_for_frame)

    while len(class_labels) < num_frames:
        class_labels.append({})
    while len(object_refs) < num_frames:
        object_refs.append({})

    semantic_tags: set[str] = set()
    task_id = str(
        (metadata.get("metadata") or {}).get("task_id")
        if isinstance(metadata.get("metadata"), dict)
        else metadata.get("task_id")
        or ""
    )
    if task_id:
        semantic_tags.add(f"task:{_normalize_semantic_label(task_id)}")
    semantic_tags.update(
        str(tag)
        for tag in (metadata.get("semantic_tags") or metadata.get("tags") or [])
        if str(tag).strip()
    )
    for obj in object_catalog:
        semantic_tags.update(str(tag) for tag in obj.get("semantic_tags", []) if str(tag).strip())
    for class_name in known_class_names:
        semantic_tags.add(f"object:{class_name}")

    label_source = "scene_spec_ordered" if object_catalog else "unlabeled"
    if _extract_segmentation_label_map(metadata):
        label_source = "metadata_segmentation_map"

    task_metadata = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
    semantic_context = {
        "task_id": str(task_metadata.get("task_id") or metadata.get("task_id") or ""),
        "episode_id": str(task_metadata.get("episode_id") or metadata.get("episode_id") or path.name),
        "semantic_tags": sorted(tag for tag in semantic_tags if tag),
        "label_source": label_source,
        "scene_object_catalog": object_catalog,
        "segmentation_labels": segmentation_map,
        "class_label_coverage": float(len(known_class_names) / max(len(object_catalog), 1)) if object_catalog else 0.0,
        "observed_object_ref_count": int(len(observed_object_refs)),
    }
    return class_labels, object_refs, semantic_context


def _extract_scene_spec_from_metadata(metadata: Dict[str, Any]) -> Optional[Any]:
    if not isinstance(metadata, dict):
        return None
    candidates = [
        metadata.get("scene_spec"),
        metadata.get("metadata", {}).get("env_params", {}).get("scene_spec")
        if isinstance(metadata.get("metadata"), dict)
        else None,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            from src.envs.workcell_env.scene.scene_spec import WorkcellSceneSpec

            if isinstance(candidate, WorkcellSceneSpec):
                return candidate
            if isinstance(candidate, dict):
                return WorkcellSceneSpec.from_dict(candidate)
        except Exception:
            return None
    return None


def _build_scene_object_catalog(scene_spec: Optional[Any]) -> List[Dict[str, Any]]:
    if scene_spec is None:
        return []

    catalog: List[Dict[str, Any]] = []

    def _append(
        *,
        object_id: str,
        class_name: str,
        category: str,
        semantic_tags: List[str],
        affordances: Optional[List[str]] = None,
    ) -> None:
        catalog.append(
            {
                "object_id": str(object_id),
                "class_name": _normalize_semantic_label(class_name),
                "category": str(category),
                "semantic_tags": sorted({str(tag) for tag in semantic_tags if str(tag).strip()}),
                "affordances": sorted({str(tag) for tag in (affordances or []) if str(tag).strip()}),
            }
        )

    for station in getattr(scene_spec, "stations", []) or []:
        _append(
            object_id=station.id,
            class_name=getattr(station, "station_type", station.id),
            category="station",
            semantic_tags=["station", "support_surface", "category:station", f"object:{getattr(station, 'station_type', station.id)}"],
            affordances=[str(cap).lower() for cap in getattr(station, "capabilities", ())],
        )
    for fixture in getattr(scene_spec, "fixtures", []) or []:
        _append(
            object_id=fixture.id,
            class_name=getattr(fixture, "fixture_type", fixture.id),
            category="fixture",
            semantic_tags=["fixture", "category:fixture", f"object:{getattr(fixture, 'fixture_type', fixture.id)}"],
            affordances=["stabilize", "clamp"],
        )
    for container in getattr(scene_spec, "containers", []) or []:
        _append(
            object_id=container.id,
            class_name=getattr(container, "container_type", container.id),
            category="container",
            semantic_tags=["container", "category:container", f"object:{getattr(container, 'container_type', container.id)}"],
            affordances=["contain", "receive"],
        )
    for conveyor in getattr(scene_spec, "conveyors", []) or []:
        _append(
            object_id=conveyor.id,
            class_name="conveyor_segment",
            category="conveyor",
            semantic_tags=["conveyor", "support_surface", "dynamic_surface", "object:conveyor_segment"],
            affordances=["transport"],
        )
    for part in getattr(scene_spec, "parts", []) or []:
        semantic_tags = [
            "part",
            "category:part",
            f"object:{getattr(part, 'part_type', part.id)}",
        ]
        material = str(getattr(part, "material", "") or "")
        if material:
            semantic_tags.append(f"material:{_normalize_semantic_label(material)}")
        _append(
            object_id=part.id,
            class_name=getattr(part, "part_type", part.id),
            category="part",
            semantic_tags=semantic_tags,
            affordances=["pick", "place"],
        )
    for tool in getattr(scene_spec, "tools", []) or []:
        _append(
            object_id=tool.id,
            class_name=getattr(tool, "tool_type", tool.id),
            category="tool",
            semantic_tags=["tool", "category:tool", f"object:{getattr(tool, 'tool_type', tool.id)}"],
            affordances=["grasp", "manipulate"],
        )
    if not getattr(scene_spec, "tools", []):
        _append(
            object_id="end_effector",
            class_name="end_effector",
            category="tool",
            semantic_tags=["tool", "manipulator", "object:end_effector"],
            affordances=["grasp", "manipulate"],
        )
    return catalog


def _extract_segmentation_label_map(metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return {}
    candidates = [
        metadata.get("segmentation_label_map"),
        metadata.get("segmentation_labels"),
        metadata.get("seg_label_map"),
        metadata.get("sensor_bundle", {}).get("segmentation_label_map")
        if isinstance(metadata.get("sensor_bundle"), dict)
        else None,
    ]
    mapping: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key, value in candidate.items():
            normalized_key = str(key)
            if isinstance(value, dict):
                mapping[normalized_key] = {
                    "object_id": str(value.get("object_id") or value.get("id") or ""),
                    "class_name": _normalize_semantic_label(value.get("class_name") or value.get("label") or value.get("type") or ""),
                    "category": str(value.get("category") or ""),
                }
            else:
                mapping[normalized_key] = {
                    "object_id": "",
                    "class_name": _normalize_semantic_label(value),
                    "category": "",
                }
    return {key: value for key, value in mapping.items() if value.get("class_name")}


def _normalize_semantic_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return "_".join(part for part in text.replace("-", " ").split() if part)


def compute_datapack_frame_hash(metadata: Dict[str, Any]) -> str:
    """Compute a deterministic hash for datapack frame metadata."""
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

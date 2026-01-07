"""
MuJoCo renderer (with deterministic fallback) for workcell datapack frames.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import os

from src.envs.workcell_env.scene.fixtures import DEFAULT_FIXTURE_DIMENSIONS_MM
from src.envs.workcell_env.scene.scene_spec import WorkcellSceneSpec
from src.vision.nag.types import CameraParams


def render_workcell_frames(
    *,
    scene_spec: WorkcellSceneSpec,
    states: Optional[List[Dict[str, Any]]] = None,
    camera_name: str = "front",
    width: int = 128,
    height: int = 128,
    max_frames: Optional[int] = None,
    seed: Optional[int] = None,
    sensor_noise: Optional[Dict[str, Any]] = None,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]], CameraParams]:
    """Render RGB(+depth/seg) frames for a workcell scene."""
    try:
        frames, depth_frames, seg_frames, camera_params = _render_with_mujoco(
            scene_spec=scene_spec,
            states=states,
            camera_name=camera_name,
            width=width,
            height=height,
            max_frames=max_frames,
        )
    except Exception:
        frames, depth_frames, seg_frames, camera_params = _render_simple(
            scene_spec=scene_spec,
            states=states,
            camera_name=camera_name,
            width=width,
            height=height,
            max_frames=max_frames,
            seed=seed,
        )
    if os.getenv("WORKCELL_SEG_DEBUG") == "1":
        _debug_validate_segmentation(seg_frames or [], states)
    if sensor_noise:
        try:
            from src.envs.workcell_env.observations.sensor_noise import apply_sensor_noise

            frames, depth_frames, seg_frames, camera_params = apply_sensor_noise(
                rgb_frames=frames,
                depth_frames=depth_frames,
                seg_frames=seg_frames,
                camera_params=camera_params,
                seed=seed,
                config=sensor_noise,
            )
        except Exception:
            pass
    return frames, depth_frames, seg_frames, camera_params


def _render_with_mujoco(
    *,
    scene_spec: WorkcellSceneSpec,
    states: Optional[List[Dict[str, Any]]],
    camera_name: str,
    width: int,
    height: int,
    max_frames: Optional[int],
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]], CameraParams]:
    from src.envs.workcell_env.physics.mujoco_adapter import MujocoPhysicsAdapter

    adapter = MujocoPhysicsAdapter(spatial_bounds=scene_spec.spatial_bounds)
    adapter.reset(scene_spec, seed=0)

    frames: List[np.ndarray] = []
    depth_frames: List[np.ndarray] = []
    seg_frames: List[np.ndarray] = []

    if states:
        count = min(len(states), max_frames or len(states))
        for idx in range(count):
            _apply_state(adapter, states[idx])
            rgb = adapter.render(camera_name=camera_name, width=width, height=height)
            frames.append(_ensure_uint8(rgb))
            try:
                depth = adapter.render(camera_name=camera_name, width=width, height=height, depth=True)
                depth_frames.append(np.asarray(depth))
            except Exception:
                depth_frames = []
            try:
                seg = adapter.render(camera_name=camera_name, width=width, height=height, segmentation=True)
                seg_frames.append(_ensure_segmentation(seg))
            except Exception:
                seg_frames = []
    else:
        rgb = adapter.render(camera_name=camera_name, width=width, height=height)
        frames.append(_ensure_uint8(rgb))
        try:
            depth = adapter.render(camera_name=camera_name, width=width, height=height, depth=True)
            depth_frames.append(np.asarray(depth))
        except Exception:
            depth_frames = []
        try:
            seg = adapter.render(camera_name=camera_name, width=width, height=height, segmentation=True)
            seg_frames.append(_ensure_segmentation(seg))
        except Exception:
            seg_frames = []

    if not seg_frames:
        _, _, fallback_seg, _ = _render_simple(
            scene_spec=scene_spec,
            states=states,
            camera_name=camera_name,
            width=width,
            height=height,
            max_frames=len(frames),
            seed=None,
        )
        seg_frames = fallback_seg or []
    if not depth_frames:
        _, fallback_depth, _, _ = _render_simple(
            scene_spec=scene_spec,
            states=states,
            camera_name=camera_name,
            width=width,
            height=height,
            max_frames=len(frames),
            seed=None,
        )
        depth_frames = fallback_depth or []
    camera_params = _build_camera_params_from_mujoco(adapter, camera_name, width, height, len(frames))
    if camera_params is None:
        camera_params = _build_camera_params(camera_name, width, height, len(frames))
    return frames, depth_frames or None, seg_frames or None, camera_params


def _render_simple(
    *,
    scene_spec: WorkcellSceneSpec,
    states: Optional[List[Dict[str, Any]]],
    camera_name: str,
    width: int,
    height: int,
    max_frames: Optional[int],
    seed: Optional[int],
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]], CameraParams]:
    rng = np.random.RandomState(seed or 0)
    frames: List[np.ndarray] = []
    depth_frames: List[np.ndarray] = []
    seg_frames: List[np.ndarray] = []

    if states:
        count = min(len(states), max_frames or len(states))
    else:
        count = max_frames or 1

    objects = _scene_objects(scene_spec)
    bounds = scene_spec.spatial_bounds
    for t in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32)
        seg = np.zeros((height, width), dtype=np.int32)

        state_objects = None
        if states and t < len(states):
            state_objects = states[t].get("objects", {})

        for idx, obj in enumerate(objects, start=1):
            obj_id = obj["id"]
            pos = obj["position"]
            size = obj["size"]
            if state_objects and obj_id in state_objects:
                pos = state_objects[obj_id].get("position", pos)

            color = _color_for_id(obj_id, rng)
            _draw_object(frame, seg, depth, pos, size, bounds, color, idx)

        frames.append(frame)
        depth_frames.append(depth)
        seg_frames.append(seg)

    camera_params = _build_camera_params(camera_name, width, height, count)
    return frames, depth_frames, seg_frames, camera_params


def _scene_objects(scene_spec: WorkcellSceneSpec) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
    for station in scene_spec.stations:
        objects.append({"id": station.id, "position": station.position, "size": (0.3, 0.3, 0.05)})
    for fixture in scene_spec.fixtures:
        dims = DEFAULT_FIXTURE_DIMENSIONS_MM.get(fixture.fixture_type, (300.0, 200.0, 120.0))
        size = tuple(d / 1000.0 for d in dims)
        objects.append({"id": fixture.id, "position": fixture.position, "size": size})
    for container in scene_spec.containers:
        size = tuple(s / 1000.0 for s in container.slot_size_mm)
        objects.append({"id": container.id, "position": container.position, "size": size})
    for conveyor in scene_spec.conveyors:
        objects.append({"id": conveyor.id, "position": conveyor.position, "size": (conveyor.length_m, conveyor.width_m, 0.1)})
    for part in scene_spec.parts:
        size = tuple(s / 1000.0 for s in part.dimensions_mm)
        objects.append({"id": part.id, "position": part.position, "size": size})
    for tool in scene_spec.tools:
        objects.append({"id": tool.id, "position": tool.position, "size": (0.1, 0.1, 0.1)})
    if not scene_spec.tools:
        objects.append({"id": "end_effector", "position": (0.0, 0.0, 0.3), "size": (0.1, 0.1, 0.1)})
    return objects


def build_segmentation_label_map(scene_spec: WorkcellSceneSpec) -> Dict[str, int]:
    """Return a stable object_id -> seg_id mapping for the simple renderer."""
    return {obj["id"]: idx for idx, obj in enumerate(_scene_objects(scene_spec), start=1)}


def _color_for_id(obj_id: str, rng: np.random.RandomState) -> Tuple[int, int, int]:
    import hashlib

    token = int(hashlib.sha256(obj_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(token)
    color = rng.randint(50, 220, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def _draw_object(
    frame: np.ndarray,
    seg: np.ndarray,
    depth: np.ndarray,
    position: Tuple[float, float, float],
    size: Tuple[float, float, float],
    bounds: Tuple[float, float, float],
    color: Tuple[int, int, int],
    label: int,
) -> None:
    x, y, z = position
    bound_x, bound_y, _ = bounds
    u = int((x + bound_x / 2.0) / max(bound_x, 1e-6) * (frame.shape[1] - 1))
    v = int((y + bound_y / 2.0) / max(bound_y, 1e-6) * (frame.shape[0] - 1))
    radius_px = max(2, int(max(size[0], size[1]) / max(bound_x, bound_y) * frame.shape[0]))

    yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
    mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius_px ** 2
    frame[mask] = color
    seg[mask] = label
    depth[mask] = float(z)


def _apply_state(adapter: Any, state: Dict[str, Any]) -> None:
    try:
        import mujoco  # type: ignore
    except Exception:
        mujoco = None  # type: ignore
    objects = state.get("objects", {})
    for obj_id, obj in objects.items():
        if hasattr(adapter, "_free_joints"):
            qpos_addr = adapter._free_joints.get(obj_id)  # type: ignore[attr-defined]
            if qpos_addr is None:
                continue
            pos = np.array(obj.get("position", (0.0, 0.0, 0.0)), dtype=np.float64)
            quat = np.array(obj.get("orientation", (1.0, 0.0, 0.0, 0.0)), dtype=np.float64)
            adapter._data.qpos[qpos_addr : qpos_addr + 3] = pos  # type: ignore[attr-defined]
            adapter._data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat  # type: ignore[attr-defined]
    if mujoco is not None and getattr(adapter, "_model", None) is not None:
        mujoco.mj_forward(adapter._model, adapter._data)  # type: ignore[arg-type]


def _build_camera_params(camera_name: str, width: int, height: int, num_frames: int) -> CameraParams:
    pose = {
        "front": {"position": (0.0, -1.5, 1.0), "look_at": (0.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)},
        "top": {"position": (0.0, 0.0, 2.0), "look_at": (0.0, 0.0, 0.0), "up": (0.0, 1.0, 0.0)},
        "wrist": {"position": (0.3, -0.3, 0.6), "look_at": (0.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)},
    }.get(camera_name, {"position": (0.0, -1.5, 1.0), "look_at": (0.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)})
    fov_deg = 60.0
    camera = CameraParams.from_single_pose(
        position=pose["position"],
        look_at=pose["look_at"],
        up=pose["up"],
        fov_deg=fov_deg,
        width=width,
        height=height,
        camera_id=camera_name,
    )
    if num_frames > 1 and camera.world_from_cam.shape[0] == 1:
        camera.world_from_cam = np.repeat(camera.world_from_cam, num_frames, axis=0)
    return camera


def _build_camera_params_from_mujoco(
    adapter: Any,
    camera_name: str,
    width: int,
    height: int,
    num_frames: int,
) -> Optional[CameraParams]:
    try:
        import mujoco  # type: ignore
    except Exception:
        return None
    model = getattr(adapter, "_model", None)
    data = getattr(adapter, "_data", None)
    if model is None or data is None:
        return None
    try:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    except Exception:
        return None
    if cam_id is None or cam_id < 0:
        return None
    try:
        mujoco.mj_forward(model, data)
    except Exception:
        pass
    pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float32)
    mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float32).reshape(3, 3)
    world_from_cam = np.eye(4, dtype=np.float32)
    world_from_cam[:3, :3] = mat
    world_from_cam[:3, 3] = pos
    world_from_cam = world_from_cam[np.newaxis, ...]
    if num_frames > 1:
        world_from_cam = np.repeat(world_from_cam, num_frames, axis=0)

    fovy = 60.0
    if hasattr(model, "cam_fovy"):
        try:
            fovy = float(model.cam_fovy[cam_id])
        except Exception:
            fovy = 60.0
    fy = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return CameraParams(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        world_from_cam=world_from_cam,
        camera_id=camera_name,
    )


def _ensure_uint8(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr


def _ensure_segmentation(seg: Any) -> np.ndarray:
    arr = np.asarray(seg)
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            arr = arr[..., 0]
        elif arr.shape[-1] in (1, 3, 4):
            raise ValueError("Unexpected segmentation shape; expected (H, W, 2).")
    return arr.astype(np.int32)


def _debug_state_signature(state: Dict[str, Any]) -> Tuple[Tuple[str, Tuple[float, float, float]], ...]:
    objects = state.get("objects", {}) if state else {}
    items: List[Tuple[str, Tuple[float, float, float]]] = []
    for key in sorted(objects):
        pos = objects[key].get("position")
        if pos is None:
            continue
        items.append((key, tuple(round(float(p), 4) for p in pos)))
    return tuple(items)


def _debug_validate_segmentation(seg_frames: List[np.ndarray], states: Optional[List[Dict[str, Any]]]) -> None:
    if not seg_frames:
        raise AssertionError("Segmentation frames missing in debug mode.")
    unique_ids = np.unique(seg_frames[0])
    if unique_ids.size < 2:
        raise AssertionError("Segmentation IDs are not populated in debug mode.")
    if not np.any(seg_frames[0] == 0):
        raise AssertionError("Segmentation background ID missing in debug mode.")
    for frame in seg_frames[1:]:
        if not np.any(frame == 0):
            raise AssertionError("Segmentation background ID missing in debug mode.")
    if states and len(states) > 1:
        prev_sig: Optional[Tuple[Tuple[str, Tuple[float, float, float]], ...]] = None
        prev_frame: Optional[np.ndarray] = None
        for state, frame in zip(states, seg_frames):
            sig = _debug_state_signature(state)
            if prev_sig is not None and sig == prev_sig:
                if prev_frame is not None and not np.array_equal(frame, prev_frame):
                    raise AssertionError("Segmentation IDs changed for a static state in debug mode.")
            prev_sig = sig
            prev_frame = frame

"""
MuJoCo physics adapter for workcell environments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.envs.workcell_env.scene.fixtures import DEFAULT_FIXTURE_DIMENSIONS_MM
from src.envs.workcell_env.scene.scene_spec import WorkcellSceneSpec


@dataclass
class ContactMetrics:
    contact_count: int
    contact_force_N: float
    contact_impulse_Ns: float
    constraint_error: float


class MujocoPhysicsAdapter:
    """Physics adapter using MuJoCo for rigid-body simulation."""

    def __init__(
        self,
        *,
        spatial_bounds: Tuple[float, float, float] = (5.0, 5.0, 3.0),
    ) -> None:
        self.spatial_bounds = spatial_bounds
        self._model = None
        self._data = None
        self._body_name_to_id: Dict[str, int] = {}
        self._free_joints: Dict[str, int] = {}
        self._object_types: Dict[str, str] = {}
        self._contact_metrics = ContactMetrics(0, 0.0, 0.0, 0.0)
        self._renderer = None

    def reset(self, scene_spec: WorkcellSceneSpec, seed: Optional[int] = None) -> None:
        import mujoco  # type: ignore

        xml = _build_mjcf(scene_spec)
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        self._body_name_to_id = _build_body_map(self._model, scene_spec)
        self._free_joints = _build_free_joint_map(self._model, self._body_name_to_id)
        self._object_types = _build_object_type_map(scene_spec)
        self._renderer = None

        _set_initial_state(self._model, self._data, scene_spec, self._free_joints)
        mujoco.mj_forward(self._model, self._data)
        self._contact_metrics = _compute_contact_metrics(self._model, self._data)

    def step(self, time_step_s: float) -> None:
        import mujoco  # type: ignore

        if self._model is None or self._data is None:
            return
        self._model.opt.timestep = float(time_step_s)
        mujoco.mj_step(self._model, self._data)
        self._contact_metrics = _compute_contact_metrics(self._model, self._data)

    def apply_action(self, action: Any) -> None:
        import mujoco  # type: ignore

        if self._model is None or self._data is None:
            return
        if isinstance(action, dict):
            object_id = action.get("object_id", "end_effector")
            if object_id not in self._free_joints:
                return
            qpos_addr = self._free_joints[object_id]
            pos = self._data.qpos[qpos_addr : qpos_addr + 3].copy()
            quat = self._data.qpos[qpos_addr + 3 : qpos_addr + 7].copy()

            if "delta_position" in action:
                dx, dy, dz = action["delta_position"]
                pos = pos + np.array([dx, dy, dz], dtype=np.float64)
            elif "target_position" in action:
                pos = np.array(action["target_position"], dtype=np.float64)
            elif "position" in action:
                pos = np.array(action["position"], dtype=np.float64)

            self._data.qpos[qpos_addr : qpos_addr + 3] = pos
            self._data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat
            mujoco.mj_forward(self._model, self._data)
        elif isinstance(action, (list, tuple)) and len(action) == 3:
            object_id = "end_effector"
            if object_id not in self._free_joints:
                return
            qpos_addr = self._free_joints[object_id]
            pos = self._data.qpos[qpos_addr : qpos_addr + 3].copy()
            pos = pos + np.array(action, dtype=np.float64)
            self._data.qpos[qpos_addr : qpos_addr + 3] = pos
            mujoco.mj_forward(self._model, self._data)

    def get_state(self) -> Dict[str, Any]:
        if self._model is None or self._data is None:
            return {"objects": {}, "time_s": 0.0}
        objects: Dict[str, Any] = {}
        for name, body_id in self._body_name_to_id.items():
            pos = self._data.xpos[body_id].copy()
            quat = self._data.xquat[body_id].copy()
            vel = self._data.cvel[body_id][:3].copy()
            objects[name] = {
                "type": self._object_types.get(name, "object"),
                "position": tuple(float(x) for x in pos),
                "orientation": tuple(float(x) for x in quat),
                "velocity": tuple(float(x) for x in vel),
            }

        return {
            "objects": objects,
            "time_s": float(self._data.time),
            "collision_count": int(self._contact_metrics.contact_count),
            "contact_force_N": float(self._contact_metrics.contact_force_N),
            "contact_impulse_Ns": float(self._contact_metrics.contact_impulse_Ns),
            "constraint_error": float(self._contact_metrics.constraint_error),
        }

    def check_collision(self, object_id_a: str, object_id_b: str) -> bool:
        import mujoco  # type: ignore

        if self._model is None or self._data is None:
            return False
        body_a = self._body_name_to_id.get(object_id_a)
        body_b = self._body_name_to_id.get(object_id_b)
        if body_a is None or body_b is None:
            return False
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            body1 = self._model.geom_bodyid[geom1]
            body2 = self._model.geom_bodyid[geom2]
            if (body1 == body_a and body2 == body_b) or (body1 == body_b and body2 == body_a):
                return True
        return False

    def render(
        self,
        *,
        camera_name: str = "front",
        width: int = 128,
        height: int = 128,
        depth: bool = False,
        segmentation: bool = False,
    ) -> Any:
        import mujoco  # type: ignore

        if self._model is None or self._data is None:
            return None
        if (
            self._renderer is None
            or getattr(self._renderer, "width", None) != width
            or getattr(self._renderer, "height", None) != height
        ):
            self._renderer = mujoco.Renderer(self._model, height=height, width=width)
        try:
            self._renderer.update_scene(self._data, camera=camera_name)
        except Exception:
            self._renderer.update_scene(self._data)

        if hasattr(self._renderer, "enable_segmentation_rendering"):
            if segmentation:
                self._renderer.enable_segmentation_rendering()
                if hasattr(self._renderer, "disable_depth_rendering"):
                    self._renderer.disable_depth_rendering()
            elif depth:
                self._renderer.enable_depth_rendering()
                if hasattr(self._renderer, "disable_segmentation_rendering"):
                    self._renderer.disable_segmentation_rendering()
            else:
                if hasattr(self._renderer, "disable_depth_rendering"):
                    self._renderer.disable_depth_rendering()
                if hasattr(self._renderer, "disable_segmentation_rendering"):
                    self._renderer.disable_segmentation_rendering()
            return self._renderer.render()
        try:
            return self._renderer.render(depth=depth, segmentation=segmentation)
        except TypeError:
            return self._renderer.render()


def _build_mjcf(scene_spec: WorkcellSceneSpec) -> str:
    geoms = []
    bodies = []
    for station in scene_spec.stations:
        size = (0.5, 0.5, 0.05)
        bodies.append(_static_body_xml(station.id, station.position, station.orientation, size))
    for fixture in scene_spec.fixtures:
        dims = DEFAULT_FIXTURE_DIMENSIONS_MM.get(fixture.fixture_type, (300.0, 200.0, 120.0))
        size = tuple(d / 2000.0 for d in dims)
        bodies.append(_static_body_xml(fixture.id, fixture.position, fixture.orientation, size))
    for container in scene_spec.containers:
        size = tuple(s / 2000.0 for s in container.slot_size_mm)
        bodies.append(_static_body_xml(container.id, container.position, container.orientation, size))
    for conveyor in scene_spec.conveyors:
        size = (conveyor.length_m / 2.0, conveyor.width_m / 2.0, 0.05)
        bodies.append(_static_body_xml(conveyor.id, conveyor.position, conveyor.orientation, size))
    for part in scene_spec.parts:
        size = tuple(s / 2000.0 for s in part.dimensions_mm)
        bodies.append(_dynamic_body_xml(part.id, part.position, part.orientation, size))
    for tool in scene_spec.tools:
        size = (0.05, 0.05, 0.05)
        bodies.append(_dynamic_body_xml(tool.id, tool.position, tool.orientation, size))

    if not scene_spec.tools:
        bodies.append(_dynamic_body_xml("end_effector", (0.0, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0), (0.04, 0.04, 0.04)))

    geoms.append('<geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>')
    cameras = _default_camera_xml()

    xml = f"""
<mujoco>
  <option gravity="0 0 -9.81" integrator="Euler"/>
  <worldbody>
    {''.join(geoms)}
    {''.join(bodies)}
    {cameras}
  </worldbody>
</mujoco>
"""
    return xml


def _static_body_xml(name: str, position: Tuple[float, float, float], orientation: Tuple[float, float, float, float], size: Tuple[float, float, float]) -> str:
    pos = " ".join(f"{v:.4f}" for v in position)
    quat = " ".join(f"{v:.4f}" for v in orientation)
    size_str = " ".join(f"{v:.4f}" for v in size)
    return (
        f'<body name="{name}" pos="{pos}" quat="{quat}">'
        f'<geom type="box" size="{size_str}" rgba="0.4 0.4 0.4 1" friction="1 0.01 0.001"/>'
        "</body>"
    )


def _dynamic_body_xml(name: str, position: Tuple[float, float, float], orientation: Tuple[float, float, float, float], size: Tuple[float, float, float]) -> str:
    pos = " ".join(f"{v:.4f}" for v in position)
    quat = " ".join(f"{v:.4f}" for v in orientation)
    size_str = " ".join(f"{v:.4f}" for v in size)
    return (
        f'<body name="{name}" pos="{pos}" quat="{quat}">'
        f'<joint name="{name}_joint" type="free"/>'
        f'<geom type="box" size="{size_str}" rgba="0.8 0.3 0.3 1" friction="1 0.01 0.001"/>'
        "</body>"
    )


def _default_camera_xml() -> str:
    cameras = [
        ("front", (0.0, -1.5, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ("top", (0.0, 0.0, 2.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("wrist", (0.3, -0.3, 0.6), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ]
    return "".join(_camera_xml(name, pos, look_at, up) for name, pos, look_at, up in cameras)


def _camera_xml(
    name: str,
    position: Tuple[float, float, float],
    look_at: Tuple[float, float, float],
    up: Tuple[float, float, float],
) -> str:
    quat = _quat_from_lookat(position, look_at, up)
    pos = " ".join(f"{v:.4f}" for v in position)
    return f'<camera name="{name}" pos="{pos}" quat="{quat}"/>'


def _quat_from_lookat(
    position: Tuple[float, float, float],
    look_at: Tuple[float, float, float],
    up: Tuple[float, float, float],
) -> str:
    pos = np.asarray(position, dtype=np.float32)
    target = np.asarray(look_at, dtype=np.float32)
    up_vec = np.asarray(up, dtype=np.float32)
    forward = target - pos
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        forward = forward / norm
    right = np.cross(forward, up_vec)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / right_norm
    up_corrected = np.cross(right, forward)
    rot = np.stack([right, up_corrected, -forward], axis=1)
    quat = _quat_from_matrix(rot)
    return " ".join(f"{v:.6f}" for v in quat)


def _quat_from_matrix(rot: np.ndarray) -> Tuple[float, float, float, float]:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rot[2, 1] - rot[1, 2]) * s
        qy = (rot[0, 2] - rot[2, 0]) * s
        qz = (rot[1, 0] - rot[0, 1]) * s
    else:
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


def _build_body_map(model: Any, scene_spec: WorkcellSceneSpec) -> Dict[str, int]:
    import mujoco  # type: ignore

    names = []
    for group in (scene_spec.stations, scene_spec.fixtures, scene_spec.containers, scene_spec.conveyors, scene_spec.parts, scene_spec.tools):
        for item in group:
            names.append(item.id)
    if not scene_spec.tools:
        names.append("end_effector")
    mapping: Dict[str, int] = {}
    for name in names:
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id != -1:
                mapping[name] = body_id
        except Exception:
            continue
    return mapping


def _build_free_joint_map(model: Any, body_map: Dict[str, int]) -> Dict[str, int]:
    free_joints: Dict[str, int] = {}
    for name, body_id in body_map.items():
        jntnum = int(model.body_jntnum[body_id])
        if jntnum == 0:
            continue
        jntadr = int(model.body_jntadr[body_id])
        qpos_addr = int(model.jnt_qposadr[jntadr])
        free_joints[name] = qpos_addr
    return free_joints


def _build_object_type_map(scene_spec: WorkcellSceneSpec) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for station in scene_spec.stations:
        mapping[station.id] = "station"
    for fixture in scene_spec.fixtures:
        mapping[fixture.id] = "fixture"
    for container in scene_spec.containers:
        mapping[container.id] = "container"
    for conveyor in scene_spec.conveyors:
        mapping[conveyor.id] = "conveyor"
    for part in scene_spec.parts:
        mapping[part.id] = "part"
    for tool in scene_spec.tools:
        mapping[tool.id] = "tool"
    if not scene_spec.tools:
        mapping["end_effector"] = "tool"
    return mapping


def _set_initial_state(model: Any, data: Any, scene_spec: WorkcellSceneSpec, free_joints: Dict[str, int]) -> None:
    for part in scene_spec.parts:
        _set_body_pose(data, free_joints, part.id, part.position, part.orientation)
    for tool in scene_spec.tools:
        _set_body_pose(data, free_joints, tool.id, tool.position, tool.orientation)
    if not scene_spec.tools:
        _set_body_pose(data, free_joints, "end_effector", (0.0, 0.0, 0.4), (1.0, 0.0, 0.0, 0.0))


def _set_body_pose(data: Any, free_joints: Dict[str, int], name: str, position: Tuple[float, float, float], orientation: Tuple[float, float, float, float]) -> None:
    qpos_addr = free_joints.get(name)
    if qpos_addr is None:
        return
    data.qpos[qpos_addr : qpos_addr + 3] = np.array(position, dtype=np.float64)
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.array(orientation, dtype=np.float64)


def _compute_contact_metrics(model: Any, data: Any) -> ContactMetrics:
    import mujoco  # type: ignore

    contact_count = int(data.ncon)
    total_force = 0.0
    total_impulse = 0.0
    for i in range(data.ncon):
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, force)
        total_force += float(np.linalg.norm(force[:3]))
        total_impulse += float(np.linalg.norm(force[:3]) * model.opt.timestep)
    constraint_error = 0.0
    if data.efc_pos.size:
        constraint_error = float(np.max(np.abs(data.efc_pos)))
    return ContactMetrics(
        contact_count=contact_count,
        contact_force_N=total_force,
        contact_impulse_Ns=total_impulse,
        constraint_error=constraint_error,
    )

"""
Simple physics adapter for workcell smoke tests.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, Optional, Tuple

from src.envs.workcell_env.scene.fixtures import DEFAULT_FIXTURE_DIMENSIONS_MM
from src.envs.workcell_env.scene.scene_spec import (
    ContainerSpec,
    ConveyorSpec,
    FixtureSpec,
    PartSpec,
    StationSpec,
    ToolSpec,
    WorkcellSceneSpec,
)


class SimplePhysicsAdapter:
    """
    Minimal kinematic physics adapter for fast iteration.

    This adapter maintains a dictionary of object poses and supports
    basic position updates without dynamic simulation.
    """

    def __init__(
        self,
        *,
        spatial_bounds: Tuple[float, float, float] = (5.0, 5.0, 3.0),
        kinematic_only: bool = True,
    ) -> None:
        self.spatial_bounds = spatial_bounds
        self.kinematic_only = bool(kinematic_only)
        self._rng = random.Random()
        self._state: Dict[str, Any] = {"objects": {}, "time_s": 0.0}

    def reset(self, scene_spec: WorkcellSceneSpec, seed: Optional[int] = None) -> None:
        """Reset state from a scene specification."""
        if seed is not None:
            self._rng = random.Random(seed)
        self.spatial_bounds = scene_spec.spatial_bounds
        self._state = {"objects": {}, "time_s": 0.0}

        for station in scene_spec.stations:
            self._register_station(station)
        for fixture in scene_spec.fixtures:
            self._register_fixture(fixture)
        for container in scene_spec.containers:
            self._register_container(container)
        for conveyor in scene_spec.conveyors:
            self._register_conveyor(conveyor)
        for tool in scene_spec.tools:
            self._register_tool(tool)
        for part in scene_spec.parts:
            self._register_part(part)

    def step(self, time_step_s: float) -> None:
        """Advance simulation state by a timestep."""
        self._state["time_s"] = float(self._state.get("time_s", 0.0)) + float(time_step_s)
        if not self.kinematic_only:
            for obj in self._state["objects"].values():
                velocity = obj.get("velocity")
                if not velocity:
                    continue
                px, py, pz = obj["position"]
                vx, vy, vz = velocity
                obj["position"] = self._clamp_position(
                    (px + vx * time_step_s, py + vy * time_step_s, pz + vz * time_step_s)
                )

    def apply_action(self, action: Any) -> None:
        """Apply a kinematic action to object state."""
        if isinstance(action, dict):
            object_id = action.get("object_id")
            if object_id is None:
                return
            obj = self._state["objects"].setdefault(object_id, {"position": (0.0, 0.0, 0.0)})
            if "delta_position" in action:
                dx, dy, dz = action["delta_position"]
                px, py, pz = obj.get("position", (0.0, 0.0, 0.0))
                obj["position"] = self._clamp_position((px + dx, py + dy, pz + dz))
            elif "target_position" in action:
                obj["position"] = self._clamp_position(tuple(action["target_position"]))
            elif "position" in action:
                obj["position"] = self._clamp_position(tuple(action["position"]))
            if "velocity" in action:
                obj["velocity"] = tuple(action["velocity"])
            return

        if isinstance(action, (list, tuple)) and len(action) == 3:
            obj = self._state["objects"].setdefault(
                "end_effector", {"position": (0.0, 0.0, 0.0), "radius_m": 0.05, "type": "ee"}
            )
            px, py, pz = obj.get("position", (0.0, 0.0, 0.0))
            obj["position"] = self._clamp_position((px + action[0], py + action[1], pz + action[2]))

    def get_state(self) -> Dict[str, Any]:
        """Return a copy of the physics state."""
        return copy.deepcopy(self._state)

    def check_collision(self, object_id_a: str, object_id_b: str) -> bool:
        """Check collision using bounding spheres."""
        obj_a = self._state["objects"].get(object_id_a)
        obj_b = self._state["objects"].get(object_id_b)
        if not obj_a or not obj_b:
            return False
        radius_a = float(obj_a.get("radius_m", 0.0))
        radius_b = float(obj_b.get("radius_m", 0.0))
        ax, ay, az = obj_a.get("position", (0.0, 0.0, 0.0))
        bx, by, bz = obj_b.get("position", (0.0, 0.0, 0.0))
        dist_sq = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
        return dist_sq <= (radius_a + radius_b) ** 2

    def _register_station(self, station: StationSpec) -> None:
        radius_m = 0.5
        self._state["objects"][station.id] = {
            "type": "station",
            "position": station.position,
            "orientation": station.orientation,
            "radius_m": radius_m,
        }

    def _register_fixture(self, fixture: FixtureSpec) -> None:
        dims = DEFAULT_FIXTURE_DIMENSIONS_MM.get(fixture.fixture_type, (300.0, 200.0, 120.0))
        radius_m = max(dims[0], dims[1]) / 2000.0
        self._state["objects"][fixture.id] = {
            "type": "fixture",
            "position": fixture.position,
            "orientation": fixture.orientation,
            "radius_m": radius_m,
        }

    def _register_container(self, container: ContainerSpec) -> None:
        radius_m = max(container.slot_size_mm[0], container.slot_size_mm[1]) / 2000.0
        self._state["objects"][container.id] = {
            "type": "container",
            "position": container.position,
            "orientation": container.orientation,
            "radius_m": radius_m,
        }

    def _register_conveyor(self, conveyor: ConveyorSpec) -> None:
        radius_m = ((conveyor.length_m / 2.0) ** 2 + (conveyor.width_m / 2.0) ** 2) ** 0.5
        self._state["objects"][conveyor.id] = {
            "type": "conveyor",
            "position": conveyor.position,
            "orientation": conveyor.orientation,
            "radius_m": radius_m,
            "speed_m_s": conveyor.speed_m_s,
        }

    def _register_tool(self, tool: ToolSpec) -> None:
        self._state["objects"][tool.id] = {
            "type": "tool",
            "position": tool.position,
            "orientation": tool.orientation,
            "radius_m": 0.15,
        }

    def _register_part(self, part: PartSpec) -> None:
        radius_m = max(part.dimensions_mm[0], part.dimensions_mm[1]) / 2000.0
        self._state["objects"][part.id] = {
            "type": "part",
            "position": part.position,
            "orientation": part.orientation,
            "radius_m": radius_m,
            "part_type": part.part_type,
        }

    def _clamp_position(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        x_bound, y_bound, z_bound = self.spatial_bounds
        x = max(min(position[0], x_bound / 2.0), -x_bound / 2.0)
        y = max(min(position[1], y_bound / 2.0), -y_bound / 2.0)
        z = max(min(position[2], z_bound), 0.0)
        return x, y, z


__all__ = ["SimplePhysicsAdapter"]

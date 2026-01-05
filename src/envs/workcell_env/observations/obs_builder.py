"""
Observation builder for workcell environments.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


class WorkcellObservationBuilder:
    """
    Build observation dictionaries from scene state.

    Vector-only mode emits object positions and orientations in lists.
    """

    def __init__(self, *, vector_only: bool = True, sort_ids: bool = True) -> None:
        self.vector_only = bool(vector_only)
        self.sort_ids = bool(sort_ids)

    def build(
        self, scene_state: Mapping[str, Any], *, vector_only: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Build an observation dict from scene state."""
        vector_only = self.vector_only if vector_only is None else bool(vector_only)
        objects = scene_state.get("objects", {})
        object_ids = list(objects.keys())
        if self.sort_ids:
            object_ids.sort()

        positions = []
        orientations = []
        types = []
        velocities = []
        state_vector = []

        for object_id in object_ids:
            obj = objects[object_id]
            pos = obj.get("position", (0.0, 0.0, 0.0))
            ori = obj.get("orientation", (1.0, 0.0, 0.0, 0.0))
            positions.append(list(pos))
            orientations.append(list(ori))
            types.append(obj.get("type", "unknown"))
            vel = obj.get("velocity", (0.0, 0.0, 0.0))
            velocities.append(list(vel))
            state_vector.extend([float(pos[0]), float(pos[1]), float(pos[2])])
            state_vector.extend([float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])])
            state_vector.extend([float(vel[0]), float(vel[1]), float(vel[2])])

        obs: Dict[str, Any] = {
            "object_ids": object_ids,
            "positions": positions,
            "orientations": orientations,
            "types": types,
            "velocities": velocities,
            "time_s": float(scene_state.get("time_s", 0.0)),
            "state_vector": state_vector,
        }

        if not vector_only:
            obs["objects"] = objects
        return obs


__all__ = ["WorkcellObservationBuilder"]

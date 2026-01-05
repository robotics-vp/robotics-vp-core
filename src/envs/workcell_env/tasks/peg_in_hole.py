"""
Canonical peg-in-hole assembly task implementation.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.envs.workcell_env.tasks.task_base import TaskSpec


class PegInHoleTask:
    """
    Precision insertion task for inserting a peg into a hole.

    The task tracks approach progress via peg-hole distance and awards a sparse
    bonus when insertion succeeds within tolerance and force limits.
    """

    def __init__(
        self,
        *,
        peg_id: str,
        hole_id: str,
        tolerance_mm: float = 2.0,
        max_force_N: float = 50.0,
        dense_reward_scale: float = 0.2,
        completion_bonus: float = 1.0,
    ) -> None:
        self.peg_id = str(peg_id)
        self.hole_id = str(hole_id)
        self.tolerance_mm = float(tolerance_mm)
        self.max_force_N = float(max_force_N)
        self.dense_reward_scale = float(dense_reward_scale)
        self.completion_bonus = float(completion_bonus)

        self._initial_distance_mm: Optional[float] = None
        self._last_progress: float = 0.0
        self._completed = False

    def reset(self) -> None:
        """Reset task progress tracking."""
        self._initial_distance_mm = None
        self._last_progress = 0.0
        self._completed = False

    def evaluate(self, task_state: Mapping[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Evaluate task progress and compute reward.

        Expected task_state keys:
            peg_to_hole_distance_mm: Optional float
            peg_position: Optional (x, y, z)
            hole_position: Optional (x, y, z)
            position_unit_scale: Optional float to convert positions to mm
            insertion_depth_mm: Optional float
            required_insertion_depth_mm: Optional float
            contact_force_N: Optional float
            is_inserted: Optional bool
        """
        distance_mm = self._compute_distance_mm(task_state)
        if distance_mm is None:
            info = {"missing_distance": True}
            return 0.0, False, info

        if self._initial_distance_mm is None:
            self._initial_distance_mm = distance_mm
            self._last_progress = 0.0

        initial = max(self._initial_distance_mm, 1e-6)
        progress = max(0.0, min(1.0, (initial - distance_mm) / initial))
        progress_delta = max(progress - self._last_progress, 0.0)
        self._last_progress = progress

        reward = self.dense_reward_scale * progress_delta

        force = float(task_state.get("contact_force_N", 0.0))
        force_violation = force > self.max_force_N

        depth = float(task_state.get("insertion_depth_mm", 0.0))
        required_depth = float(task_state.get("required_insertion_depth_mm", self.tolerance_mm))
        inserted_flag = bool(task_state.get("is_inserted", False))
        inserted = inserted_flag or (distance_mm <= self.tolerance_mm and depth >= required_depth)

        success = inserted and not force_violation
        if success and not self._completed:
            reward += self.completion_bonus
            self._completed = True

        info = {
            "distance_mm": float(distance_mm),
            "progress": float(progress),
            "force_violation": force_violation,
            "inserted": inserted,
            "completed": self._completed,
        }
        return reward, success, info

    def _compute_distance_mm(self, task_state: Mapping[str, Any]) -> Optional[float]:
        if "peg_to_hole_distance_mm" in task_state:
            return float(task_state["peg_to_hole_distance_mm"])

        peg_pos = task_state.get("peg_position")
        hole_pos = task_state.get("hole_position")
        if peg_pos is None or hole_pos is None:
            return None

        scale = float(task_state.get("position_unit_scale", 1.0))
        px, py, pz = float(peg_pos[0]), float(peg_pos[1]), float(peg_pos[2])
        hx, hy, hz = float(hole_pos[0]), float(hole_pos[1]), float(hole_pos[2])
        dist = ((px - hx) ** 2 + (py - hy) ** 2 + (pz - hz) ** 2) ** 0.5
        return dist * scale


def generate_peg_in_hole_task_spec(
    *,
    task_id: str = "peg_in_hole",
    peg_id: str = "peg",
    hole_id: str = "hole",
    tolerance_mm: float = 2.0,
    max_force_N: float = 50.0,
    time_limit: int = 120,
) -> TaskSpec:
    """
    Build a TaskSpec for a peg-in-hole task.
    """
    return TaskSpec(
        task_id=task_id,
        task_type="ASSEMBLY",
        success_metric="peg_inserted",
        reward_type="SHAPED",
        time_limit=int(time_limit),
        parameters={
            "peg_id": peg_id,
            "hole_id": hole_id,
            "tolerance_mm": float(tolerance_mm),
            "max_force_N": float(max_force_N),
        },
    )

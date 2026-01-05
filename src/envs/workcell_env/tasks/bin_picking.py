"""
Canonical bin picking task implementation.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from src.envs.workcell_env.tasks.task_base import TaskSpec


class BinPickingTask:
    """
    Bin picking task: pick a specified part from a cluttered bin.

    Success requires selecting the correct part type without collisions.
    """

    def __init__(
        self,
        *,
        target_part_type: str,
        bin_id: str,
        num_distractors: int = 4,
        reward_correct_pick: float = 0.2,
        collision_penalty: float = 0.1,
        success_bonus: float = 1.0,
    ) -> None:
        self.target_part_type = str(target_part_type)
        self.bin_id = str(bin_id)
        self.num_distractors = int(num_distractors)
        self.reward_correct_pick = float(reward_correct_pick)
        self.collision_penalty = float(collision_penalty)
        self.success_bonus = float(success_bonus)

        self._completed = False

    def reset(self) -> None:
        """Reset task progress tracking."""
        self._completed = False

    def evaluate(self, task_state: Mapping[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Evaluate task progress and compute reward.

        Expected task_state keys:
            picked_part_type: Optional str
            picked_part_id: Optional str
            part_types_by_id: Optional mapping for ID -> type
            collision_count: Optional int
            collision_free: Optional bool
        """
        picked_part_type = task_state.get("picked_part_type")
        if picked_part_type is None and "picked_part_id" in task_state:
            part_id = task_state.get("picked_part_id")
            type_lookup = task_state.get("part_types_by_id", {})
            picked_part_type = type_lookup.get(part_id)

        collision_count = int(task_state.get("collision_count", 0))
        if bool(task_state.get("collision_free", False)):
            collision_count = 0

        correct_pick = picked_part_type == self.target_part_type
        reward = 0.0
        if correct_pick:
            reward += self.reward_correct_pick
        reward -= self.collision_penalty * collision_count

        success = correct_pick and collision_count == 0
        if success and not self._completed:
            reward += self.success_bonus
            self._completed = True

        info = {
            "correct_pick": correct_pick,
            "collision_count": collision_count,
            "completed": self._completed,
        }
        return reward, success, info


def generate_bin_picking_task_spec(
    *,
    task_id: str = "bin_picking",
    target_part_type: str = "bolt",
    bin_id: str = "bin_0",
    num_distractors: int = 4,
    time_limit: int = 80,
) -> TaskSpec:
    """
    Build a TaskSpec for a bin picking task.
    """
    return TaskSpec(
        task_id=task_id,
        task_type="SORTING",
        success_metric="target_part_picked",
        reward_type="SHAPED",
        time_limit=int(time_limit),
        parameters={
            "target_part_type": target_part_type,
            "bin_id": bin_id,
            "num_distractors": int(num_distractors),
        },
    )

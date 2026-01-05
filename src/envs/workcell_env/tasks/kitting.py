"""
Canonical kitting task implementation for workcell environments.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.envs.workcell_env.tasks.task_base import TaskSpec


class KittingTask:
    """
    Kitting/packaging task: pick items from bins and place into a tray/box.

    The task tracks which target slots are correctly filled. If order_matters is
    True, the placement order must match the target slot order.
    """

    def __init__(
        self,
        *,
        num_items: int,
        target_positions: Optional[Sequence[Any]] = None,
        order_matters: bool = True,
        dense_reward: float = 0.1,
        completion_bonus: float = 1.0,
        position_tolerance: float = 0.02,
    ) -> None:
        if num_items <= 0:
            raise ValueError("num_items must be positive")
        if target_positions is None:
            target_positions = list(range(num_items))
        if len(target_positions) != num_items:
            raise ValueError("target_positions must match num_items")

        self.num_items = int(num_items)
        self.target_positions = [self._normalize_target(pos) for pos in target_positions]
        self.order_matters = bool(order_matters)
        self.dense_reward = float(dense_reward)
        self.completion_bonus = float(completion_bonus)
        self.position_tolerance = float(position_tolerance)

        self._targets_are_positions = all(
            isinstance(pos, tuple) and len(pos) == 3 for pos in self.target_positions
        )
        self._correct_slots: set[int] = set()
        self._completed = False

    def reset(self) -> None:
        """Reset task progress tracking."""
        self._correct_slots = set()
        self._completed = False

    def evaluate(self, task_state: Mapping[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Evaluate task progress and compute reward.

        Expected task_state keys:
            placements: Mapping[item_id -> target_index or position]
            placement_order: Optional list of item_ids in placement order
            position_tolerance: Optional float override
        """
        placements = task_state.get("placements", {})
        placement_order = list(task_state.get("placement_order", []))
        tolerance = float(task_state.get("position_tolerance", self.position_tolerance))

        if isinstance(placements, (list, tuple)):
            ordered_targets = list(placements)
        elif placement_order:
            ordered_targets = [placements[item_id] for item_id in placement_order if item_id in placements]
        elif isinstance(placements, Mapping):
            ordered_targets = [placements[key] for key in sorted(placements.keys())]
        else:
            ordered_targets = []

        correct_slots = set()
        if self.order_matters:
            for idx, placement in enumerate(ordered_targets):
                resolved = self._resolve_target_index(placement, tolerance)
                if resolved is not None and resolved == idx:
                    correct_slots.add(idx)
        else:
            seen = set()
            for placement in ordered_targets:
                resolved = self._resolve_target_index(placement, tolerance)
                if resolved is not None and resolved not in seen:
                    correct_slots.add(resolved)
                    seen.add(resolved)

        new_correct = correct_slots - self._correct_slots
        reward = len(new_correct) * self.dense_reward
        self._correct_slots = correct_slots

        success = len(correct_slots) >= self.num_items
        if success and not self._completed:
            reward += self.completion_bonus
            self._completed = True

        info = {
            "correct_count": len(correct_slots),
            "remaining": max(self.num_items - len(correct_slots), 0),
            "order_matters": self.order_matters,
            "completed": self._completed,
        }
        return reward, success, info

    def _normalize_target(self, target: Any) -> Any:
        if isinstance(target, (list, tuple)) and len(target) == 3:
            return (float(target[0]), float(target[1]), float(target[2]))
        return target

    def _resolve_target_index(self, placement: Any, tolerance: float) -> Optional[int]:
        if isinstance(placement, Mapping):
            if "target_index" in placement:
                placement = placement["target_index"]
            elif "position" in placement:
                placement = placement["position"]

        if isinstance(placement, int):
            return placement if 0 <= placement < self.num_items else None

        if (
            self._targets_are_positions
            and isinstance(placement, (list, tuple))
            and len(placement) == 3
        ):
            px, py, pz = float(placement[0]), float(placement[1]), float(placement[2])
            best_idx = None
            best_dist = None
            for idx, target_pos in enumerate(self.target_positions):
                tx, ty, tz = target_pos
                dist = ((px - tx) ** 2 + (py - ty) ** 2 + (pz - tz) ** 2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_dist is not None and best_dist <= tolerance:
                return best_idx

        return None


def generate_kitting_task_spec(
    *,
    task_id: str = "kitting",
    num_items: int = 4,
    target_positions: Optional[Sequence[Any]] = None,
    order_matters: bool = True,
    time_limit: Optional[int] = None,
) -> TaskSpec:
    """
    Build a TaskSpec for a kitting task.
    """
    if target_positions is None:
        target_positions = list(range(num_items))
    if time_limit is None:
        time_limit = max(num_items * 20, 60)
    return TaskSpec(
        task_id=task_id,
        task_type="PACKAGING",
        success_metric="all_items_placed",
        reward_type="SHAPED",
        time_limit=int(time_limit),
        parameters={
            "num_items": int(num_items),
            "target_positions": list(target_positions),
            "order_matters": bool(order_matters),
        },
    )

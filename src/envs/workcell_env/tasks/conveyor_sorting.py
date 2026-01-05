"""
Canonical conveyor sorting task implementation.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from src.envs.workcell_env.tasks.task_base import TaskSpec


class ConveyorSortingTask:
    """
    Conveyor sorting task: classify items and place in destination bins.
    """

    def __init__(
        self,
        *,
        conveyor_speed: float = 0.2,
        num_categories: int = 3,
        items_per_episode: int = 10,
        reward_per_item: float = 0.1,
        incorrect_penalty: float = 0.1,
        success_bonus: float = 1.0,
    ) -> None:
        self.conveyor_speed = float(conveyor_speed)
        self.num_categories = int(num_categories)
        self.items_per_episode = int(items_per_episode)
        self.reward_per_item = float(reward_per_item)
        self.incorrect_penalty = float(incorrect_penalty)
        self.success_bonus = float(success_bonus)

        self._seen_item_ids: set[str] = set()
        self._correct_count = 0
        self._incorrect_count = 0
        self._completed = False

    def reset(self) -> None:
        """Reset task progress tracking."""
        self._seen_item_ids = set()
        self._correct_count = 0
        self._incorrect_count = 0
        self._completed = False

    def evaluate(self, task_state: Mapping[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Evaluate task progress and compute reward.

        Expected task_state keys:
            sorted_items: Optional list of {"item_id": str, "correct": bool}
            sorted_correct_count: Optional int total
            sorted_incorrect_count: Optional int total
        """
        reward = 0.0
        sorted_items = task_state.get("sorted_items")

        if sorted_items:
            for item in sorted_items:
                item_id = item.get("item_id") if isinstance(item, Mapping) else None
                correct = bool(item.get("correct", False)) if isinstance(item, Mapping) else bool(item)
                if item_id is None:
                    item_id = f"item_{len(self._seen_item_ids)}"
                if item_id in self._seen_item_ids:
                    continue
                self._seen_item_ids.add(item_id)
                if correct:
                    self._correct_count += 1
                    reward += self.reward_per_item
                else:
                    self._incorrect_count += 1
                    reward -= self.incorrect_penalty
        else:
            total_correct = int(task_state.get("sorted_correct_count", self._correct_count))
            total_incorrect = int(task_state.get("sorted_incorrect_count", self._incorrect_count))
            delta_correct = max(total_correct - self._correct_count, 0)
            delta_incorrect = max(total_incorrect - self._incorrect_count, 0)
            reward += delta_correct * self.reward_per_item
            reward -= delta_incorrect * self.incorrect_penalty
            self._correct_count = max(self._correct_count, total_correct)
            self._incorrect_count = max(self._incorrect_count, total_incorrect)

        success = self._correct_count >= self.items_per_episode and self._incorrect_count == 0
        if success and not self._completed:
            reward += self.success_bonus
            self._completed = True

        info = {
            "correct_sorted": self._correct_count,
            "incorrect_sorted": self._incorrect_count,
            "remaining": max(self.items_per_episode - self._correct_count, 0),
            "completed": self._completed,
        }
        return reward, success, info


def generate_conveyor_sorting_task_spec(
    *,
    task_id: str = "conveyor_sorting",
    conveyor_speed: float = 0.2,
    num_categories: int = 3,
    items_per_episode: int = 10,
    time_limit: int = 150,
) -> TaskSpec:
    """
    Build a TaskSpec for a conveyor sorting task.
    """
    return TaskSpec(
        task_id=task_id,
        task_type="SORTING",
        success_metric="all_items_sorted",
        reward_type="DENSE",
        time_limit=int(time_limit),
        parameters={
            "conveyor_speed": float(conveyor_speed),
            "num_categories": int(num_categories),
            "items_per_episode": int(items_per_episode),
        },
    )

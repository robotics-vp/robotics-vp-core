"""
Physics adapter interface for workcell environments.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from src.envs.workcell_env.scene.scene_spec import WorkcellSceneSpec


class PhysicsAdapter(Protocol):
    """Protocol for physics backends used by workcell environments."""

    def reset(self, scene_spec: WorkcellSceneSpec, seed: Optional[int] = None) -> None:
        """Reset the physics state from a scene spec."""

    def step(self, time_step_s: float) -> None:
        """Advance simulation by one time step."""

    def apply_action(self, action: Any) -> None:
        """Apply an action to the simulation state."""

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of the simulation state."""

    def check_collision(self, object_id_a: str, object_id_b: str) -> bool:
        """Check collision between two objects by ID."""

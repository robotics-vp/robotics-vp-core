"""
Concrete workcell environment implementation.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from src.envs.workcell_env.base import WorkcellEnvBase
from src.envs.workcell_env.config import WorkcellEnvConfig
from src.envs.workcell_env.observations.obs_builder import WorkcellObservationBuilder
from src.envs.workcell_env.physics.physics_adapter import PhysicsAdapter
from src.envs.workcell_env.physics.simple_physics import SimplePhysicsAdapter
from src.envs.workcell_env.scene.generators import WorkcellSceneGenerator
from src.envs.workcell_env.scene.scene_spec import WorkcellSceneSpec


class WorkcellEnv(WorkcellEnvBase):
    """
    Workcell environment with procedural scene generation and simple physics.
    """

    def __init__(
        self,
        config: WorkcellEnvConfig,
        *,
        scene_spec: Optional[WorkcellSceneSpec] = None,
        scene_generator: Optional[WorkcellSceneGenerator] = None,
        physics_adapter: Optional[PhysicsAdapter] = None,
        observation_builder: Optional[WorkcellObservationBuilder] = None,
        task: Optional[Any] = None,
        task_id: str = "workcell_task",
        robot_family: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.scene_generator = scene_generator or WorkcellSceneGenerator()
        if scene_spec is None:
            scene_spec = self.scene_generator.generate(config, seed=seed)

        self.physics_adapter = physics_adapter or SimplePhysicsAdapter(
            spatial_bounds=scene_spec.spatial_bounds
        )
        self.observation_builder = observation_builder or WorkcellObservationBuilder(vector_only=True)
        self.task = task

        super().__init__(
            config=config,
            scene_spec=scene_spec,
            task_id=task_id,
            robot_family=robot_family,
            seed=seed,
        )

    def _reset_impl(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Reset the environment and return the initial observation."""
        scene_spec = options.get("scene_spec")
        if scene_spec is None:
            scene_spec = self.scene_generator.generate(self.config, seed=self.seed)
        elif isinstance(scene_spec, dict):
            scene_spec = WorkcellSceneSpec.from_dict(scene_spec)

        self.scene_spec = scene_spec
        self.physics_adapter.reset(scene_spec, seed=self.seed)

        if self.task and hasattr(self.task, "reset"):
            self.task.reset()

        state = self.physics_adapter.get_state()
        vector_only = options.get("vector_only")
        return self.observation_builder.build(state, vector_only=vector_only)

    def _step_impl(self, action: Any) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Advance the environment by one action."""
        self.physics_adapter.apply_action(action)
        self.physics_adapter.step(self.config.time_step_s)
        state = self.physics_adapter.get_state()
        obs = self.observation_builder.build(state)

        info: Dict[str, Any] = {
            "time_s": float(state.get("time_s", 0.0)),
            "step": self._step_count,
        }

        success = False
        if self.task and hasattr(self.task, "evaluate"):
            task_state: Mapping[str, Any]
            task_state = state if isinstance(state, Mapping) else {}
            if isinstance(action, Mapping) and "task_state" in action:
                task_state = action.get("task_state", {})
            elif isinstance(state, Mapping) and "task_state" in state:
                task_state = state.get("task_state", {})
            reward, success, task_info = self.task.evaluate(task_state)
            info["reward"] = reward
            info["task"] = task_info
            info["success"] = success

        done = success or (self._step_count + 1 >= self.config.max_steps)
        if not success:
            info["success"] = False

        return obs, info, done

    def render(self) -> Any:
        """Render stub for the workcell environment."""
        return None


__all__ = ["WorkcellEnv"]

"""
Workcell environment suite for manufacturing and assembly scenarios.

Includes base environment hooks, configuration schema, scene specs, task specs,
and reward/difficulty helpers.
"""

from src.envs.workcell_env.base import EpisodeLog, WorkcellEnvBase
from src.envs.workcell_env.config import PRESETS, WorkcellEnvConfig
from src.envs.workcell_env.env import WorkcellEnv
from src.envs.workcell_env.scene.scene_spec import (
    ContainerSpec,
    ConveyorSpec,
    FixtureSpec,
    PartSpec,
    StationSpec,
    ToolSpec,
    WorkcellSceneSpec,
)
from src.envs.workcell_env.tasks.task_base import ActionStepSpec, TaskGraphSpec, TaskSpec
from src.envs.workcell_env.rewards.reward_terms import WorkcellRewardTerms, compute_reward
from src.envs.workcell_env.difficulty.difficulty_features import (
    WorkcellDifficultyFeatures,
    compute_difficulty_features,
)
from src.envs.workcell_env.compiler import (
    CompilationResult,
    WorkcellTaskCompiler,
    compile_workcell_task,
)

__all__ = [
    "EpisodeLog",
    "WorkcellEnvBase",
    "WorkcellEnv",
    "WorkcellEnvConfig",
    "PRESETS",
    "WorkcellSceneSpec",
    "StationSpec",
    "FixtureSpec",
    "PartSpec",
    "ToolSpec",
    "ConveyorSpec",
    "ContainerSpec",
    "TaskSpec",
    "TaskGraphSpec",
    "ActionStepSpec",
    "WorkcellRewardTerms",
    "compute_reward",
    "WorkcellDifficultyFeatures",
    "compute_difficulty_features",
    "CompilationResult",
    "WorkcellTaskCompiler",
    "compile_workcell_task",
]

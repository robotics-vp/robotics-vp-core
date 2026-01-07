from __future__ import annotations

import importlib.util

import pytest


def _mujoco_available() -> bool:
    return importlib.util.find_spec("mujoco") is not None


@pytest.mark.no_mujoco
def test_mujoco_missing_dependency_error() -> None:
    if _mujoco_available():
        pytest.skip("mujoco installed")
    from src.envs.workcell_env import WorkcellEnv
    from src.envs.workcell_env.config import WorkcellEnvConfig

    config = WorkcellEnvConfig(physics_mode="MUJOCO")
    with pytest.raises(ImportError, match="mujoco"):
        WorkcellEnv(config=config)

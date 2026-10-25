"""The base installation must not require the optional PyBullet simulator."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def _run_isolated(code: str) -> None:
    # A fresh interpreter avoids a prior test's imports hiding this regression.
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_backend_imports_do_not_require_pybullet() -> None:
    _run_isolated(
        """
        import sys

        # Model an installation without PyBullet, even if the host has it.
        sys.modules["pybullet"] = None
        sys.modules["pybullet_data"] = None

        from src.envs.physics import PhysicsBackend, PyBulletBackend, IsaacBackend
        from src.envs.physics.backend_factory import make_backend
        from src.world_model.sim_synth_physics import materialize_backend_shadow_execution

        assert issubclass(IsaacBackend, PhysicsBackend)
        assert issubclass(PyBulletBackend, PhysicsBackend)
        assert callable(make_backend)
        assert callable(materialize_backend_shadow_execution)
        assert "src.envs.physics.dishwashing_physics_env" not in sys.modules
        """
    )


def test_legacy_environment_exports_still_require_pybullet() -> None:
    _run_isolated(
        """
        import sys

        sys.modules["pybullet"] = None
        sys.modules["pybullet_data"] = None

        import src.envs.physics as physics

        for name in ("DishwashingPhysicsEnv", "create_physics_env"):
            try:
                getattr(physics, name)
            except ModuleNotFoundError as exc:
                assert exc.name == "pybullet", exc
            else:
                raise AssertionError(f"{name} must require PyBullet")

        try:
            physics.unknown_backend
        except AttributeError:
            pass
        else:
            raise AssertionError("Unknown exports must raise AttributeError")
        """
    )


def test_legacy_exports_keep_original_class_and_factory() -> None:
    _run_isolated(
        """
        import sys
        from types import ModuleType

        # Import-only doubles; this test does not claim to run PyBullet physics.
        sys.modules["pybullet"] = ModuleType("pybullet")
        sys.modules["pybullet_data"] = ModuleType("pybullet_data")

        from src.envs.physics import DishwashingPhysicsEnv, create_physics_env
        from src.envs.physics import dishwashing_physics_env

        assert DishwashingPhysicsEnv is dishwashing_physics_env.DishwashingPhysicsEnv
        assert create_physics_env is dishwashing_physics_env.create_physics_env
        """
    )

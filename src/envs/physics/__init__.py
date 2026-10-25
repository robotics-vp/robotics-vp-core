"""
Physics-based Simulation Environments

This module contains physics-enabled environments for:
- PyBullet (CPU-friendly, easy prototyping) ✅ IMPLEMENTED
- Isaac Gym / Isaac Sim shadow-contract backend now present; real asset/runtime binding remains TODO
- MuJoCo (high-quality contact physics) - TODO

All physics envs:
1. Accept (speed, care) actions
2. Return video observations (T, C, H, W) from virtual camera
3. Provide error/success outcomes based on physics
4. Match DishwashingEnv interface: reset() -> obs, step(action) -> (obs, info, done)

Physics Backend Abstraction Layer:
- PhysicsBackend: Abstract base class for engine-agnostic interface
- PyBulletBackend: Wrapper for existing PyBullet environments
- IsaacBackend: Explicit shadow-contract backend for future Isaac Gym / Isaac Sim integration
- make_backend(): Factory function to create backends

The legacy dishwashing environment requires the optional ``pybullet`` package.
It is loaded only when requested, so the backend interfaces and Isaac shadow
backend remain usable with the base installation.
"""

from typing import Any

# Physics Backend Abstraction Layer
from .backend_factory import get_backend_info, list_available_backends, make_backend
from .base_engine import PhysicsBackend
from .isaac_backend import IsaacBackend
from .pybullet_backend import PyBulletBackend

__all__ = [
    "DishwashingPhysicsEnv",
    "IsaacBackend",
    "PhysicsBackend",
    "PyBulletBackend",
    "create_physics_env",
    "get_backend_info",
    "list_available_backends",
    "make_backend",
]


def __getattr__(name: str) -> Any:
    """Load optional PyBullet exports without changing their public names."""
    if name in {"DishwashingPhysicsEnv", "create_physics_env"}:
        from . import dishwashing_physics_env

        return getattr(dishwashing_physics_env, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

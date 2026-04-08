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
"""

from src.envs.physics.dishwashing_physics_env import DishwashingPhysicsEnv, create_physics_env

# Physics Backend Abstraction Layer
from .base_engine import PhysicsBackend
from .pybullet_backend import PyBulletBackend
from .isaac_backend import IsaacBackend
from .backend_factory import make_backend, list_available_backends, get_backend_info

__all__ = [
    # Legacy exports
    'DishwashingPhysicsEnv',
    'create_physics_env',
    # Backend abstraction layer
    'PhysicsBackend',
    'PyBulletBackend',
    'IsaacBackend',
    'make_backend',
    'list_available_backends',
    'get_backend_info',
]

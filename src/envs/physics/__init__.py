"""
Physics-based Simulation Environments

This module contains physics-enabled environments for:
- PyBullet (CPU-friendly, easy prototyping) ✅ IMPLEMENTED
- Isaac Gym (GPU-accelerated, high-throughput) - TODO
- MuJoCo (high-quality contact physics) - TODO

All physics envs:
1. Accept (speed, care) actions
2. Return video observations (T, C, H, W) from virtual camera
3. Provide error/success outcomes based on physics
4. Match DishwashingEnv interface: reset() -> obs, step(action) -> (obs, info, done)
"""

from src.envs.physics.dishwashing_physics_env import DishwashingPhysicsEnv, create_physics_env

__all__ = ['DishwashingPhysicsEnv', 'create_physics_env']

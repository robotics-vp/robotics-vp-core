"""
Physics Backend Factory.

Factory function to create physics backends based on engine type.
Centralizes backend instantiation logic.
"""

from typing import Any, Dict, Optional, Literal

from .base_engine import PhysicsBackend
from .pybullet_backend import PyBulletBackend
from .isaac_backend import IsaacBackend


def make_backend(
    engine_type: Literal["pybullet", "isaac"],
    env: Optional[Any] = None,
    env_name: str = "unknown",
    summarize_fn=None,
    env_config: Optional[Dict[str, Any]] = None,
    num_envs: int = 1,
    device: str = "cuda:0",
) -> PhysicsBackend:
    """
    Create a physics backend based on engine type.

    This factory centralizes backend instantiation and allows seamless
    switching between physics engines while maintaining consistent API.

    Args:
        engine_type: Physics engine to use ("pybullet" or "isaac")
        env: For PyBullet - existing environment instance to wrap
        env_name: Environment identifier (e.g., "drawer_vase", "dishwashing")
        summarize_fn: Optional custom summarization function for episode info
        env_config: For Isaac - configuration dictionary
        num_envs: For Isaac - number of parallel environments
        device: For Isaac - CUDA device string

    Returns:
        PhysicsBackend: Configured backend instance

    Raises:
        ValueError: If engine_type is unknown or required args are missing
        NotImplementedError: If Isaac backend is requested (stub only)

    Examples:
        # Wrap existing PyBullet environment
        from src.envs.drawer_vase_physics_env import DrawerVasePhysicsEnv
        pybullet_env = DrawerVasePhysicsEnv()
        backend = make_backend("pybullet", env=pybullet_env, env_name="drawer_vase")

        # Create Isaac backend (future)
        backend = make_backend(
            "isaac",
            env_config={"task": "drawer_vase", "robot": "franka"},
            num_envs=32,
            device="cuda:0"
        )

    Integration with EconParams:
        The returned backend provides consistent interface for:
        - reset() / step() - standard RL interface
        - get_episode_info() - returns EpisodeInfoSummary for datapack creation
        - get_info_history() - for detailed analysis

        Use with ObjectiveProfile:
        ```python
        from src.valuation.datapack_schema import ObjectiveProfile
        from src.valuation.datapacks import build_datapack_meta_from_episode

        backend = make_backend("pybullet", env=my_env, env_name="drawer_vase")

        # Run episode
        obs = backend.reset()
        done = False
        while not done:
            action = policy(obs)
            obs, reward, done, info = backend.step(action)

        # Get summary
        summary = backend.get_episode_info()

        # Build datapack with objective profile
        obj_profile = ObjectiveProfile(
            objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],
            env_name="drawer_vase",
            engine_type=backend.engine_type,
            ...
        )
        datapack = build_datapack_meta_from_episode(
            summary, econ_params,
            ...,
            objective_profile=obj_profile
        )
        ```
    """
    if engine_type == "pybullet":
        if env is None:
            raise ValueError(
                "PyBullet backend requires 'env' argument - "
                "pass an existing PyBullet environment instance"
            )
        return PyBulletBackend(env=env, env_name=env_name, summarize_fn=summarize_fn)

    elif engine_type == "isaac":
        # Isaac backend is currently a stub - will raise NotImplementedError
        return IsaacBackend(
            env_config=env_config,
            num_envs=num_envs,
            device=device,
        )

    else:
        raise ValueError(
            f"Unknown engine_type '{engine_type}'. "
            f"Supported: 'pybullet', 'isaac'"
        )


def list_available_backends() -> list:
    """
    List available physics backends.

    Returns:
        list: Names of available backends
    """
    return ["pybullet", "isaac"]


def get_backend_info(engine_type: str) -> Dict[str, Any]:
    """
    Get information about a specific backend.

    Args:
        engine_type: Backend to query

    Returns:
        dict: Backend information including status, requirements, features
    """
    info = {
        "pybullet": {
            "status": "available",
            "description": "PyBullet physics engine wrapper",
            "requirements": ["pybullet"],
            "features": [
                "Single environment execution",
                "CPU-based physics",
                "Gym API compatible",
                "Episode summarization",
                "Energy tracking",
            ],
            "use_cases": [
                "drawer_vase",
                "dishwashing",
            ],
        },
        "isaac": {
            "status": "stub",
            "description": "Isaac Gym / Isaac Lab physics engine (not yet implemented)",
            "requirements": ["isaacgym", "torch"],
            "features": [
                "Parallel environment execution (vectorized)",
                "GPU-based physics",
                "High-performance training",
                "Tensor-based observations/actions",
            ],
            "use_cases": [
                "Large-scale policy training",
                "Sim-to-real transfer",
            ],
        },
    }

    if engine_type not in info:
        raise ValueError(f"Unknown engine_type '{engine_type}'")

    return info[engine_type]


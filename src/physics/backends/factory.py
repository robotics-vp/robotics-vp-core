"""
Factory for physics backends.
"""
from typing import Dict

from src.physics.backends.base import PhysicsBackend
from src.physics.backends.pybullet_backend import PyBulletBackend
from src.physics.backends.isaac_stub_backend import IsaacStubBackend
from src.physics.backends.mobility_heuristics import HeuristicMobilityPolicy


def make_backend(name: str, config: Dict) -> PhysicsBackend:
    name = (name or "pybullet").lower()
    if name == "pybullet":
        mobility_policy = HeuristicMobilityPolicy() if config.get("use_mobility_policy") else None
        return PyBulletBackend(econ_preset=config.get("econ_preset", "toy"), mobility_policy=mobility_policy)
    if name == "isaac_stub":
        mobility_policy = HeuristicMobilityPolicy() if config.get("use_mobility_policy") else None
        return IsaacStubBackend(mobility_policy=mobility_policy)
    raise ValueError(f"Unknown physics backend: {name}")

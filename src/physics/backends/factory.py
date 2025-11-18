"""
Factory for physics backends.
"""
from typing import Dict

from src.physics.backends.base import PhysicsBackend
from src.physics.backends.pybullet_backend import PyBulletBackend
from src.physics.backends.isaac_stub_backend import IsaacStubBackend


def make_backend(name: str, config: Dict) -> PhysicsBackend:
    name = (name or "pybullet").lower()
    if name == "pybullet":
        return PyBulletBackend(econ_preset=config.get("econ_preset", "toy"))
    if name == "isaac_stub":
        return IsaacStubBackend()
    raise ValueError(f"Unknown physics backend: {name}")

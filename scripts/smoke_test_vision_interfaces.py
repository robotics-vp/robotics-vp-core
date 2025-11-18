#!/usr/bin/env python3
"""
Smoke for vision/VLA interfaces across backends.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.physics.backends.pybullet_backend import PyBulletBackend
from src.physics.backends.isaac_stub_backend import IsaacStubBackend
from src.vision.backbone_stub import VisionBackboneStub
from src.vision.policy_observation_builder import PolicyObservationBuilder
from src.vision.interfaces import VisionFrame, PolicyObservation


def main():
    backbone = VisionBackboneStub()
    builder = PolicyObservationBuilder(backbone)

    # PyBullet frame
    pyb = PyBulletBackend()
    pyb.reset(seed=0)
    frame_py = pyb.build_vision_frame(task_id="task_vis", episode_id="ep_vis", timestep=0)
    latent_py = backbone.encode_frame(frame_py)
    obs_py = builder.build(frame_py, pyb.get_state_summary())
    assert obs_py.to_dict()["latent"]["backend"] == "pybullet"
    # Round-trip
    rt_py = PolicyObservation.from_dict(obs_py.to_dict())
    assert rt_py.to_dict() == obs_py.to_dict()
    # Deterministic PolicyObservation serialization
    assert obs_py.to_dict() == builder.build(frame_py, pyb.get_state_summary()).to_dict()

    # Isaac stub frame
    stub = IsaacStubBackend()
    frame_stub = stub.build_vision_frame(task_id="task_vis", episode_id="ep_vis", timestep=1)
    latent_stub = backbone.encode_frame(frame_stub)
    obs_stub = builder.build(frame_stub, stub.get_state_summary())
    rt_stub = PolicyObservation.from_dict(obs_stub.to_dict())
    assert rt_stub.to_dict() == obs_stub.to_dict()
    assert obs_stub.latent.backend == "isaac_stub"
    assert obs_stub.to_dict() == builder.build(frame_stub, stub.get_state_summary()).to_dict()

    # Determinism
    latent_again = backbone.encode_frame(frame_py)
    assert latent_again.latent == latent_py.latent

    print("[smoke_test_vision_interfaces] All tests passed.")


if __name__ == "__main__":
    main()

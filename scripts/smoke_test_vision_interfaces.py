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
from src.policies.registry import build_all_policies
from src.vision.policy_observation_builder import PolicyObservationBuilder
from src.vision.interfaces import VisionFrame, PolicyObservation


def main():
    vision_encoder = build_all_policies().vision_encoder
    builder = PolicyObservationBuilder(vision_encoder)

    # PyBullet frame
    pyb = PyBulletBackend()
    pyb.reset(seed=0)
    frame_py = pyb.build_vision_frame(task_id="task_vis", episode_id="ep_vis", timestep=0)
    latent_py = vision_encoder.encode(frame_py)
    obs_py = builder.build(frame_py, pyb.get_state_summary())
    feats_py = builder.build_policy_features(frame_py, pyb.get_state_summary())
    assert feats_py["vision_latent"]["backend"] == "pybullet"
    assert obs_py.to_dict()["latent"]["backend"] == "pybullet"
    assert frame_py.width > 0 and frame_py.height > 0
    assert frame_py.camera_intrinsics and frame_py.camera_extrinsics
    assert frame_py.state_digest
    assert obs_py.metadata.get("backend") == "pybullet"
    # Round-trip
    rt_py = PolicyObservation.from_dict(obs_py.to_dict())
    assert rt_py.to_dict() == obs_py.to_dict()
    # Deterministic PolicyObservation serialization
    assert obs_py.to_dict() == builder.build(frame_py, pyb.get_state_summary()).to_dict()
    # Deterministic state digest for fixed seeds
    frame_py_again = pyb.build_vision_frame(task_id="task_vis", episode_id="ep_vis", timestep=0)
    assert frame_py.state_digest == frame_py_again.state_digest

    # Isaac stub frame
    stub = IsaacStubBackend()
    frame_stub = stub.build_vision_frame(task_id="task_vis", episode_id="ep_vis", timestep=1)
    latent_stub = vision_encoder.encode(frame_stub)
    obs_stub = builder.build(frame_stub, stub.get_state_summary())
    rt_stub = PolicyObservation.from_dict(obs_stub.to_dict())
    assert rt_stub.to_dict() == obs_stub.to_dict()
    assert obs_stub.latent.backend == "isaac_stub"
    assert frame_stub.width > 0 and frame_stub.height > 0
    assert frame_stub.camera_intrinsics and frame_stub.camera_extrinsics
    assert frame_stub.state_digest
    assert obs_stub.metadata.get("backend") == "isaac_stub"
    assert obs_stub.to_dict() == builder.build(frame_stub, stub.get_state_summary()).to_dict()

    # Determinism
    latent_again = vision_encoder.encode(frame_py)
    assert latent_again.latent == latent_py.latent

    print("[smoke_test_vision_interfaces] All tests passed.")


if __name__ == "__main__":
    main()

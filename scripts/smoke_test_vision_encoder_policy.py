#!/usr/bin/env python3
"""
Smoke test for VisionEncoderPolicy determinism and shape.
"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.policies.registry import build_all_policies
from src.physics.backends.pybullet_backend import PyBulletBackend
from src.vision.config import load_vision_config


def main():
    policies = build_all_policies()
    encoder = policies.vision_encoder
    cfg = load_vision_config()
    backend = PyBulletBackend()
    backend.reset(seed=0)
    frame = backend.build_vision_frame(task_id="task_vis", episode_id="ep_vis", timestep=0)
    latent = encoder.encode(frame)
    assert latent.model_name
    assert len(latent.latent) == int(cfg.get("latent_dim", 16))
    repeat = encoder.encode(frame)
    assert latent.latent == repeat.latent
    print("[smoke_test_vision_encoder_policy] PASS")


if __name__ == "__main__":
    main()

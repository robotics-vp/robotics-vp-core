#!/usr/bin/env python3
"""
End-to-end vision stack smoke across physics backends.

Builds VisionFrames, encodes them, constructs PolicyObservations, and runs a
tiny SAC actor forward pass to ensure deterministic wiring.
"""
import sys
from pathlib import Path

import numpy as np
import torch

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.physics.backends.factory import make_backend  # noqa: E402
from src.policies.registry import build_all_policies  # noqa: E402
from src.vision.policy_observation_builder import PolicyObservationBuilder  # noqa: E402
from src.rl.sac import Actor  # noqa: E402


def _sample_action(latent_vec):
    torch.manual_seed(0)
    actor = Actor(latent_dim=len(latent_vec), action_dim=2, hidden_dim=16)
    with torch.no_grad():
        latent_t = torch.tensor([latent_vec], dtype=torch.float32)
        action, _ = actor.sample(latent_t, deterministic=True, return_log_prob=False)
    return action.numpy()[0].tolist()


def run_backend(name: str):
    policies = build_all_policies()
    builder = PolicyObservationBuilder(policies.vision_encoder)
    backend = make_backend(name, {"econ_preset": "toy"})
    if name == "pybullet":
        backend.reset(seed=0)
    state_summary = backend.get_state_summary()
    frame = backend.build_vision_frame(task_id="task_e2e", episode_id="ep_e2e", timestep=0)
    obs = builder.build(frame, state_summary)
    features = builder.build_policy_features(frame, state_summary)
    assert features["backend"] in ("pybullet", "isaac", "isaac_stub")
    assert obs.latent.latent == builder.build(frame, state_summary).latent.latent
    action = _sample_action(obs.latent.latent)
    assert len(action) == 2
    # Deterministic reuse with identical seed/state
    frame_again = backend.build_vision_frame(task_id="task_e2e", episode_id="ep_e2e", timestep=0)
    obs_again = builder.build(frame_again, state_summary)
    assert obs_again.latent.latent == obs.latent.latent
    assert np.allclose(np.array(_sample_action(obs_again.latent.latent)), np.array(action))


def main():
    for backend_name in ("pybullet", "isaac_stub", "isaac"):
        run_backend(backend_name)
    print("[smoke_test_vision_stack_e2e] PASS")


if __name__ == "__main__":
    main()

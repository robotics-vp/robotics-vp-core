#!/usr/bin/env python3
"""
Smoke test for engine selection via make_backend (pybullet fallback).
"""
import argparse
from src.envs.drawer_vase_physics_env import DrawerVasePhysicsEnv, DrawerVaseConfig, summarize_drawer_vase_episode
from src.envs.physics.backend_factory import make_backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-type", type=str, default="pybullet")
    args = parser.parse_args()
    env = DrawerVasePhysicsEnv(DrawerVaseConfig(), obs_mode="state", render_mode=None)
    backend = make_backend(engine_type=args.engine_type, env=env, env_name="drawer_vase", summarize_fn=summarize_drawer_vase_episode)
    obs = backend.reset()
    done = False
    steps = 0
    while not done and steps < 5:
        import numpy as np
        action = np.zeros(3, dtype=float)
        obs, reward, done, info = backend.step(action)
        steps += 1
    summary = backend.get_episode_info()
    print("Engine type:", backend.engine_type)
    print("Episode summary MPL:", summary.mpl_episode, "Energy:", summary.energy_Wh)


if __name__ == "__main__":
    main()

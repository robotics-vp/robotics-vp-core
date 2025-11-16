#!/usr/bin/env python3
"""
Quick energy debug for the drawer+vase articulated arm env.

Runs a few random-action episodes and prints per-limb/joint energy plus coordination metrics.
Purely for sanity; no training or weighting changes.
"""
import argparse
import numpy as np

from src.envs.drawer_vase_arm_env import DrawerVaseArmEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = DrawerVaseArmEnv()
    rng = np.random.default_rng(0)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action = rng.uniform(low=-1.0, high=1.0, size=len(env.controlled_joint_ids))
            obs, _, done, truncated, info = env.step(action)
        print(f"[Episode {ep+1}] energy_Wh={info.get('energy_Wh', 0.0):.4f}")
        print(f"  energy_per_limb: {info.get('energy_per_limb', {})}")
        print(f"  energy_per_joint keys: {list(info.get('energy_per_joint', {}).keys())}")
        print(f"  coordination_metrics: {info.get('coordination_metrics', {})}")


if __name__ == "__main__":
    main()

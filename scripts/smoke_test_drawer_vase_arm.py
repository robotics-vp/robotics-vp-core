#!/usr/bin/env python3
"""
Smoke test for DrawerVaseArmEnv to verify energy metrics are populated.
"""
import numpy as np

from src.envs.drawer_vase_arm_env import DrawerVaseArmEnv
from src.envs.dishwashing_env import EpisodeInfoSummary
from src.envs.dishwashing_env import summarize_episode_info as summarize_dish


def main():
    env = DrawerVaseArmEnv()
    obs, info = env.reset()
    info_history = []
    done = False
    while not done:
        action = np.random.uniform(-1, 1, size=len(env.controlled_joint_ids))
        obs, _, done, truncated, info = env.step(action)
        info_history.append(info)
        if truncated:
            done = True
    # Reuse EpisodeInfoSummary structure by converting to minimal dict
    summary = summarize_dish(info_history)
    print("=== DrawerVaseArmEnv Summary ===")
    print(f"MPL: {summary.mpl_episode:.3f} Err: {summary.error_rate_episode:.3f}")
    print(f"Energy Wh: {summary.energy_Wh:.6f}")
    print(f"Energy per limb: {summary.energy_per_limb}")
    print(f"Energy per joint keys: {list(summary.energy_per_joint.keys())}")


if __name__ == "__main__":
    main()

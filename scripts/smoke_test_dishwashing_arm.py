#!/usr/bin/env python3
"""
Smoke test for DishwashingArmEnv to verify energy metrics are populated.
"""
import numpy as np

from src.envs.dishwashing_arm_env import DishwashingArmEnv
from src.envs.dishwashing_env import summarize_episode_info


def main():
    env = DishwashingArmEnv()
    obs = env.reset()
    info_history = []
    done = False
    while not done:
        action = np.random.uniform(-1, 1, size=len(env.controlled_joint_ids))
        obs, info, done = env.step(action)
        info_history.append(info)
    summary = summarize_episode_info(info_history)
    print("=== DishwashingArmEnv Summary ===")
    print(f"MPL: {summary.mpl_episode:.3f} Err: {summary.error_rate_episode:.3f}")
    print(f"Energy Wh: {summary.energy_Wh:.6f}")
    print(f"Energy per limb: {summary.energy_per_limb}")
    print(f"Energy per joint keys: {list(summary.energy_per_joint.keys())}")


if __name__ == "__main__":
    main()

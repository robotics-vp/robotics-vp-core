#!/usr/bin/env python3
"""
Run paired energy profile interventions and log EpisodeInfoSummary.
"""
import argparse
import json
import numpy as np

from src.envs.drawer_vase_physics_env import DrawerVasePhysicsEnv, DrawerVaseConfig, summarize_drawer_vase_episode
from src.envs.drawer_vase_arm_env import DrawerVaseArmEnv
from src.envs.dishwashing_arm_env import DishwashingArmEnv
from policies.scripted.drawer_open_avoid_vase import DrawerOpenAvoidVasePolicy
from src.controllers.energy_profile import EnergyProfile, apply_energy_profile_to_action


PROFILES = {
    "BASE": EnergyProfile(),
    "BOOST": EnergyProfile(speed_scale=1.2, torque_scale_shoulder=1.2, torque_scale_elbow=1.2, torque_scale_wrist=1.2, torque_scale_gripper=1.1),
    "SAVER": EnergyProfile(speed_scale=0.8, torque_scale_shoulder=0.8, torque_scale_elbow=0.8, torque_scale_wrist=0.8, torque_scale_gripper=0.9, safety_margin_scale=1.1),
    "SAFE": EnergyProfile(speed_scale=0.7, torque_scale_shoulder=0.7, torque_scale_elbow=0.7, torque_scale_wrist=0.7, torque_scale_gripper=0.8, safety_margin_scale=1.2),
}


def run_episode(env, policy, profile_name, env_type):
    profile = PROFILES[profile_name]
    if env_type == "drawer_vase_arm":
        obs, info = env.reset()
    else:
        obs, info = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    info_history = []
    done = False
    while not done:
        if env_type == "drawer_vase_arm":
            action = np.random.uniform(-1, 1, size=len(env.controlled_joint_ids))
        else:
            action = policy.predict(obs)
        action = apply_energy_profile_to_action(action, profile)
        if env_type == "drawer_vase_arm":
            obs, _, done, truncated, info = env.step(action)
        else:
            obs, reward, done, truncated, info = env.step(action)
        info_history.append(info)
        if truncated:
            done = True
    if env_type == "drawer_vase_arm":
        summary = summarize_drawer_vase_episode(info_history)
    else:
        summary = summarize_drawer_vase_episode(info_history)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Energy interventions")
    parser.add_argument("--env", type=str, default="drawer_vase")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=str, default="data/energy_interventions.jsonl")
    args = parser.parse_args()

    if args.env == "drawer_vase":
        env = DrawerVasePhysicsEnv(DrawerVaseConfig(), obs_mode="state", render_mode=None)
        policy = DrawerOpenAvoidVasePolicy()
    elif args.env == "drawer_vase_arm":
        env = DrawerVaseArmEnv()
        policy = None
    elif args.env == "dishwashing_arm":
        from src.envs.dishwashing_arm_env import DishwashingArmEnv
        env = DishwashingArmEnv()
        policy = None
    else:
        raise ValueError(f"Unknown env {args.env}")

    records = []
    for ep in range(args.episodes):
        for profile_name in PROFILES.keys():
            summary = run_episode(env, policy, profile_name, args.env)
            summ_dict = {k: (v if not hasattr(v, "items") else {kk: vv for kk, vv in v.items()}) for k, v in summary.__dict__.items()}
            # Convert numpy types to python
            def _to_py(obj):
                import numpy as np
                if isinstance(obj, np.generic):
                    return obj.item()
                if isinstance(obj, dict):
                    return {kk: _to_py(vv) for kk, vv in obj.items()}
                if isinstance(obj, list):
                    return [_to_py(vv) for vv in obj]
                return obj
            summ_dict = _to_py(summ_dict)
            record = {
                "episode": int(ep),
                "profile": profile_name,
                "env": args.env,
                "summary": summ_dict,
            }
            records.append(record)

    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

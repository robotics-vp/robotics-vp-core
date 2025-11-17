#!/usr/bin/env python3
"""
Run paired energy profile interventions and log EpisodeInfoSummary.
"""
import argparse
import json
import numpy as np
import pybullet as p

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


def run_scripted_drawer_open(env: DrawerVaseArmEnv, profile: EnergyProfile):
    """Scripted drawer open with IK waypoints; returns info_history."""
    _, info = env.reset()
    info_history = [info]

    # Waypoints: approach, skim near vase, contact drawer face
    pregrasp = env.drawer_target + np.array([0.0, -0.18 * profile.safety_margin_scale, 0.08])
    offset = max(0.0, (profile.safety_margin_scale - 1.0) * 0.02)
    skim_vase = env.vase_pos + np.array([offset, 0.0, 0.01])
    contact = env.drawer_target  # reaching this should mark success

    def _steps(base, speed_override=None):
        sp = speed_override if speed_override is not None else profile.speed_scale
        return max(3, int(base / sp))

    def _interpolate_to(target, base_steps, speed_override=None):
        start, _ = env._ee_state()
        steps = _steps(base_steps, speed_override)
        for i in range(steps):
            frac = float(i + 1) / steps
            interp = start + frac * (target - start)
            action = {"target_pos": interp, "speed_scale": speed_override or profile.speed_scale}
            _, _, done, truncated, info = env.step(action)
            info_history.append(info)
            if done or truncated:
                return True
        return False

    for target, base_steps, speed_override in [
        (pregrasp, 80, None),
        # Aggressive skim to invite near-miss/collision when speed is high
        (skim_vase, 6, profile.speed_scale * 2.0),
        # Fast poke directly at vase before final contact
        (env.vase_pos, 3, profile.speed_scale * 3.0),
        (contact, 140, None),
    ]:
        if _interpolate_to(target, base_steps, speed_override):
            break

    # If not yet successful, keep nudging toward contact for a short horizon
    tries = 0
    while not env.success and env.step_count < env.max_steps and tries < 200:
        if _interpolate_to(contact, 12, speed_override=profile.speed_scale):
            break
        tries += 1

    return info_history


def run_episode(env, policy, profile_name, env_type):
    profile = PROFILES[profile_name]
    if env_type == "drawer_vase_arm":
        info_history = run_scripted_drawer_open(env, profile)
        # Early exit handled inside scripted controller; summarize below
        summary = summarize_drawer_vase_episode(info_history)
        return summary
    obs, info = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    info_history = []
    done = False
    while not done:
        if env_type == "drawer_vase_arm":
            # Simple IK-based controller toward drawer target to make task non-trivial
            joint_states = p.getJointStates(env.robot_id, env.controlled_joint_ids)
            joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
            target_joint = p.calculateInverseKinematics(env.robot_id, env.controlled_joint_ids[-1], env.drawer_target)
            target_joint = np.array(target_joint[: len(env.controlled_joint_ids)], dtype=np.float32)
            action = 0.5 * (target_joint - joint_pos)
            action = np.clip(action, -1.0, 1.0)
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

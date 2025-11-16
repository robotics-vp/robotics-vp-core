#!/usr/bin/env python3
"""
Evaluate the articulated drawer+vase arm env with a trivial scripted policy and emit datapacks.

This is plumbing-only: builds EpisodeInfoSummary and datapacks (schema_version 2.0-energy)
without touching Phase B weighting or rewards.
"""
import argparse
import json
import numpy as np

from dataclasses import asdict

from src.envs.drawer_vase_arm_env import DrawerVaseArmEnv
from src.envs.dishwashing_env import EpisodeInfoSummary
from src.valuation.datapacks import build_datapack_from_episode


def make_summary(info_history) -> EpisodeInfoSummary:
    """Build EpisodeInfoSummary from the arm env's per-step info."""
    last = info_history[-1]
    t_hours = last.get("t", 0.0) / 3600.0 if "t" in last else 0.0
    units = last.get("completed", 0.0)
    errors = last.get("errors", 0.0)
    energy = last.get("energy_Wh", 0.0)
    mpl_episode = (units / t_hours) if t_hours > 0 else 0.0
    ep_episode = (units / energy) if energy > 0 else 0.0
    err_rate = errors / max(units, 1.0)
    return EpisodeInfoSummary(
        termination_reason=last.get("terminated_reason", "unknown") or "unknown",
        mpl_episode=mpl_episode,
        ep_episode=ep_episode,
        error_rate_episode=err_rate,
        throughput_units_per_hour=mpl_episode,
        energy_Wh=energy,
        energy_Wh_per_unit=energy / max(units, 1e-6) if energy > 0 else 0.0,
        energy_Wh_per_hour=energy / max(t_hours, 1e-6) if energy > 0 else 0.0,
        limb_energy_Wh=last.get("limb_energy_Wh", {}),
        skill_energy_Wh=last.get("skill_energy_Wh", {}),
        energy_per_limb=last.get("energy_per_limb", {}),
        energy_per_skill=last.get("energy_per_skill", {}),
        energy_per_joint=last.get("energy_per_joint", {}),
        energy_per_effector=last.get("energy_per_effector", {}),
        coordination_metrics=last.get("coordination_metrics", {}),
        profit=0.0,
        wage_parity=None,
    )


def scripted_policy(obs):
    """Very simple proportional-to-zero policy to keep motion mild."""
    return -0.5 * np.tanh(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out-datapacks", type=str, default="data/datapacks_drawer_vase_arm.jsonl")
    args = parser.parse_args()

    env = DrawerVaseArmEnv()
    datapacks = []

    for ep in range(args.episodes):
        obs, info = env.reset()
        info_history = [info]
        done = False
        truncated = False
        steps = 0
        while not (done or truncated):
            action = scripted_policy(obs)
            obs, _, done, truncated, info = env.step(action)
            info_history.append(info)
            steps += 1
            if steps >= env.max_steps:
                break

        summary = make_summary(info_history)
        datapack = build_datapack_from_episode(
            episode_info=summary,
            econ_params=env.econ_params,
            condition_profile={"env": "drawer_vase_arm", "tags": ["scripted"]},
            agent_profile={"policy": "scripted"},
            brick_id=None,
            env_type="drawer_vase_arm",
            extra_tags=["drawer_vase_arm"],
            semantic_energy_drivers=info_history[-1].get("energy_driver_tags", None),
        )
        datapacks.append(datapack)
        print(f"[Episode {ep+1}] term={summary.termination_reason} mpl={summary.mpl_episode:.3f} "
              f"err={summary.error_rate_episode:.3f} energy_Wh={summary.energy_Wh:.4f}")

    # Write JSONL for easy downstream ingest
    with open(args.out_datapacks, "w") as f:
        for dp in datapacks:
            f.write(json.dumps(dp) + "\n")
    print(f"Wrote {len(datapacks)} datapacks to {args.out_datapacks}")


if __name__ == "__main__":
    main()

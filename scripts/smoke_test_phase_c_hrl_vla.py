#!/usr/bin/env python3
"""
Lightweight smoke test for Phase C HRL/VLA stack plumbing.

Runs a few Drawer+Vase episodes with the scripted low-level policy,
summarizes with EpisodeInfoSummary, builds datapacks, and sanity-checks imports.
"""
import argparse
import json
import os
import numpy as np

# Import sanity checks for Phase C modules
from src.hrl import skills as _hrl_skills  # noqa: F401
from src.hrl import low_level_policy as _hrl_ll  # noqa: F401
from src.hrl import high_level_controller as _hrl_hl  # noqa: F401
from src.hrl import skill_termination as _hrl_term  # noqa: F401
from src.hrl import hrl_trainer as _hrl_trainer  # noqa: F401
from src.vision import encoder_with_heads as _vision_heads  # noqa: F401
from src.vla import transformer_planner as _vla_planner  # noqa: F401
from src.vla import vla_trainer as _vla_trainer  # noqa: F401
from src.sima import co_agent as _sima_co  # noqa: F401
from src.sima import narrator as _sima_narrator  # noqa: F401
from src.sima import trajectory_generator as _sima_traj  # noqa: F401

from src.envs.drawer_vase_physics_env import DrawerVasePhysicsEnv, DrawerVaseConfig, summarize_drawer_vase_episode
from policies.scripted.drawer_open_avoid_vase import DrawerOpenAvoidVasePolicy
from src.config.internal_profile import get_internal_experiment_profile
from src.config.econ_params import load_econ_params
from src.valuation.datapacks import build_datapack_from_episode


def run_smoke(episodes: int, datapack_path: str | None):
    profile = get_internal_experiment_profile("dishwashing")
    econ_params = load_econ_params(profile, preset=profile.get("econ_preset", "toy"))

    env = DrawerVasePhysicsEnv(DrawerVaseConfig(), obs_mode="state", render_mode=None, econ_params=econ_params)
    policy = DrawerOpenAvoidVasePolicy()

    summaries = []
    datapacks = []

    for ep in range(episodes):
        obs, info = env.reset()
        policy.reset()
        info_history = []
        done = False
        while not done:
            action = policy.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            info_history.append(info)
        summary = summarize_drawer_vase_episode(info_history)
        summaries.append(summary)
        datapacks.append(build_datapack_from_episode(
            summary,
            econ_params,
            condition_profile={"task": "drawer_vase", "tags": ["phase_c_smoke"]},
            agent_profile={"policy": "scripted_low_level", "controller": "hierarchy_stub"},
            brick_id=None,
        ))
        print(f"[Episode {ep+1}] term={summary.termination_reason} mpl={summary.mpl_episode:.2f} "
              f"ep={summary.ep_episode:.4f} err={summary.error_rate_episode:.3f}")

    env.close()

    if datapack_path:
        out_dir = os.path.dirname(datapack_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(datapack_path, "w") as f:
            json.dump(datapacks, f, indent=2)
        print(f"Wrote Phase C datapacks to {datapack_path}")

    print("\nAggregate:")
    print(f"  MPL mean: {np.mean([s.mpl_episode for s in summaries]):.4f}")
    print(f"  EP mean: {np.mean([s.ep_episode for s in summaries]):.4f}")
    print(f"  Error rate mean: {np.mean([s.error_rate_episode for s in summaries]):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase C HRL/VLA smoke test")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out-datapacks", type=str, default=None)
    args = parser.parse_args()

    run_smoke(args.episodes, args.out_datapacks)

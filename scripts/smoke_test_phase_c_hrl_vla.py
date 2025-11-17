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
from src.valuation.datapacks import build_datapack_from_episode, build_datapack_meta_from_episode
from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import ObjectiveProfile


from typing import Optional


def run_smoke(episodes: int, datapack_path: Optional[str], use_repo: bool = True):
    profile = get_internal_experiment_profile("dishwashing")
    econ_params = load_econ_params(profile, preset=profile.get("econ_preset", "toy"))

    env = DrawerVasePhysicsEnv(DrawerVaseConfig(), obs_mode="state", render_mode=None, econ_params=econ_params)
    policy = DrawerOpenAvoidVasePolicy()

    summaries = []
    datapacks = []
    datapack_metas = []

    def _to_python(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        return obj

    # Compute baseline from first episode
    baseline_mpl = None
    baseline_error = None
    baseline_ep = None

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

        # Set baseline from first episode
        if ep == 0:
            baseline_mpl = summary.mpl_episode
            baseline_error = summary.error_rate_episode
            baseline_ep = summary.ep_episode

        # Build unified DataPackMeta
        if use_repo:
            # Build ObjectiveProfile for econ profile logging
            obj_profile = ObjectiveProfile(
                objective_vector=[1.0, 1.0, 1.0, 1.0, 0.0],  # balanced
                wage_human=18.0,
                energy_price_kWh=0.12,
                market_region="US",
                task_family="drawer_vase",
                customer_segment="balanced",
                baseline_mpl_human=20.0,
                baseline_error_human=0.05,
                env_name="drawer_vase",
                engine_type="pybullet",
                task_type="fragility",
                econ_profile_deltas=None,
                econ_params_effective={
                    "base_rate": float(econ_params.base_rate),
                    "damage_cost": float(econ_params.damage_cost),
                    "care_cost": float(econ_params.care_cost),
                    "energy_Wh_per_attempt": float(econ_params.energy_Wh_per_attempt),
                    "max_steps": int(econ_params.max_steps),
                },
                reward_weights=None,
            )

            dp_meta = build_datapack_meta_from_episode(
                summary, econ_params,
                condition_profile={"task": "drawer_vase", "tags": ["phase_c_smoke"], "engine_type": "pybullet"},
                agent_profile={"policy": "scripted_low_level", "controller": "hierarchy_stub", "version": "v1"},
                brick_id=f"phase_c_smoke_{ep:04d}",
                env_type="drawer_vase",
                baseline_mpl=baseline_mpl,
                baseline_error=baseline_error,
                baseline_ep=baseline_ep,
                objective_profile=obj_profile,
            )
            datapack_metas.append(dp_meta)

        # Legacy format
        datapacks.append(build_datapack_from_episode(
            summary,
            econ_params,
            condition_profile={"task": "drawer_vase", "tags": ["phase_c_smoke"]},
            agent_profile={"policy": "scripted_low_level", "controller": "hierarchy_stub"},
            brick_id=None,
            env_type="drawer_vase",
        ))
        print(f"[Episode {ep+1}] term={summary.termination_reason} mpl={summary.mpl_episode:.2f} "
              f"ep={summary.ep_episode:.4f} err={summary.error_rate_episode:.3f}")

    env.close()

    # Write to DataPackRepo (unified format)
    if use_repo and datapack_metas:
        repo_dir = "data/datapacks/phase_c"
        repo = DataPackRepo(base_dir=repo_dir)
        repo.append_batch(datapack_metas)
        stats = repo.get_statistics("drawer_vase")
        print(f"\nSaved {len(datapack_metas)} datapacks to DataPackRepo ({repo_dir})")
        print(f"  Total in repo: {stats['total']}")
        print(f"  Positive: {stats['positive']}, Negative: {stats['negative']}")

    # Legacy JSON output
    if datapack_path:
        out_dir = os.path.dirname(datapack_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(datapack_path, "w") as f:
            json.dump(_to_python(datapacks), f, indent=2)
        print(f"Wrote legacy Phase C datapacks to {datapack_path}")

    print("\nAggregate:")
    print(f"  MPL mean: {np.mean([s.mpl_episode for s in summaries]):.4f}")
    print(f"  EP mean: {np.mean([s.ep_episode for s in summaries]):.4f}")
    print(f"  Error rate mean: {np.mean([s.error_rate_episode for s in summaries]):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase C HRL/VLA smoke test")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out-datapacks", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None, help="Alias for out-datapacks")
    parser.add_argument("--no-repo", action="store_true", help="Skip writing to DataPackRepo")
    args = parser.parse_args()

    target_path = args.out_datapacks or args.out_json
    run_smoke(args.episodes, target_path, use_repo=not args.no_repo)

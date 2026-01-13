#!/usr/bin/env python3
"""
Smoke test for dishwashing SAC wiring.

Runs a few episodes with random actions to sanity-check:
- Env termination reasons
- MPL/EP/error logging
- Energy/profit accounting

REGALITY COMPLIANCE: BASIC (smoke test only)
--------------------------------------------
Produces: JSON/CSV summaries with MPL/EP/error metrics
Does NOT produce: manifest, ledger, trajectory audit, selection manifest

For FULL dishwashing regality compliance, use:
    python scripts/run_dishwashing_regal.py --output-dir artifacts/dishwashing_regal

For FULL workcell (manufacturing cell) regality compliance, use:
    python scripts/run_workcell_regal.py --output-dir artifacts/workcell_regal
"""
import argparse
import json
import csv
from dataclasses import asdict
import numpy as np

from src.envs.dishwashing_env import DishwashingEnv, summarize_episode_info
from src.config.internal_profile import get_internal_experiment_profile
from src.config.econ_params import load_econ_params


from typing import Optional


def run_smoke(episodes: int, econ_preset: str, out_json: Optional[str], out_csv: Optional[str]):
    profile = get_internal_experiment_profile("dishwashing")
    econ_params = load_econ_params(profile, preset=econ_preset)
    env = DishwashingEnv(econ_params)

    summaries = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        info_history = []
        while not done:
            action = np.random.rand(2)  # random speed, care in [0,1]
            obs, info, done = env.step(action)
            info_history.append(info)

        summary = summarize_episode_info(info_history)
        summary_dict = asdict(summary)
        summary_dict["econ_preset"] = econ_params.preset
        summaries.append(summary_dict)

        print(f"[Episode {ep+1}] preset={econ_params.preset} term={summary.termination_reason} "
              f"MPL={summary.mpl_episode:.2f}/h EP={summary.ep_episode:.4f} "
              f"err_rate={summary.error_rate_episode:.3f} "
              f"energy_Wh={summary.energy_Wh:.3f} profit={summary.profit:.3f}")

    if out_json:
        with open(out_json, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"Wrote JSON summaries to {out_json}")

    if out_csv:
        keys = list(summaries[0].keys()) if summaries else []
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Wrote CSV summaries to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dishwashing SAC smoke test")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--econ-preset", type=str, default="toy", choices=["toy", "realistic"])
    parser.add_argument("--out-json", type=str, default=None, help="Path to write JSON summaries")
    parser.add_argument("--out-csv", type=str, default=None, help="Path to write CSV summaries")
    args = parser.parse_args()

    run_smoke(args.episodes, args.econ_preset, args.out_json, args.out_csv)

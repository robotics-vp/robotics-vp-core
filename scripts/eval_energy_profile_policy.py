#!/usr/bin/env python3
"""
Evaluate EnergyProfilePolicy against preset profiles on intervention contexts.
"""
import argparse
import json
import numpy as np
import torch

from src.controllers.energy_profile_policy import EnergyProfilePolicy


def load_interventions(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def build_context(summary):
    return np.array([
        summary.get("mpl_episode", 0.0),
        summary.get("error_rate_episode", 0.0),
        summary.get("energy_Wh_per_unit", 0.0),
        summary.get("energy_Wh_per_hour", 0.0),
    ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Eval EnergyProfilePolicy vs presets")
    parser.add_argument("--interventions", type=str, default="data/energy_interventions.jsonl")
    parser.add_argument("--policy", type=str, default="checkpoints/energy_profile_policy.pt")
    args = parser.parse_args()

    records = load_interventions(args.interventions)
    contexts = torch.from_numpy(np.stack([build_context(r["summary"]) for r in records]))
    model = EnergyProfilePolicy(input_dim=contexts.shape[1])
    model.load_state_dict(torch.load(args.policy, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        profiles, params = model(contexts)

    print(f"Evaluated {len(records)} contexts. Sample output params (first 5 rows):")
    print(params[:5])


if __name__ == "__main__":
    main()

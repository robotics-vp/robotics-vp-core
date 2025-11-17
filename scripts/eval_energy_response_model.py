#!/usr/bin/env python3
"""
Evaluate the trained energy response model on held-out interventions.
"""
import argparse
import json
import numpy as np
import torch

from src.valuation.energy_response_model import (
    EnergyResponseNet,
    load_energy_interventions,
    build_deltas,
    encode_sample,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interventions", type=str, default="data/energy_interventions.jsonl")
    parser.add_argument("--model", type=str, default="checkpoints/energy_response_model.pt")
    args = parser.parse_args()

    samples = load_energy_interventions(args.interventions)
    grouped = build_deltas(samples)
    env_vocab = sorted({s.env_name for s in samples})
    profile_vocab = ["BASE", "BOOST", "SAVER", "SAFE"]
    knob_keys = set()
    for s in samples:
        knob_keys.update(s.energy_knobs.keys())
    knob_keys = sorted(knob_keys)

    X_list = []
    Y_list = []
    prof_list = []
    for (_, _), prof_map in grouped.items():
        base = prof_map.get("BASE")
        if base is None:
            continue
        for name, sample in prof_map.items():
            if name == "BASE":
                continue
            x, y = encode_sample(sample, base, env_vocab, profile_vocab)
            X_list.append(x)
            Y_list.append(y)
            prof_list.append(name)

    if not X_list:
        print("No samples to evaluate.")
        return

    X = torch.from_numpy(np.stack(X_list))
    Y = torch.from_numpy(np.stack(Y_list))

    net = EnergyResponseNet(in_dim=X.shape[1])
    state = torch.load(args.model, map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        preds = net(X).numpy()
    y_true = Y.numpy()

    # Correlations
    corr = {}
    for i, name in enumerate(["delta_mpl", "delta_error", "delta_energy_Wh", "delta_risk"]):
        corr[name] = float(np.corrcoef(preds[:, i], y_true[:, i])[0, 1]) if len(preds) > 1 else 0.0

    # Per-profile metrics
    per_profile = {}
    for prof in set(prof_list):
        idx = [i for i, p in enumerate(prof_list) if p == prof]
        if not idx:
            continue
        per_profile[prof] = {
            "mse": [float(np.mean((preds[idx, j] - y_true[idx, j]) ** 2)) for j in range(4)],
            "mean_pred": [float(np.mean(preds[idx, j])) for j in range(4)],
            "mean_true": [float(np.mean(y_true[idx, j])) for j in range(4)],
        }

    print("Correlations:", corr)
    print("Per-profile MSE (delta_mpl, delta_error, delta_energy_Wh, delta_risk):")
    for prof, vals in per_profile.items():
        print(f"  {prof}: mse={vals['mse']} mean_pred={vals['mean_pred']} mean_true={vals['mean_true']}")

    # Save a small summary
    summary = {"correlations": corr, "per_profile": per_profile}
    with open("results/energy_response_model_eval.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

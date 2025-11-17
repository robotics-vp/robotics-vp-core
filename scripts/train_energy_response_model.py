#!/usr/bin/env python3
"""
Train a simple energy response model on intervention data (analysis-only).
"""
import argparse
import json
import os

import numpy as np
import torch

from src.valuation.energy_response_model import (
    EnergyResponseModel,
    load_energy_interventions,
    build_deltas,
    encode_sample,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interventions", type=str, default="data/energy_interventions.jsonl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--out-model", type=str, default="checkpoints/energy_response_model.pt")
    parser.add_argument("--out-metrics", type=str, default="results/energy_response_model_metrics.json")
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

    if not X_list:
        print("No samples found for training.")
        return

    X = torch.from_numpy(np.stack(X_list))
    Y = torch.from_numpy(np.stack(Y_list))

    # train/val split
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    model = EnergyResponseModel(in_dim=X.shape[1], hidden_dim=args.hidden_dim)
    device = "cpu"
    optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    metrics = {"train_loss": [], "val_loss": [], "per_output_mse": {}}
    for epoch in range(args.epochs):
        model.net.train()
        optimizer.zero_grad()
        pred = model.net(X_train.to(device))
        loss = loss_fn(pred, Y_train.to(device))
        loss.backward()
        optimizer.step()
        model.net.eval()
        with torch.no_grad():
            val_pred = model.net(X_val.to(device))
            val_loss = loss_fn(val_pred, Y_val.to(device))
        metrics["train_loss"].append(float(loss.item()))
        metrics["val_loss"].append(float(val_loss.item()))

    with torch.no_grad():
        full_pred = model.net(X.to(device))
        mse = torch.mean((full_pred - Y.to(device)) ** 2, dim=0).cpu().numpy()
        metrics["per_output_mse"] = {
            "delta_mpl": float(mse[0]),
            "delta_error": float(mse[1]),
            "delta_energy_Wh": float(mse[2]),
            "delta_risk": float(mse[3]),
        }

    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_metrics), exist_ok=True)
    model.save(args.out_model)
    with open(args.out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {args.out_model}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()

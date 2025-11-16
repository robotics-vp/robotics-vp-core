#!/usr/bin/env python3
"""
Supervise EnergyProfilePolicy on interventions data to pick best profile per context.
"""
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

from src.controllers.energy_profile_policy import EnergyProfilePolicy


def load_interventions(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_score(summary, weights):
    mpl = summary.get("mpl_episode", 0.0)
    err = summary.get("error_rate_episode", 0.0)
    energy = summary.get("energy_Wh_per_unit", 0.0)
    return weights["mpl"] * mpl - weights["error"] * err - weights["energy"] * energy


def build_dataset(records, weights):
    contexts = []
    targets = []
    for r in records:
        summary = r["summary"]
        profile = r["profile"]
        ctx = np.array([
            summary.get("mpl_episode", 0.0),
            summary.get("error_rate_episode", 0.0),
            summary.get("energy_Wh_per_unit", 0.0),
            summary.get("energy_Wh_per_hour", 0.0),
        ], dtype=np.float32)
        contexts.append(ctx)
        targets.append(profile)
    return np.stack(contexts), targets


def main():
    parser = argparse.ArgumentParser(description="Train EnergyProfilePolicy from interventions")
    parser.add_argument("--interventions", type=str, default="data/energy_interventions.jsonl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="checkpoints/energy_profile_policy.pt")
    parser.add_argument("--results", type=str, default="results/energy_profile_policy_training.json")
    args = parser.parse_args()

    records = load_interventions(args.interventions)
    weights = {"mpl": 1.0, "error": 1.0, "energy": 0.1}
    x_np, targets = build_dataset(records, weights)
    x = torch.from_numpy(x_np)

    # Map profile names to target vectors (speed, tau_sh, tau_el, tau_wr, tau_gr, safety)
    preset_map = {
        "BASE": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "BOOST": np.array([1.2, 1.2, 1.2, 1.2, 1.1, 1.0], dtype=np.float32),
        "SAVER": np.array([0.8, 0.8, 0.8, 0.8, 0.9, 1.1], dtype=np.float32),
        "SAFE": np.array([0.7, 0.7, 0.7, 0.7, 0.8, 1.2], dtype=np.float32),
    }
    y = torch.from_numpy(np.stack([preset_map[t] for t in targets]))

    model = EnergyProfilePolicy(input_dim=x.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        _, params = model(X_train)
        loss = loss_fn(params, y_train)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            _, val_params = model(X_val)
            val_loss = loss_fn(val_params, y_val)
        print(f"Epoch {epoch+1}/{args.epochs} Loss: {loss.item():.6f} Val: {val_loss.item():.6f}")

    torch.save(model.state_dict(), args.output)
    with open(args.results, "w") as f:
        json.dump({
            "train_loss": float(loss.item()),
            "val_loss": float(val_loss.item())
        }, f, indent=2)
    print(f"Saved policy to {args.output}")


if __name__ == "__main__":
    main()

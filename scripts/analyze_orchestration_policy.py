#!/usr/bin/env python3
"""
Analyze orchestration transformer policy vs heuristic teacher across regimes.
"""
import argparse
import json
import os
import numpy as np
import torch

from src.orchestrator.orchestration_transformer import OrchestrationTransformer, decode_tool
from src.orchestrator.training_dataset import build_training_dataset


def sample_contexts(regime: str, num: int = 16):
    base = build_training_dataset(num_samples=num)
    # Simple regime filters (toy)
    if regime == "high_energy_price":
        for c in base:
            c["context"]["energy_price_kWh"] = 0.5
    elif regime == "low_mpl":
        for c in base:
            c["context"]["mean_delta_mpl"] = -1.0
    elif regime == "high_safety":
        for c in base:
            c["context"]["mean_trust"] = 0.2
    return base


def analyze_regime(model, regime, device):
    data = sample_contexts(regime, num=32)
    tool_counts = {}
    for sample in data:
        ctx_vec = torch.tensor(sample["features"]["context"], dtype=torch.float32, device=device).unsqueeze(0)
        instr_tokens = torch.tensor(sample["features"]["instr_tokens"], dtype=torch.long, device=device).unsqueeze(0)
        logits, arg_vec = model(instr_tokens, ctx_vec)
        tool = decode_tool(logits)
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    return tool_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="results/orchestration_policy_analysis.json")
    args = parser.parse_args()

    device = torch.device("cpu")
    sample_ds = build_training_dataset(num_samples=1)
    ctx_dim = len(sample_ds[0]["features"]["context"])
    vocab_size = max(max(s["features"]["instr_tokens"]) for s in sample_ds) + 2
    model = OrchestrationTransformer(vocab_size=vocab_size, ctx_dim=ctx_dim)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
    model.eval()

    regimes = ["high_energy_price", "low_mpl", "high_safety"]
    summary = {}
    for r in regimes:
        counts = analyze_regime(model, r, device)
        summary[r] = counts
        print(f"Regime {r}: {counts}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

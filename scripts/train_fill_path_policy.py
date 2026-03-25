#!/usr/bin/env python3
"""Train the learned fill-path policy from fill-outcome records.

Usage:
    python3 scripts/train_fill_path_policy.py \\
        --outcome-store data/fill_outcomes.jsonl \\
        --epochs 50 \\
        --save-dir checkpoints/fill_path_policy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.world_model.fill_outcome_store import FillOutcomeStore
from src.world_model.fill_path_policy import train_fill_path_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fill-path policy")
    parser.add_argument(
        "--outcome-store",
        type=str,
        default="data/fill_outcomes.jsonl",
        help="Path to fill-outcome JSONL store",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="checkpoints/fill_path_policy")
    args = parser.parse_args()

    store = FillOutcomeStore(args.outcome_store)
    records = store.load_all()
    if not records:
        print(f"ERROR: No records in {args.outcome_store}")
        sys.exit(1)

    print(f"Training fill-path policy on {len(records)} records...")
    save_path = str(Path(args.save_dir) / "fill_path_policy.pt")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_fill_path_policy(
        records,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        save_path=save_path,
    )

    print(f"Saved checkpoint to {save_path}")
    print(json.dumps(store.summary(), indent=2))


if __name__ == "__main__":
    main()

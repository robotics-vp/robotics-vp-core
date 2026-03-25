#!/usr/bin/env python3
"""Train the learned gap ranker from fill-outcome records.

Usage:
    python3 scripts/train_gap_ranker.py \\
        --outcome-store data/fill_outcomes.jsonl \\
        --epochs 50 \\
        --save-dir checkpoints/gap_ranker
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.world_model.fill_outcome_store import FillOutcomeStore
from src.world_model.gap_ranker import train_gap_ranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the learned gap ranker")
    parser.add_argument(
        "--outcome-store",
        type=str,
        default="data/fill_outcomes.jsonl",
        help="Path to fill-outcome JSONL store",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="checkpoints/gap_ranker")
    args = parser.parse_args()

    store = FillOutcomeStore(args.outcome_store)
    records = store.load_all()
    if not records:
        print(f"ERROR: No records in {args.outcome_store}")
        sys.exit(1)

    print(f"Training gap ranker on {len(records)} records...")
    save_path = str(Path(args.save_dir) / "gap_ranker.pt")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train_gap_ranker(
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

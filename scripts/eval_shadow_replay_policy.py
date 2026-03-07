#!/usr/bin/env python3
"""Evaluate the shadow replay BC policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.learning.replay_policy_trainer import evaluate_replay_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate replay BC policy")
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    args = parser.parse_args()

    result = evaluate_replay_policy(
        dataset_dir=args.dataset_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

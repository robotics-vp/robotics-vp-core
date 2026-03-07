#!/usr/bin/env python3
"""Train the shadow replay BC policy on a canonical replay dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.learning.replay_policy_trainer import train_replay_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train replay BC policy")
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    result = train_replay_policy(
        dataset_dir=args.dataset_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

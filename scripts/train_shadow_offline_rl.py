#!/usr/bin/env python3
"""Train the additive offline RL shadow bridge on canonical replay data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.learning.offline_rl import train_offline_rl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TD3+BC-style shadow offline RL bridge")
    parser.add_argument("--dataset-dir", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()

    result = train_offline_rl(
        dataset_dir=args.dataset_dir,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

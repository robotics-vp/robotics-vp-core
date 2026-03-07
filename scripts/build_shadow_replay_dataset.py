#!/usr/bin/env python3
"""Build a canonical replay dataset from shadow or workcell artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.replay.dataset import ReplayDatasetBuilder
from src.shadow_runtime.control_plane import run_shadow_control_plane


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical replay dataset artifacts")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--shadow-run-dir", type=str, default=None)
    parser.add_argument("--workcell-episode-log", type=str, default=None)
    parser.add_argument("--generate-shadow-run", action="store_true")
    parser.add_argument("--shadow-run-output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=4)
    args = parser.parse_args()

    builder = ReplayDatasetBuilder()
    if args.generate_shadow_run:
        shadow_output = Path(args.shadow_run_output_dir or Path(args.output_dir).parent / "shadow_run")
        run_shadow_control_plane(
            output_dir=shadow_output,
            seed=args.seed,
            episodes=args.episodes,
            objective_profile_id="balanced_contract",
            include_regal=True,
            timestamp_base="2026-01-01T00:00:00+00:00",
        )
        builder.add_shadow_run(shadow_output)
    if args.shadow_run_dir:
        builder.add_shadow_run(args.shadow_run_dir)
    if args.workcell_episode_log:
        builder.add_workcell_episode_log(args.workcell_episode_log)

    bundle = builder.write(args.output_dir)
    print(json.dumps(bundle.to_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

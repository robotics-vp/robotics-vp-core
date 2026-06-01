#!/usr/bin/env python3
"""Check that repo defaults keep Unitree G1 as the primary environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.world_model.humanoid_readiness.g1_primary_environment import (  # noqa: E402
    run_g1_primary_env_hygiene,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/g1_primary_env_hygiene",
    )
    args = parser.parse_args()
    payload = run_g1_primary_env_hygiene(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["blocking_issue_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

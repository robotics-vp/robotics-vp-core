#!/usr/bin/env python3
"""Run the deterministic shadow economic control plane end to end."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shadow_runtime.control_plane import run_shadow_control_plane


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the shadow economic control plane")
    parser.add_argument("--output-dir", type=str, default="artifacts/shadow_econ_control_plane")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--objective-profile", type=str, default="balanced_contract")
    parser.add_argument("--pricing-policy", type=str, default="config/pricing/default.yaml")
    parser.add_argument("--timestamp-base", type=str, default="2026-01-01T00:00:00+00:00")
    parser.add_argument("--disable-regal", action="store_true", help="Disable shadow regal aggregation")
    args = parser.parse_args()

    result = run_shadow_control_plane(
        output_dir=Path(args.output_dir),
        seed=args.seed,
        episodes=args.episodes,
        objective_profile_id=args.objective_profile,
        pricing_policy_path=args.pricing_policy,
        include_regal=not args.disable_regal,
        timestamp_base=args.timestamp_base,
    )

    print(f"[shadow_econ_control_plane] run_id={result.run_id}")
    print(f"[shadow_econ_control_plane] summary={result.artifact_paths['summary_json']}")
    print(f"[shadow_econ_control_plane] report={result.artifact_paths['summary_md']}")


if __name__ == "__main__":
    main()

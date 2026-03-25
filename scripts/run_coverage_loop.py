#!/usr/bin/env python3
"""Run Coverage Loop — standalone CLI for the 9-step evidence cycle.

Usage:
    python scripts/run_coverage_loop.py
    python scripts/run_coverage_loop.py --env drawer_vase dishwashing --write-artifacts
    python scripts/run_coverage_loop.py --replay-dir data/replay --artifact-dir data/coverage

This script executes the full coverage evidence loop:

1. Load replay / runtime learning data from disk
2. Harvest evidence counts from observed task→skill→env-primitive edges
3. Build coverage graph with real evidence
4. Rank missing edges and compile simulation agenda
5. Compile gap-driven diffusion prompts
6. Compute fill-path decisions (real sim / diffusion / synth branch / blocked)
7. Write artifacts to disk (optional)

The output artifacts can then be consumed by:
- ``semantic_simulation.py`` for targeted simulation runs
- ``diffusion_requests.py`` for gap-aware diffusion prompts
- ``collect_local_synthetic_branches.py`` for gap-aware branch selection
"""
import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.orchestrator.coverage_loop import run_coverage_loop


def _load_runtime_rows(replay_dir: str) -> list:
    """Load runtime learning rows from JSONL files in replay_dir."""
    rows = []
    replay_path = Path(replay_dir)
    if not replay_path.exists():
        print(f"  Replay dir {replay_dir} does not exist, using empty corpus")
        return rows

    # Look for .jsonl files
    for jsonl_file in sorted(replay_path.glob("**/*.jsonl")):
        try:
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception as e:
            print(f"  Warning: could not parse {jsonl_file}: {e}")

    # Also look for .json array files
    for json_file in sorted(replay_path.glob("**/*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    rows.extend(data)
                elif isinstance(data, dict) and "rows" in data:
                    rows.extend(data["rows"])
        except Exception:
            pass

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run the coverage evidence loop to identify and prioritise semantic gaps."
    )
    parser.add_argument(
        "--replay-dir", default="data/replay",
        help="Directory containing replay / runtime learning JSONL data",
    )
    parser.add_argument(
        "--artifact-dir", default="data/coverage",
        help="Directory to write output artifacts",
    )
    parser.add_argument(
        "--env", nargs="*", default=None,
        help="Environment IDs to include (default: all registered)",
    )
    parser.add_argument(
        "--economic-weight", type=float, default=1.0,
        help="Weight for economic priority in gap ranking",
    )
    parser.add_argument(
        "--trust-weight", type=float, default=1.0,
        help="Weight for trust priority in gap ranking",
    )
    parser.add_argument(
        "--readiness-weight", type=float, default=1.0,
        help="Weight for readiness in gap ranking",
    )
    parser.add_argument(
        "--sim-limit", type=int, default=10,
        help="Max simulation agenda items",
    )
    parser.add_argument(
        "--diffusion-limit", type=int, default=10,
        help="Max diffusion prompt items",
    )
    parser.add_argument(
        "--write-artifacts", action="store_true", default=True,
        help="Write output artifacts to disk (default: True)",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="Suppress writing artifacts to disk",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just print summary, don't write artifacts",
    )

    args = parser.parse_args()
    write = args.write_artifacts and not args.no_write and not args.dry_run

    print("=" * 60)
    print("Coverage Evidence Loop")
    print("=" * 60)

    # Step 1: Load runtime rows
    print(f"\n1. Loading replay data from {args.replay_dir}...")
    rows = _load_runtime_rows(args.replay_dir)
    print(f"   Loaded {len(rows)} runtime rows")

    # Step 2: Run the coverage loop
    print("\n2. Running coverage loop...")
    result = run_coverage_loop(
        rows,
        env_names=args.env,
        economic_weight=args.economic_weight,
        trust_weight=args.trust_weight,
        readiness_weight=args.readiness_weight,
        sim_agenda_limit=args.sim_limit,
        diffusion_limit=args.diffusion_limit,
        write_artifacts=write,
        artifact_dir=args.artifact_dir,
    )

    # Step 3: Print summary
    summary = result.coverage_summary
    print("\n3. Coverage Summary:")
    print(f"   Total edges:    {summary.get('total_edges', 0)}")
    print(f"   Covered edges:  {summary.get('covered_edges', 0)}")
    print(f"   Missing edges:  {summary.get('missing_edges', 0)}")
    print(f"   Coverage ratio: {summary.get('coverage_ratio', 0):.2%}")

    harvest = result.evidence_harvest
    print("\n4. Evidence Harvest:")
    print(f"   Rows processed:    {harvest.rows_processed}")
    print(f"   Edges discovered:  {harvest.edges_discovered}")

    print("\n5. Outputs:")
    print(f"   Sim agenda items:    {len(result.simulation_agenda)}")
    print(f"   Diffusion prompts:   {len(result.diffusion_prompts)}")
    print(f"   Fill decisions:      {len(result.fill_decisions)}")

    if result.fill_decisions:
        methods = {}
        for d in result.fill_decisions:
            m = d.get("fill_method", "unknown")
            methods[m] = methods.get(m, 0) + 1
        print(f"   Fill method breakdown: {methods}")

    if write:
        paths = result.write_artifacts(args.artifact_dir)
        print(f"\n6. Artifacts written to {args.artifact_dir}:")
        for name, path in paths.items():
            print(f"   {name}: {path}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()

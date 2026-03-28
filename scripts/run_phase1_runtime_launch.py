#!/usr/bin/env python3
"""Prepare or execute a Phase-1 runtime launch from WM-owned artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.world_model.sim_synth_physics.runtime_launch import (
    build_backend_runtime_launch_receipt,
    execute_backend_runtime_launch,
    load_runtime_artifacts,
)
from src.world_model.sim_synth_physics.runtime_outcomes import (
    build_backend_runtime_outcome_receipt,
    build_backend_runtime_output_contract,
    harvest_backend_runtime_outcomes,
)


def _default_artifact_paths(runtime_root: Path) -> tuple[Path, Path]:
    return (
        runtime_root / "backend_runtime_bundle.json",
        runtime_root / "backend_launch_spec.json",
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, help="Directory containing runtime bundle/spec artifacts.")
    parser.add_argument("--runtime-bundle", type=Path, help="Path to backend_runtime_bundle.json.")
    parser.add_argument("--launch-spec", type=Path, help="Path to backend_launch_spec.json.")
    parser.add_argument("--execute", action="store_true", help="Actually execute the launch command.")
    parser.add_argument("--cwd", type=Path, help="Override working directory for the launch command.")
    parser.add_argument("--output", type=Path, help="Write the launch report JSON to this path.")
    parser.add_argument(
        "--receipt-output",
        type=Path,
        help="Optionally write a pure backend_runtime_launch_receipt_v1 artifact.",
    )
    parser.add_argument(
        "--outcome-output",
        type=Path,
        help="Optionally write a pure backend_runtime_outcome_receipt_v1 artifact.",
    )
    parser.add_argument(
        "--harvest-outcomes",
        action="store_true",
        help="Harvest upstream runtime outputs using the launch bundle/spec conventions.",
    )
    args = parser.parse_args(argv)

    runtime_bundle_path = args.runtime_bundle
    launch_spec_path = args.launch_spec
    if args.runtime_root is not None:
        default_bundle_path, default_launch_path = _default_artifact_paths(args.runtime_root)
        runtime_bundle_path = runtime_bundle_path or default_bundle_path
        launch_spec_path = launch_spec_path or default_launch_path
    if runtime_bundle_path is None or launch_spec_path is None:
        raise SystemExit("Provide --runtime-root or both --runtime-bundle and --launch-spec.")
    runtime_bundle, launch_spec = load_runtime_artifacts(
        runtime_bundle_path=runtime_bundle_path,
        launch_spec_path=launch_spec_path,
    )
    result = execute_backend_runtime_launch(
        runtime_bundle,
        launch_spec,
        execute=bool(args.execute),
        cwd=args.cwd,
    )
    receipt = build_backend_runtime_launch_receipt(runtime_bundle, launch_spec, result)
    outcome_receipt = None
    output_summary = None
    if args.harvest_outcomes:
        output_contract = build_backend_runtime_output_contract(runtime_bundle, launch_spec)
        output_summary = harvest_backend_runtime_outcomes(
            output_contract,
            executed=bool(receipt.executed),
        )
        outcome_receipt = build_backend_runtime_outcome_receipt(
            runtime_bundle=runtime_bundle,
            launch_receipt=receipt,
            output_summary=output_summary,
        )
    payload = {
        "runtime_bundle_path": str(Path(runtime_bundle_path).resolve()),
        "launch_spec_path": str(Path(launch_spec_path).resolve()),
        "result": result,
        "receipt": receipt.to_dict(),
        "outcome_receipt": None if outcome_receipt is None else outcome_receipt.to_dict(),
        "output_summary": output_summary,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.receipt_output is not None:
        args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_output.write_text(
            json.dumps(receipt.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.outcome_output is not None and outcome_receipt is not None:
        args.outcome_output.parent.mkdir(parents=True, exist_ok=True)
        args.outcome_output.write_text(
            json.dumps(outcome_receipt.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

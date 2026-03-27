#!/usr/bin/env python3
"""Inspect Phase-1 runtime roots, layouts, and policy contracts for sim/synth backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.world_model.sim_synth_physics.runtime_layouts import (
    describe_holosoma_policy_contract,
    describe_holosoma_runtime_layouts,
    describe_isaac_policy_contract,
    describe_isaac_runtime_layouts,
)
from src.world_model.sim_synth_physics.runtime_targets import (
    describe_holosoma_runtime_targets,
    describe_isaac_runtime_targets,
)


def _load_mapping(path: str | None) -> Optional[Mapping[str, Any]]:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Expected mapping payload in {path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Phase-1 backend runtime layouts")
    parser.add_argument("--embodiment-context", help="Optional JSON mapping with embodiment context")
    parser.add_argument(
        "--output-path",
        default="artifacts/sim_synth_runtime_layout_scan.json",
        help="Path to write the scan summary",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    args = parse_args(argv)
    embodiment_context = _load_mapping(args.embodiment_context) or {}
    summary = {
        "version": "phase1_runtime_layout_scan_v1",
        "isaac_runtime_targets": describe_isaac_runtime_targets(embodiment_context),
        "isaac_runtime_layouts": describe_isaac_runtime_layouts(embodiment_context),
        "isaac_policy_contract": describe_isaac_policy_contract(embodiment_context),
        "holosoma_runtime_targets": describe_holosoma_runtime_targets(embodiment_context),
        "holosoma_runtime_layouts": describe_holosoma_runtime_layouts(embodiment_context),
        "holosoma_policy_contract": describe_holosoma_policy_contract(embodiment_context),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {"output_path": str(output_path.resolve())}


if __name__ == "__main__":
    main()

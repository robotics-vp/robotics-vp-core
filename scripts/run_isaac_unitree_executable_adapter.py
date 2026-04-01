#!/usr/bin/env python3
"""Prepare or execute the Isaac/Unitree executable-adapter request from WM artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.world_model.sim_synth_physics.adapters import (
    build_isaac_unitree_adapter_receipt,
    build_isaac_unitree_adapter_realization,
    finalize_isaac_unitree_adapter_execution,
    prepare_isaac_unitree_adapter_execution,
)
from src.world_model.sim_synth_physics.runtime_launch import (
    build_backend_runtime_launch_receipt,
    execute_backend_runtime_launch,
    load_runtime_artifacts,
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
    parser.add_argument("--output", type=Path, help="Write the adapter report JSON to this path.")
    args = parser.parse_args(argv)

    runtime_bundle_path = args.runtime_bundle
    launch_spec_path = args.launch_spec
    if args.runtime_root is not None:
        runtime_bundle_path, launch_spec_path = _default_artifact_paths(args.runtime_root)
    if runtime_bundle_path is None or launch_spec_path is None:
        raise SystemExit("Provide --runtime-root or both --runtime-bundle and --launch-spec.")

    runtime_bundle, launch_spec = load_runtime_artifacts(
        runtime_bundle_path=runtime_bundle_path,
        launch_spec_path=launch_spec_path,
    )
    if str(runtime_bundle.get("backend", "") or "") != "isaac":
        raise SystemExit("Isaac/Unitree executable adapter only supports backend='isaac'.")
    executable_adapter_request = (
        launch_spec.get("executable_adapter_request")
        or runtime_bundle.get("executable_adapter_request")
        or {}
    )
    executable_adapter_consumer = (
        launch_spec.get("executable_adapter_consumer")
        or runtime_bundle.get("executable_adapter_consumer")
        or {}
    )
    if not executable_adapter_request:
        raise SystemExit("Runtime artifacts do not include an executable_adapter_request.")
    if not executable_adapter_consumer:
        raise SystemExit("Runtime artifacts do not include an executable_adapter_consumer.")

    adapter_execution = prepare_isaac_unitree_adapter_execution(
        executable_adapter_request,
        executable_adapter_consumer,
    )
    adapter_realization = build_isaac_unitree_adapter_realization(
        executable_adapter_request=executable_adapter_request,
        executable_adapter_consumer=executable_adapter_consumer,
        adapter_execution=adapter_execution,
        runtime_bundle=runtime_bundle,
        launch_spec=launch_spec,
    )
    result = execute_backend_runtime_launch(
        runtime_bundle,
        launch_spec,
        execute=bool(args.execute),
        cwd=args.cwd,
    )
    adapter_execution = finalize_isaac_unitree_adapter_execution(
        adapter_execution,
        launch_result=result,
    )
    adapter_realization = build_isaac_unitree_adapter_realization(
        executable_adapter_request=executable_adapter_request,
        executable_adapter_consumer=executable_adapter_consumer,
        adapter_execution=adapter_execution,
        runtime_bundle=runtime_bundle,
        launch_spec=launch_spec,
    )
    adapter_receipt = build_isaac_unitree_adapter_receipt(
        adapter_execution,
        realization=adapter_realization,
    )
    receipt = build_backend_runtime_launch_receipt(runtime_bundle, launch_spec, result)
    payload = {
        "runtime_bundle_path": str(Path(runtime_bundle_path).resolve()),
        "launch_spec_path": str(Path(launch_spec_path).resolve()),
        "executable_adapter_request": executable_adapter_request,
        "executable_adapter_consumer": executable_adapter_consumer,
        "adapter_execution": adapter_execution,
        "adapter_realization": adapter_realization,
        "adapter_receipt": adapter_receipt.to_dict(),
        "result": result,
        "receipt": receipt.to_dict(),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

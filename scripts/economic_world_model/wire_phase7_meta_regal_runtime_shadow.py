#!/usr/bin/env python3
"""Run Phase 7 shadow-only runtime/event-spine wiring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from scripts.economic_world_model.prepare_phase7_meta_regal_control_scaffold import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE7_SCAFFOLD_DIR,
)
from scripts.economic_world_model.prepare_phase7_meta_regal_control_scaffold import (  # noqa: E402
    run_prepare_phase7_meta_regal_control_scaffold,
)
from src.shadow_runtime.control_plane import run_shadow_control_plane  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase7_meta_regal_shadow_runtime"
)


def _required_scaffold_paths(scaffold_dir: Path) -> dict[str, Path]:
    return {
        "report": scaffold_dir / "phase7_meta_regal_control_scaffold_report_v1.json",
        "control_fields": scaffold_dir / "phase7_control_field_slots_v1.jsonl",
        "conflict_receipts": scaffold_dir
        / "phase7_conflict_override_receipts_v1.jsonl",
    }


def _resolve_scaffold(
    *,
    scaffold_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = _required_scaffold_paths(scaffold_dir)
    if all(path.exists() for path in paths.values()):
        return paths
    if not run_dependencies_if_missing:
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Missing Phase 7 scaffold runtime inputs: " + ", ".join(missing)
        )
    run_prepare_phase7_meta_regal_control_scaffold(
        output_dir=scaffold_dir,
        run_dependencies_if_missing=True,
    )
    if not all(path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Phase 7 scaffold builder did not materialize: " + ", ".join(missing)
        )
    return paths


def run_wire_phase7_meta_regal_runtime_shadow(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase7_scaffold_dir: str | Path = DEFAULT_PHASE7_SCAFFOLD_DIR,
    seed: int = 42,
    episodes: int = 2,
    objective_profile_id: str = "balanced_contract",
    timestamp_base: str | None = "2026-05-25T00:00:00+00:00",
    run_id: str | None = None,
    phase7_signal_adapter_dir: str | Path | None = None,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    scaffold_dir = Path(phase7_scaffold_dir)
    _resolve_scaffold(
        scaffold_dir=scaffold_dir,
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    result = run_shadow_control_plane(
        output_dir=output,
        seed=seed,
        episodes=episodes,
        objective_profile_id=objective_profile_id,
        include_regal=True,
        timestamp_base=timestamp_base,
        run_id=run_id,
        include_phase7_meta_regal_shadow=True,
        phase7_scaffold_dir=scaffold_dir,
        phase7_signal_adapter_dir=phase7_signal_adapter_dir,
    )
    payload: Mapping[str, Any] = result.summary
    print(json.dumps(payload, indent=2, sort_keys=True))
    return dict(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--phase7-scaffold-dir",
        default=str(DEFAULT_PHASE7_SCAFFOLD_DIR),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--objective-profile-id", default="balanced_contract")
    parser.add_argument("--timestamp-base", default="2026-05-25T00:00:00+00:00")
    parser.add_argument("--run-id")
    parser.add_argument("--phase7-signal-adapter-dir")
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_wire_phase7_meta_regal_runtime_shadow(
        output_dir=args.output_dir,
        phase7_scaffold_dir=args.phase7_scaffold_dir,
        seed=args.seed,
        episodes=args.episodes,
        objective_profile_id=args.objective_profile_id,
        timestamp_base=args.timestamp_base,
        run_id=args.run_id,
        phase7_signal_adapter_dir=args.phase7_signal_adapter_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    phase7 = payload["phase7_meta_regal_shadow"]
    return (
        0
        if phase7["local_shadow_runtime_wiring_complete"]
        and phase7["shadow_event_spine_wiring_executed"]
        and phase7["decision_ledger_wiring_executed"]
        and not phase7["phase7_authority_granted"]
        and not phase7["live_dispatch_allowed"]
        and not phase7["hard_veto_dispatch"]
        and not phase7["training_executed"]
        and not phase7["weights_written"]
        and not phase7["provider_executed"]
        and not phase7["hardware_executed"]
        and not phase7["unitree_sim_runtime_executed"]
        and not phase7["live_policy_control"]
        and not phase7["reward_math_mutation"]
        and not phase7["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

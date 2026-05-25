#!/usr/bin/env python3
"""Adapt lower-WM receipts into Phase 7 governance-node shadow signals."""

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

from scripts.economic_world_model.evaluate_phase7_meta_governance_shadow import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE7_EVAL_DIR,
)
from scripts.economic_world_model.evaluate_phase7_meta_governance_shadow import (  # noqa: E402
    run_evaluate_phase7_meta_governance_shadow,
)
from scripts.economic_world_model.prepare_phase7_meta_regal_control_scaffold import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE7_SCAFFOLD_DIR,
)
from scripts.economic_world_model.prepare_phase7_meta_regal_control_scaffold import (  # noqa: E402
    run_prepare_phase7_meta_regal_control_scaffold,
)
from scripts.economic_world_model.wire_phase7_meta_regal_runtime_shadow import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE7_RUNTIME_DIR,
)
from scripts.economic_world_model.wire_phase7_meta_regal_runtime_shadow import (  # noqa: E402
    run_wire_phase7_meta_regal_runtime_shadow,
)
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase7_governance_signal_adapters,
    load_phase7_signal_adapter_scaffold_surfaces,
    save_phase7_governance_signal_adapters,
)
from src.world_model.humanoid_readiness.common import load_json  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase7_governance_signal_adapters"
)
DEFAULT_LOWER_ARTIFACT_ROOT = Path("artifacts/economic_world_model")


def _required_scaffold_paths(scaffold_dir: Path) -> dict[str, Path]:
    return {
        "report": scaffold_dir / "phase7_meta_regal_control_scaffold_report_v1.json",
        "surfaces": scaffold_dir / "phase7_governance_node_surfaces_v1.jsonl",
    }


def _required_runtime_paths(runtime_dir: Path) -> dict[str, Path]:
    return {
        "summary": runtime_dir / "summary.json",
        "field_receipts": runtime_dir / "phase7_control_field_runtime_receipts.jsonl",
        "conflict_receipts": runtime_dir
        / "phase7_conflict_runtime_join_receipts.jsonl",
    }


def _required_eval_paths(eval_dir: Path) -> dict[str, Path]:
    return {
        "report": eval_dir / "phase7_meta_governance_evaluation_report_v1.json",
        "outcome_rows": eval_dir / "phase7_outcome_join_rows_v1.jsonl",
        "field_evals": eval_dir / "phase7_control_field_eval_reports_v1.jsonl",
    }


def _resolve_inputs(
    *,
    scaffold_dir: Path,
    runtime_dir: Path,
    eval_dir: Path,
    run_dependencies_if_missing: bool,
) -> None:
    scaffold_paths = _required_scaffold_paths(scaffold_dir)
    if not all(path.exists() for path in scaffold_paths.values()):
        if not run_dependencies_if_missing:
            missing = [str(path) for path in scaffold_paths.values() if not path.exists()]
            raise FileNotFoundError(
                "Missing Phase 7 signal-adapter scaffold inputs: "
                + ", ".join(missing)
            )
        run_prepare_phase7_meta_regal_control_scaffold(
            output_dir=scaffold_dir,
            run_dependencies_if_missing=True,
        )

    runtime_paths = _required_runtime_paths(runtime_dir)
    if not all(path.exists() for path in runtime_paths.values()):
        if not run_dependencies_if_missing:
            missing = [str(path) for path in runtime_paths.values() if not path.exists()]
            raise FileNotFoundError(
                "Missing Phase 7 signal-adapter runtime inputs: "
                + ", ".join(missing)
            )
        run_wire_phase7_meta_regal_runtime_shadow(
            output_dir=runtime_dir,
            phase7_scaffold_dir=scaffold_dir,
            run_dependencies_if_missing=True,
        )

    eval_paths = _required_eval_paths(eval_dir)
    if not all(path.exists() for path in eval_paths.values()):
        if not run_dependencies_if_missing:
            missing = [str(path) for path in eval_paths.values() if not path.exists()]
            raise FileNotFoundError(
                "Missing Phase 7 signal-adapter eval inputs: " + ", ".join(missing)
            )
        run_evaluate_phase7_meta_governance_shadow(
            output_dir=eval_dir,
            phase7_runtime_dir=runtime_dir,
            run_dependencies_if_missing=True,
        )


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 7 Governance Signal Adapters",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Governance-node surfaces: `{payload['governance_node_surface_count']}`",
        f"- Signal adapters: `{payload['adapter_count']}`",
        f"- Signal receipts: `{payload['signal_receipt_count']}`",
        f"- Source artifacts: `{payload['source_artifact_count']}`",
        f"- Missing source artifacts: `{payload['missing_source_artifact_count']}`",
        "- Lower-WM receipt-backed nodes: "
        f"`{payload['lower_wm_receipt_backed_node_count']}`",
        "- All eight nodes signal-backed: "
        f"`{str(payload['all_eight_nodes_signal_backed']).lower()}`",
        "- Shadow runtime feed ready: "
        f"`{str(payload['shadow_runtime_feed_ready']).lower()}`",
        "",
        "## Boundary",
        "",
        "These adapters read existing local lower-WM receipts and summarize",
        "them as Phase 7 node signals. They do not train, write weights,",
        "dispatch live actions, execute hard vetoes, mutate reward math,",
        "promote outputs, replace lower WMs, or claim provider/hardware",
        "runtime authority.",
        "",
        "## Remaining Blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["remaining_blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_adapt_phase7_governance_node_signals(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase7_scaffold_dir: str | Path = DEFAULT_PHASE7_SCAFFOLD_DIR,
    lower_artifact_root: str | Path = DEFAULT_LOWER_ARTIFACT_ROOT,
    phase7_runtime_dir: str | Path = DEFAULT_PHASE7_RUNTIME_DIR,
    phase7_eval_dir: str | Path = DEFAULT_PHASE7_EVAL_DIR,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scaffold_dir = Path(phase7_scaffold_dir)
    runtime_dir = Path(phase7_runtime_dir)
    eval_dir = Path(phase7_eval_dir)
    _resolve_inputs(
        scaffold_dir=scaffold_dir,
        runtime_dir=runtime_dir,
        eval_dir=eval_dir,
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    scaffold_report = load_json(
        scaffold_dir / "phase7_meta_regal_control_scaffold_report_v1.json"
    )
    surfaces = load_phase7_signal_adapter_scaffold_surfaces(scaffold_dir)
    artifact_refs = {
        "phase7_scaffold_dir": str(scaffold_dir),
        "phase7_runtime_dir": str(runtime_dir),
        "phase7_eval_dir": str(eval_dir),
        "lower_artifact_root": str(lower_artifact_root),
        "report_path": str(
            output / "phase7_governance_signal_adapter_report_v1.json"
        ),
        "adapters_path": str(
            output / "phase7_governance_node_signal_adapters_v1.jsonl"
        ),
        "signal_receipts_path": str(
            output / "phase7_governance_node_signal_receipts_v1.jsonl"
        ),
        "markdown_path": str(
            output / "phase7_governance_signal_adapter_report_v1.md"
        ),
    }
    report, adapters, receipts = build_phase7_governance_signal_adapters(
        phase7_scaffold_report_id=str(scaffold_report.get("report_id", "")),
        governance_node_surfaces=surfaces,
        lower_artifact_root=lower_artifact_root,
        artifact_refs=artifact_refs,
    )
    saved_refs = save_phase7_governance_signal_adapters(
        output, report, adapters, receipts
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved_refs}
    _write_markdown(
        output / "phase7_governance_signal_adapter_report_v1.md",
        payload,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--phase7-scaffold-dir",
        default=str(DEFAULT_PHASE7_SCAFFOLD_DIR),
    )
    parser.add_argument(
        "--lower-artifact-root",
        default=str(DEFAULT_LOWER_ARTIFACT_ROOT),
    )
    parser.add_argument("--phase7-runtime-dir", default=str(DEFAULT_PHASE7_RUNTIME_DIR))
    parser.add_argument("--phase7-eval-dir", default=str(DEFAULT_PHASE7_EVAL_DIR))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_adapt_phase7_governance_node_signals(
        output_dir=args.output_dir,
        phase7_scaffold_dir=args.phase7_scaffold_dir,
        lower_artifact_root=args.lower_artifact_root,
        phase7_runtime_dir=args.phase7_runtime_dir,
        phase7_eval_dir=args.phase7_eval_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_signal_adapter_complete"]
        and payload["shadow_runtime_feed_ready"]
        and payload["all_eight_nodes_signal_backed"]
        and not payload["phase7_authority_granted"]
        and not payload["live_dispatch_allowed"]
        and not payload["hard_veto_dispatch"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["provider_executed"]
        and not payload["hardware_executed"]
        and not payload["unitree_sim_runtime_executed"]
        and not payload["live_policy_control"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        and not any(payload["denied_gates"].values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

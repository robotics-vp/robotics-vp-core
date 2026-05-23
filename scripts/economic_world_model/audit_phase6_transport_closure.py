#!/usr/bin/env python3
"""Audit local Phase-6 transport structural closure."""

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

from scripts.economic_world_model.run_phase6_transport_advisory_runtime import (  # noqa: E402
    run_phase6_transport_advisory_runtime,
)
from src.world_model.transport import (  # noqa: E402
    build_wm_transport_phase6_closure_audit,
    load_wm_transport_advisory_runtime_report,
    load_wm_transport_loss_ledger,
    load_wm_transport_neural_architecture_manifest,
    load_wm_transport_phase6_scaffold_report,
    load_wm_transport_trainer_scaffold_manifest,
    save_wm_transport_phase6_closure_audit,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase6_transport_closure_audit"
)
DEFAULT_SCAFFOLD_DIR = Path("artifacts/economic_world_model/phase6_transport_scaffold")
DEFAULT_NEURAL_DIR = Path(
    "artifacts/economic_world_model/phase6_transport_neural_manifest"
)
DEFAULT_TRAINER_DIR = Path(
    "artifacts/economic_world_model/phase6_transport_trainer_scaffold"
)
DEFAULT_RUNTIME_DIR = Path(
    "artifacts/economic_world_model/phase6_transport_advisory_runtime"
)


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6 Transport Closure Audit",
        "",
        f"- Audit: `{payload['audit_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Local Phase 6 structurally closed: `{str(payload['local_phase6_structurally_closed']).lower()}`",
        f"- Missing local runtime contracts: `{len(payload['missing_local_runtime_contracts'])}`",
        f"- Remaining evidence blockers: `{len(payload['remaining_evidence_blockers'])}`",
        "",
        "## Local surfaces closed",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["closed_local_surfaces"])
    lines.extend(
        [
            "",
            "## Remaining blockers",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in payload["remaining_evidence_blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This audit closes local Phase-6 structure only. It confirms that the",
            "remaining blockers are corpus density, GPU training, topology/latency",
            "benchmarks, provider/hardware evidence, and promotion-grade downstream",
            "benchmarks. It does not claim training, provider execution, hardware",
            "execution, live control, reward mutation, or promotion.",
            "",
            "## Artifact refs",
            "",
        ]
    )
    for key, value in sorted(dict(payload.get("artifact_refs", {}) or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_inputs(
    *,
    scaffold_dir: Path,
    neural_dir: Path,
    trainer_dir: Path,
    runtime_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = {
        "scaffold_report": scaffold_dir / "wm_transport_phase6_scaffold_report_v1.json",
        "neural_manifest": neural_dir
        / "wm_transport_neural_architecture_manifest_v1.json",
        "loss_ledger": neural_dir / "wm_transport_loss_ledger_v1.json",
        "trainer_manifest": trainer_dir
        / "wm_transport_trainer_scaffold_manifest_v1.json",
        "runtime_report": runtime_dir
        / "wm_transport_advisory_runtime_report_v1.json",
    }
    if all(path.exists() for path in paths.values()):
        return paths
    if not run_dependencies_if_missing:
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Missing Phase-6 closure audit inputs: " + ", ".join(missing)
        )
    run_phase6_transport_advisory_runtime(
        output_dir=runtime_dir,
        scaffold_dir=scaffold_dir,
        neural_dir=neural_dir,
        trainer_dir=trainer_dir,
        run_dependencies_if_missing=True,
    )
    if not all(path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Phase-6 closure dependency builders did not materialize: "
            + ", ".join(missing)
        )
    return paths


def run_audit_phase6_transport_closure(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    scaffold_dir: str | Path = DEFAULT_SCAFFOLD_DIR,
    neural_dir: str | Path = DEFAULT_NEURAL_DIR,
    trainer_dir: str | Path = DEFAULT_TRAINER_DIR,
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_paths = _resolve_inputs(
        scaffold_dir=Path(scaffold_dir),
        neural_dir=Path(neural_dir),
        trainer_dir=Path(trainer_dir),
        runtime_dir=Path(runtime_dir),
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    report_path = output / "wm_transport_phase6_closure_audit_v1.json"
    markdown_path = output / "wm_transport_phase6_closure_audit_v1.md"
    artifact_refs = {
        "scaffold_report_path": str(input_paths["scaffold_report"]),
        "neural_manifest_path": str(input_paths["neural_manifest"]),
        "loss_ledger_path": str(input_paths["loss_ledger"]),
        "trainer_manifest_path": str(input_paths["trainer_manifest"]),
        "runtime_report_path": str(input_paths["runtime_report"]),
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
    }
    report = build_wm_transport_phase6_closure_audit(
        scaffold_report=load_wm_transport_phase6_scaffold_report(
            input_paths["scaffold_report"]
        ),
        neural_manifest=load_wm_transport_neural_architecture_manifest(
            input_paths["neural_manifest"]
        ),
        loss_ledger=load_wm_transport_loss_ledger(input_paths["loss_ledger"]),
        trainer_manifest=load_wm_transport_trainer_scaffold_manifest(
            input_paths["trainer_manifest"]
        ),
        advisory_runtime_report=load_wm_transport_advisory_runtime_report(
            input_paths["runtime_report"]
        ),
        artifact_refs=artifact_refs,
    )
    save_wm_transport_phase6_closure_audit(report_path, report)
    payload = report.to_dict()
    _write_markdown(markdown_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scaffold-dir", default=str(DEFAULT_SCAFFOLD_DIR))
    parser.add_argument("--neural-dir", default=str(DEFAULT_NEURAL_DIR))
    parser.add_argument("--trainer-dir", default=str(DEFAULT_TRAINER_DIR))
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_audit_phase6_transport_closure(
        output_dir=args.output_dir,
        scaffold_dir=args.scaffold_dir,
        neural_dir=args.neural_dir,
        trainer_dir=args.trainer_dir,
        runtime_dir=args.runtime_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_phase6_structurally_closed"]
        and not payload["missing_local_runtime_contracts"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["provider_executed"]
        and not payload["hardware_executed"]
        and not payload["live_policy_control"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

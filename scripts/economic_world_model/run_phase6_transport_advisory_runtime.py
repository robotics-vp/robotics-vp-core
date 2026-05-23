#!/usr/bin/env python3
"""Run local Phase-6.4 WM transport advisory runtime scaffolding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from scripts.economic_world_model.build_phase6_transport_neural_manifest import (  # noqa: E402
    run_build_phase6_transport_neural_manifest,
)
from scripts.economic_world_model.prepare_phase6_transport_scaffold import (  # noqa: E402
    run_prepare_phase6_transport_scaffold,
)
from scripts.train_wm_transport_bridge_v0 import (  # noqa: E402
    run_train_wm_transport_bridge_v0_scaffold,
)
from src.world_model.economic_world_model.shadow_outcomes import (  # noqa: E402
    load_economic_wm_shadow_outcome_receipts,
)
from src.world_model.transport import (  # noqa: E402
    build_wm_transport_advisory_runtime,
    load_wm_transport_bridge_contracts,
    load_wm_transport_neural_architecture_manifest,
    load_wm_transport_roundtrip_receipts,
    load_wm_transport_trainer_scaffold_manifest,
    save_wm_transport_advisory_runtime,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase6_transport_advisory_runtime"
)
DEFAULT_SCAFFOLD_DIR = Path("artifacts/economic_world_model/phase6_transport_scaffold")
DEFAULT_NEURAL_DIR = Path(
    "artifacts/economic_world_model/phase6_transport_neural_manifest"
)
DEFAULT_TRAINER_DIR = Path(
    "artifacts/economic_world_model/phase6_transport_trainer_scaffold"
)
DEFAULT_SHADOW_OUTCOMES = Path(
    "artifacts/economic_world_model/economic_wm_shadow_outcome_loop/"
    "economic_wm_shadow_outcome_receipts_v1.jsonl"
)


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6.4 Transport Advisory Runtime Artifact",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Proposals: `{payload['proposal_count']}`",
        f"- Invocations: `{payload['invocation_count']}`",
        f"- Receipts: `{payload['receipt_count']}`",
        f"- Decomposed eval reports: `{payload['eval_report_count']}`",
        f"- Shadow join slots: `{payload['shadow_join_slot_count']}`",
        f"- Joined shadow outcomes: `{payload['joined_shadow_outcome_count']}`",
        f"- Ready for decomposed eval: `{str(payload['ready_for_decomposed_eval']).lower()}`",
        f"- Ready for training: `{str(payload['ready_for_training']).lower()}`",
        f"- Training executed: `{str(payload['training_executed']).lower()}`",
        f"- Weights written: `{str(payload['weights_written']).lower()}`",
        f"- Promotion eligible: `{str(payload['promotion_eligible']).lower()}`",
        "",
        "## Boundary",
        "",
        "This artifact emits advisory transport proposals, invocations, receipts,",
        "decomposed bridge/receiver/downstream eval reports, and shadow-outcome join",
        "slots. It does not train, write weights, invoke providers, execute hardware,",
        "grant live policy control, mutate reward math, bypass target receivers, or",
        "promote transport outputs.",
        "",
        "## Artifact refs",
        "",
    ]
    for key, value in sorted(dict(payload.get("artifact_refs", {}) or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_inputs(
    *,
    scaffold_dir: Path,
    neural_dir: Path,
    trainer_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = {
        "contracts": scaffold_dir / "wm_transport_bridge_contracts_v1.jsonl",
        "roundtrip_receipts": scaffold_dir
        / "wm_transport_roundtrip_receipts_v1.jsonl",
        "neural_manifest": neural_dir
        / "wm_transport_neural_architecture_manifest_v1.json",
        "trainer_manifest": trainer_dir
        / "wm_transport_trainer_scaffold_manifest_v1.json",
    }
    if all(path.exists() for path in paths.values()):
        return paths
    if not run_dependencies_if_missing:
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Missing Phase-6.4 advisory runtime inputs: " + ", ".join(missing)
        )
    run_prepare_phase6_transport_scaffold(
        output_dir=scaffold_dir,
        run_dependencies_if_missing=True,
    )
    run_build_phase6_transport_neural_manifest(
        output_dir=neural_dir,
        scaffold_dir=scaffold_dir,
        run_dependencies_if_missing=True,
    )
    run_train_wm_transport_bridge_v0_scaffold(
        output_dir=trainer_dir,
        neural_dir=neural_dir,
        scaffold_dir=scaffold_dir,
        run_dependencies_if_missing=True,
    )
    if not all(path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Phase-6.4 dependency builders did not materialize: "
            + ", ".join(missing)
        )
    return paths


def _load_shadow_outcomes(path: Optional[str | Path]) -> list[Any]:
    if not path:
        return []
    target = Path(path)
    if not target.exists():
        return []
    return list(load_economic_wm_shadow_outcome_receipts(target))


def run_phase6_transport_advisory_runtime(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    scaffold_dir: str | Path = DEFAULT_SCAFFOLD_DIR,
    neural_dir: str | Path = DEFAULT_NEURAL_DIR,
    trainer_dir: str | Path = DEFAULT_TRAINER_DIR,
    shadow_outcomes_path: Optional[str | Path] = DEFAULT_SHADOW_OUTCOMES,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_paths = _resolve_inputs(
        scaffold_dir=Path(scaffold_dir),
        neural_dir=Path(neural_dir),
        trainer_dir=Path(trainer_dir),
        run_dependencies_if_missing=run_dependencies_if_missing,
    )

    proposals_path = output / "wm_transport_advisory_proposals_v1.jsonl"
    invocations_path = output / "wm_transport_advisory_invocations_v1.jsonl"
    receipts_path = output / "wm_transport_advisory_receipts_v1.jsonl"
    eval_reports_path = output / "wm_transport_decomposed_eval_reports_v1.jsonl"
    report_path = output / "wm_transport_advisory_runtime_report_v1.json"
    markdown_path = output / "wm_transport_advisory_runtime_v1.md"
    artifact_refs = {
        "contracts_path": str(input_paths["contracts"]),
        "roundtrip_receipts_path": str(input_paths["roundtrip_receipts"]),
        "neural_manifest_path": str(input_paths["neural_manifest"]),
        "trainer_manifest_path": str(input_paths["trainer_manifest"]),
        "shadow_outcomes_path": str(shadow_outcomes_path or ""),
        "proposals_path": str(proposals_path),
        "invocations_path": str(invocations_path),
        "receipts_path": str(receipts_path),
        "eval_reports_path": str(eval_reports_path),
        "report_path": str(report_path),
        "markdown_path": str(markdown_path),
    }

    contracts = load_wm_transport_bridge_contracts(input_paths["contracts"])
    roundtrip_receipts = load_wm_transport_roundtrip_receipts(
        input_paths["roundtrip_receipts"]
    )
    neural_manifest = load_wm_transport_neural_architecture_manifest(
        input_paths["neural_manifest"]
    )
    trainer_manifest = load_wm_transport_trainer_scaffold_manifest(
        input_paths["trainer_manifest"]
    )
    shadow_outcomes = _load_shadow_outcomes(shadow_outcomes_path)

    report, proposals, invocations, receipts, eval_reports = (
        build_wm_transport_advisory_runtime(
            contracts=contracts,
            roundtrip_receipts=roundtrip_receipts,
            neural_manifest=neural_manifest,
            trainer_manifest=trainer_manifest,
            shadow_outcome_receipts=shadow_outcomes,
            artifact_refs=artifact_refs,
        )
    )
    save_wm_transport_advisory_runtime(
        report_path=report_path,
        report=report,
        proposals_path=proposals_path,
        proposals=proposals,
        invocations_path=invocations_path,
        invocations=invocations,
        receipts_path=receipts_path,
        receipts=receipts,
        eval_reports_path=eval_reports_path,
        eval_reports=eval_reports,
    )
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
    parser.add_argument("--shadow-outcomes", default=str(DEFAULT_SHADOW_OUTCOMES))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_phase6_transport_advisory_runtime(
        output_dir=args.output_dir,
        scaffold_dir=args.scaffold_dir,
        neural_dir=args.neural_dir,
        trainer_dir=args.trainer_dir,
        shadow_outcomes_path=args.shadow_outcomes,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["ready_for_decomposed_eval"]
        and not payload["ready_for_training"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["live_policy_control"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

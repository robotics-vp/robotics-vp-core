#!/usr/bin/env python3
"""Build Phase 7 meta-composition hypernetwork scaffolds without training."""

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

from scripts.economic_world_model.adapt_phase7_governance_node_signals import (  # noqa: E402
    DEFAULT_LOWER_ARTIFACT_ROOT,
)
from scripts.economic_world_model.adapt_phase7_governance_node_signals import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_SIGNAL_ADAPTER_DIR,
)
from scripts.economic_world_model.adapt_phase7_governance_node_signals import (  # noqa: E402
    run_adapt_phase7_governance_node_signals,
)
from scripts.economic_world_model.evaluate_phase7_meta_governance_shadow import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE7_EVAL_DIR,
)
from scripts.economic_world_model.prepare_phase7_meta_regal_control_scaffold import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE7_SCAFFOLD_DIR,
)
from scripts.economic_world_model.wire_phase7_meta_regal_runtime_shadow import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE7_RUNTIME_DIR,
)
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase7_meta_composition_hypernetwork_scaffold,
    load_phase7_composition_mode_specs,
    load_phase7_conflict_join_eval_reports,
    load_phase7_conflict_override_receipts,
    load_phase7_control_field_eval_reports,
    load_phase7_control_field_slots,
    load_phase7_governance_node_signal_adapters,
    load_phase7_governance_node_signal_receipts,
    load_phase7_governance_node_surfaces,
    load_phase7_governance_signal_adapter_report,
    load_phase7_meta_governance_evaluation_report,
    load_phase7_meta_regal_control_scaffold_report,
    load_phase7_outcome_join_rows,
    load_phase7_pareto_regime_eval_reports,
    load_phase7_promotion_gates,
    load_phase7_training_row_slots,
    save_phase7_meta_composition_hypernetwork_scaffold,
)
from src.world_model.humanoid_readiness.common import load_json  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase7_meta_composition_hypernetwork"
)


def _required_scaffold_paths(scaffold_dir: Path) -> dict[str, Path]:
    return {
        "report": scaffold_dir / "phase7_meta_regal_control_scaffold_report_v1.json",
        "surfaces": scaffold_dir / "phase7_governance_node_surfaces_v1.jsonl",
        "modes": scaffold_dir / "phase7_composition_mode_specs_v1.jsonl",
        "conflicts": scaffold_dir / "phase7_conflict_override_receipts_v1.jsonl",
        "control_fields": scaffold_dir / "phase7_control_field_slots_v1.jsonl",
        "training_rows": scaffold_dir / "phase7_training_row_slots_v1.jsonl",
        "promotion_gates": scaffold_dir / "phase7_promotion_gates_v1.jsonl",
    }


def _required_signal_paths(signal_dir: Path) -> dict[str, Path]:
    return {
        "report": signal_dir / "phase7_governance_signal_adapter_report_v1.json",
        "adapters": signal_dir / "phase7_governance_node_signal_adapters_v1.jsonl",
        "receipts": signal_dir / "phase7_governance_node_signal_receipts_v1.jsonl",
    }


def _required_eval_paths(eval_dir: Path) -> dict[str, Path]:
    return {
        "report": eval_dir / "phase7_meta_governance_evaluation_report_v1.json",
        "field_evals": eval_dir / "phase7_control_field_eval_reports_v1.jsonl",
        "conflict_evals": eval_dir / "phase7_conflict_join_eval_reports_v1.jsonl",
        "regime_evals": eval_dir / "phase7_pareto_regime_eval_reports_v1.jsonl",
        "outcome_rows": eval_dir / "phase7_outcome_join_rows_v1.jsonl",
    }


def _paths(
    output: Path,
    scaffold_dir: Path,
    signal_dir: Path,
    eval_dir: Path,
    runtime_dir: Path,
) -> dict[str, Path]:
    return {
        **{f"scaffold_{key}": path for key, path in _required_scaffold_paths(scaffold_dir).items()},
        **{f"signal_{key}": path for key, path in _required_signal_paths(signal_dir).items()},
        **{f"eval_{key}": path for key, path in _required_eval_paths(eval_dir).items()},
        "runtime_summary": runtime_dir / "summary.json",
        "report": output / "phase7_meta_composition_hypernetwork_scaffold_report_v1.json",
        "conditioning_specs": output / "phase7_hypernetwork_conditioning_specs_v1.jsonl",
        "output_heads": output / "phase7_hypernetwork_output_heads_v1.jsonl",
        "loss_definitions": output / "phase7_meta_composition_losses_v1.json",
        "dataset_contract": output / "phase7_hypernetwork_dataset_contract_v1.json",
        "model_config": output / "phase7_hypernetwork_model_config_v1.json",
        "cpu_smoke_forward": output / "phase7_hypernetwork_cpu_smoke_forward_v1.json",
        "markdown": output / "phase7_meta_composition_hypernetwork_scaffold_report_v1.md",
    }


def _resolve_inputs(
    *,
    scaffold_dir: Path,
    signal_dir: Path,
    eval_dir: Path,
    runtime_dir: Path,
    lower_artifact_root: Path,
    run_dependencies_if_missing: bool,
) -> None:
    required = {
        **_required_scaffold_paths(scaffold_dir),
        **_required_signal_paths(signal_dir),
        **_required_eval_paths(eval_dir),
        "runtime_summary": runtime_dir / "summary.json",
    }
    if all(path.exists() for path in required.values()):
        return
    if not run_dependencies_if_missing:
        missing = [str(path) for path in required.values() if not path.exists()]
        raise FileNotFoundError(
            "Missing Phase 7 hypernetwork scaffold inputs: " + ", ".join(missing)
        )
    run_adapt_phase7_governance_node_signals(
        output_dir=signal_dir,
        phase7_scaffold_dir=scaffold_dir,
        lower_artifact_root=lower_artifact_root,
        phase7_runtime_dir=runtime_dir,
        phase7_eval_dir=eval_dir,
        run_dependencies_if_missing=True,
    )
    if not all(path.exists() for path in required.values()):
        missing = [str(path) for path in required.values() if not path.exists()]
        raise FileNotFoundError(
            "Phase 7 dependency builders did not materialize: " + ", ".join(missing)
        )


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 7 Meta-Composition Hypernetwork Scaffold",
        "",
        f"- Report: `{payload['report_id']}`",
        f"- Status: `{payload['status']}`",
        "- Local hypernetwork scaffold complete: "
        f"`{str(payload['local_hypernetwork_scaffold_complete']).lower()}`",
        "- Conditioning wiring complete: "
        f"`{str(payload['conditioning_wiring_complete']).lower()}`",
        "- Future meta-composition explicit: "
        f"`{str(payload['future_meta_composition_explicit']).lower()}`",
        "- CPU smoke forward passed: "
        f"`{str(payload['cpu_smoke_forward_passed']).lower()}`",
        f"- Conditioning specs: `{payload['conditioning_spec_count']}`",
        f"- Output heads: `{payload['output_head_count']}`",
        f"- Losses: `{payload['loss_count']}`",
        "",
        "## Hypernetwork Conditioning",
        "",
        "The future model is wired as a parameter generator conditioned on the",
        "current Phase 7 event spine: node signals, conflict context, Pareto",
        "regime rows, shadow outcome rows, and runtime denial masks. The",
        "economic WM is explicitly a conditioned governance voice inside the",
        "Pareto/meta-composition surface, not a scalar replacement for safety,",
        "deployment truth, reward integrity, embodiment limits, or operator",
        "recovery.",
        "",
        "## Boundary",
        "",
        "This artifact only emits conditioning specs, output-head specs, loss",
        "definitions, a dataset contract, model-config metadata, and a CPU shape",
        "check. It does not train, initialize or write weights, run providers or",
        "hardware, dispatch live actions, execute hard vetoes, mutate reward",
        "math, replace lower WMs, collapse governance into a scalar, or promote",
        "outputs.",
        "",
        "## Remaining Blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["remaining_blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_build_phase7_meta_composition_hypernetwork_scaffold(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase7_scaffold_dir: str | Path = DEFAULT_PHASE7_SCAFFOLD_DIR,
    phase7_signal_adapter_dir: str | Path = DEFAULT_SIGNAL_ADAPTER_DIR,
    phase7_eval_dir: str | Path = DEFAULT_PHASE7_EVAL_DIR,
    phase7_runtime_dir: str | Path = DEFAULT_PHASE7_RUNTIME_DIR,
    lower_artifact_root: str | Path = DEFAULT_LOWER_ARTIFACT_ROOT,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scaffold_dir = Path(phase7_scaffold_dir)
    signal_dir = Path(phase7_signal_adapter_dir)
    eval_dir = Path(phase7_eval_dir)
    runtime_dir = Path(phase7_runtime_dir)
    lower_root = Path(lower_artifact_root)
    _resolve_inputs(
        scaffold_dir=scaffold_dir,
        signal_dir=signal_dir,
        eval_dir=eval_dir,
        runtime_dir=runtime_dir,
        lower_artifact_root=lower_root,
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    paths = _paths(output, scaffold_dir, signal_dir, eval_dir, runtime_dir)
    refs = {f"{key}_path": str(path) for key, path in paths.items()}
    (
        report,
        conditioning_specs,
        output_heads,
        losses,
        dataset_contract,
        model_config,
        smoke_forward,
    ) = build_phase7_meta_composition_hypernetwork_scaffold(
        phase7_report=load_phase7_meta_regal_control_scaffold_report(
            paths["scaffold_report"]
        ),
        surfaces=load_phase7_governance_node_surfaces(paths["scaffold_surfaces"]),
        modes=load_phase7_composition_mode_specs(paths["scaffold_modes"]),
        conflicts=load_phase7_conflict_override_receipts(paths["scaffold_conflicts"]),
        control_fields=load_phase7_control_field_slots(paths["scaffold_control_fields"]),
        training_rows=load_phase7_training_row_slots(paths["scaffold_training_rows"]),
        promotion_gates=load_phase7_promotion_gates(paths["scaffold_promotion_gates"]),
        signal_report=load_phase7_governance_signal_adapter_report(
            paths["signal_report"]
        ),
        signal_adapters=load_phase7_governance_node_signal_adapters(
            paths["signal_adapters"]
        ),
        signal_receipts=load_phase7_governance_node_signal_receipts(
            paths["signal_receipts"]
        ),
        eval_report=load_phase7_meta_governance_evaluation_report(
            paths["eval_report"]
        ),
        field_evals=load_phase7_control_field_eval_reports(paths["eval_field_evals"]),
        conflict_evals=load_phase7_conflict_join_eval_reports(
            paths["eval_conflict_evals"]
        ),
        regime_evals=load_phase7_pareto_regime_eval_reports(
            paths["eval_regime_evals"]
        ),
        outcome_rows=load_phase7_outcome_join_rows(paths["eval_outcome_rows"]),
        runtime_summary=load_json(paths["runtime_summary"]),
        artifact_refs=refs,
    )
    saved_refs = save_phase7_meta_composition_hypernetwork_scaffold(
        output,
        report,
        conditioning_specs,
        output_heads,
        losses,
        dataset_contract,
        model_config,
        smoke_forward,
    )
    payload = report.to_dict()
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved_refs}
    _write_markdown(paths["markdown"], payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--phase7-scaffold-dir", default=str(DEFAULT_PHASE7_SCAFFOLD_DIR))
    parser.add_argument(
        "--phase7-signal-adapter-dir",
        default=str(DEFAULT_SIGNAL_ADAPTER_DIR),
    )
    parser.add_argument("--phase7-eval-dir", default=str(DEFAULT_PHASE7_EVAL_DIR))
    parser.add_argument("--phase7-runtime-dir", default=str(DEFAULT_PHASE7_RUNTIME_DIR))
    parser.add_argument("--lower-artifact-root", default=str(DEFAULT_LOWER_ARTIFACT_ROOT))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_build_phase7_meta_composition_hypernetwork_scaffold(
        output_dir=args.output_dir,
        phase7_scaffold_dir=args.phase7_scaffold_dir,
        phase7_signal_adapter_dir=args.phase7_signal_adapter_dir,
        phase7_eval_dir=args.phase7_eval_dir,
        phase7_runtime_dir=args.phase7_runtime_dir,
        lower_artifact_root=args.lower_artifact_root,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["local_hypernetwork_scaffold_complete"]
        and payload["conditioning_wiring_complete"]
        and payload["future_meta_composition_explicit"]
        and payload["cpu_smoke_forward_passed"]
        and not payload["ready_for_training"]
        and not payload["ready_for_gpu_training"]
        and not payload["phase7_authority_granted"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["provider_executed"]
        and not payload["hardware_executed"]
        and not payload["unitree_sim_runtime_executed"]
        and not payload["live_policy_control"]
        and not payload["live_dispatch_allowed"]
        and not payload["hard_veto_dispatch"]
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        and not any(payload["denied_gates"].values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

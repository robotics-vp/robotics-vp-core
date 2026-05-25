#!/usr/bin/env python3
"""Build Phase 6.5 meta-node trainer/loss scaffolds without training."""

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

from scripts.economic_world_model.prepare_phase65_meta_node_neuralization import (  # noqa: E402
    DEFAULT_OUTPUT_DIR as DEFAULT_PHASE65_DIR,
)
from scripts.economic_world_model.prepare_phase65_meta_node_neuralization import (  # noqa: E402
    run_prepare_phase65_meta_node_neuralization,
)
from src.world_model.humanoid_readiness import (  # noqa: E402
    build_phase65_meta_node_trainer_scaffold,
    load_meta_node_counterfactual_targets,
    load_meta_node_intervention_receipts,
    load_meta_node_promotion_gates,
    load_meta_node_robustness_reports,
    load_meta_node_states,
    load_meta_node_trajectory_receipts,
    load_phase65_meta_node_neuralization_report,
    save_phase65_meta_node_trainer_scaffold,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase65_meta_node_trainer_scaffold"
)


def _required_phase65_paths(phase65_dir: Path) -> dict[str, Path]:
    return {
        "report": phase65_dir / "phase65_meta_node_neuralization_report_v1.json",
        "states": phase65_dir / "meta_node_states_v1.jsonl",
        "trajectories": phase65_dir / "meta_node_trajectory_receipts_v1.jsonl",
        "interventions": phase65_dir / "meta_node_intervention_receipts_v1.jsonl",
        "targets": phase65_dir / "meta_node_counterfactual_targets_v1.jsonl",
        "robustness": phase65_dir / "meta_node_robustness_reports_v1.jsonl",
        "gates": phase65_dir / "meta_node_promotion_gates_v1.jsonl",
    }


def _paths(output: Path, phase65_dir: Path) -> dict[str, Path]:
    return {
        **_required_phase65_paths(phase65_dir),
        "manifest": output / "phase65_meta_node_trainer_scaffold_manifest_v1.json",
        "dataset_contract": output
        / "phase65_meta_node_trainer_dataset_contract_v1.json",
        "loss_definitions": output / "phase65_meta_node_loss_definitions_v1.json",
        "model_config": output / "phase65_meta_node_model_component_config_v1.json",
        "cpu_smoke_forward": output / "phase65_meta_node_cpu_smoke_forward_v1.json",
        "markdown": output / "phase65_meta_node_trainer_scaffold_manifest_v1.md",
    }


def _resolve_inputs(
    *,
    phase65_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = _required_phase65_paths(phase65_dir)
    if all(path.exists() for path in paths.values()):
        return paths
    if not run_dependencies_if_missing:
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Missing Phase 6.5 trainer inputs: " + ", ".join(missing)
        )
    run_prepare_phase65_meta_node_neuralization(
        output_dir=phase65_dir,
        run_dependencies_if_missing=True,
    )
    if not all(path.exists() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Phase 6.5 dependency builder did not materialize: " + ", ".join(missing)
        )
    return paths


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6.5 Meta-Node Trainer Scaffold",
        "",
        f"- Manifest: `{payload['trainer_scaffold_id']}`",
        f"- Status: `{payload['status']}`",
        f"- Dataset contract ready: `{str(payload['dataset_contract_ready']).lower()}`",
        f"- Losses defined: `{str(payload['losses_defined']).lower()}`",
        f"- Model config ready: `{str(payload['model_config_ready']).lower()}`",
        "- CPU smoke forward passed: "
        f"`{str(payload['cpu_smoke_forward_passed']).lower()}`",
        f"- Loss count: `{payload['loss_count']}`",
        f"- Ready for training: `{str(payload['ready_for_training']).lower()}`",
        "- Ready for GPU training: "
        f"`{str(payload['ready_for_gpu_training']).lower()}`",
        "",
        "## Boundary",
        "",
        "This artifact defines dataset, loss, model-config, and CPU shape-check",
        "contracts for future Phase 6.5 meta-node training. It does not train,",
        "initialize or write weights, grant Phase 7 authority, mutate reward",
        "math, control live policy, or promote outputs.",
        "",
        "## Remaining Blockers",
        "",
    ]
    lines.extend(f"- `{item}`" for item in payload["remaining_blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_build_phase65_meta_node_trainer_scaffold(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    phase65_dir: str | Path = DEFAULT_PHASE65_DIR,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    phase65_path = Path(phase65_dir)
    input_paths = _resolve_inputs(
        phase65_dir=phase65_path,
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    paths = _paths(output, phase65_path)
    refs = {f"{key}_path": str(path) for key, path in paths.items()}
    manifest, dataset_contract, losses, model_config, smoke_forward = (
        build_phase65_meta_node_trainer_scaffold(
            phase65_report=load_phase65_meta_node_neuralization_report(
                input_paths["report"]
            ),
            states=load_meta_node_states(input_paths["states"]),
            trajectories=load_meta_node_trajectory_receipts(
                input_paths["trajectories"]
            ),
            interventions=load_meta_node_intervention_receipts(
                input_paths["interventions"]
            ),
            targets=load_meta_node_counterfactual_targets(input_paths["targets"]),
            robustness_reports=load_meta_node_robustness_reports(
                input_paths["robustness"]
            ),
            gates=load_meta_node_promotion_gates(input_paths["gates"]),
            artifact_refs=refs,
        )
    )
    saved_refs = save_phase65_meta_node_trainer_scaffold(
        output,
        manifest,
        dataset_contract,
        losses,
        model_config,
        smoke_forward,
    )
    payload = manifest.to_dict()
    payload["artifact_refs"] = {**payload.get("artifact_refs", {}), **saved_refs}
    _write_markdown(paths["markdown"], payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--phase65-dir", default=str(DEFAULT_PHASE65_DIR))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_build_phase65_meta_node_trainer_scaffold(
        output_dir=args.output_dir,
        phase65_dir=args.phase65_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["status"] == "ok"
        and payload["dataset_contract_ready"]
        and payload["losses_defined"]
        and payload["model_config_ready"]
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
        and not payload["reward_math_mutation"]
        and not payload["promotion_eligible"]
        and not any(payload["denied_gates"].values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

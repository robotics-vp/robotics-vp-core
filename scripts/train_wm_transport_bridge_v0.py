#!/usr/bin/env python3
"""Phase-6.3 WM transport bridge trainer scaffold.

This is intentionally non-training. It validates dataset/model/loss shapes,
runs deterministic CPU-only smoke forwards, emits manifests, and denies
promotion/training authority until GPU/provider/hardware/benchmark evidence
exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from scripts.economic_world_model.build_phase6_transport_neural_manifest import (  # noqa: E402
    run_build_phase6_transport_neural_manifest,
)
from src.world_model.transport.losses import load_wm_transport_loss_ledger  # noqa: E402
from src.world_model.transport.neural_manifest import (  # noqa: E402
    load_wm_transport_neural_architecture_manifest,
)
from src.world_model.transport.training import (  # noqa: E402
    build_wm_transport_cpu_smoke_forward_report,
    build_wm_transport_model_component_config,
    build_wm_transport_trainer_dataset_contract,
    build_wm_transport_trainer_scaffold_manifest,
    save_wm_transport_trainer_dataset_contract,
    save_wm_transport_trainer_scaffold_manifest,
)
from src.world_model.transport.training_rows import (  # noqa: E402
    load_wm_transport_training_manifest,
    load_wm_transport_training_rows,
)

DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase6_transport_trainer_scaffold"
)
DEFAULT_NEURAL_DIR = Path(
    "artifacts/economic_world_model/phase6_transport_neural_manifest"
)
DEFAULT_SCAFFOLD_DIR = Path("artifacts/economic_world_model/phase6_transport_scaffold")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_markdown(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# WM Transport Bridge Trainer Scaffold v0",
        "",
        f"- Trainer scaffold: `{manifest['trainer_scaffold_id']}`",
        f"- Neural manifest: `{manifest['neural_manifest_id']}`",
        f"- Loss ledger: `{manifest['loss_ledger_id']}`",
        f"- Dataset contract ready: `{str(manifest['dataset_contract_ready']).lower()}`",
        f"- Losses defined: `{str(manifest['losses_defined']).lower()}`",
        f"- CPU smoke forward passed: `{str(manifest['cpu_smoke_forward_passed']).lower()}`",
        f"- Ready for training: `{str(manifest['ready_for_training']).lower()}`",
        f"- Ready for GPU training: `{str(manifest['ready_for_gpu_training']).lower()}`",
        f"- Training executed: `{str(manifest['training_executed']).lower()}`",
        f"- Weights written: `{str(manifest['weights_written']).lower()}`",
        f"- Promotion eligible: `{str(manifest['promotion_eligible']).lower()}`",
        "",
        "## Boundary",
        "",
        "This scaffold emits shape contracts, component configs, loss ledgers, and",
        "finite CPU smoke-forward evidence only. It does not train, initialize or",
        "write real weights, run providers, execute hardware, grant live control, or",
        "mutate frozen reward/trust/`w_econ`/lambda math.",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- `{blocker}`" for blocker in manifest["blockers"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_neural_artifacts(
    *,
    neural_dir: Path,
    scaffold_dir: Path,
    run_dependencies_if_missing: bool,
) -> dict[str, Path]:
    paths = {
        "neural_manifest": neural_dir
        / "wm_transport_neural_architecture_manifest_v1.json",
        "loss_ledger": neural_dir / "wm_transport_loss_ledger_v1.json",
        "training_manifest": scaffold_dir / "wm_transport_training_manifest_v1.json",
        "training_rows": scaffold_dir / "wm_transport_training_rows_v1.jsonl",
    }
    if all(path.exists() for path in paths.values()):
        return paths
    if not run_dependencies_if_missing:
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Missing Phase-6.3 trainer inputs: " + ", ".join(missing)
        )
    run_build_phase6_transport_neural_manifest(
        output_dir=neural_dir,
        scaffold_dir=scaffold_dir,
        run_dependencies_if_missing=True,
    )
    if not all(path.exists() for path in paths.values()):
        raise FileNotFoundError(
            "Phase-6.3 neural builder did not materialize all inputs"
        )
    return paths


def run_train_wm_transport_bridge_v0_scaffold(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    neural_dir: str | Path = DEFAULT_NEURAL_DIR,
    scaffold_dir: str | Path = DEFAULT_SCAFFOLD_DIR,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = _resolve_neural_artifacts(
        neural_dir=Path(neural_dir),
        scaffold_dir=Path(scaffold_dir),
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    neural_manifest = load_wm_transport_neural_architecture_manifest(
        paths["neural_manifest"]
    )
    loss_ledger = load_wm_transport_loss_ledger(paths["loss_ledger"])
    training_manifest = load_wm_transport_training_manifest(paths["training_manifest"])
    training_rows = load_wm_transport_training_rows(paths["training_rows"])

    dataset_contract = build_wm_transport_trainer_dataset_contract(
        neural_manifest=neural_manifest,
        training_manifest=training_manifest,
        training_rows=training_rows,
    )
    model_config = build_wm_transport_model_component_config(
        dataset_contract=dataset_contract,
        neural_manifest=neural_manifest,
    )
    cpu_smoke = build_wm_transport_cpu_smoke_forward_report(
        dataset_contract=dataset_contract,
        model_config=model_config,
        loss_ledger=loss_ledger,
        training_rows=training_rows,
    )

    dataset_path = output / "wm_transport_trainer_dataset_contract_v1.json"
    model_config_path = output / "wm_transport_model_component_config_v1.json"
    cpu_smoke_path = output / "wm_transport_cpu_smoke_forward_v1.json"
    manifest_path = output / "wm_transport_trainer_scaffold_manifest_v1.json"
    markdown_path = output / "wm_transport_trainer_scaffold_v1.md"
    artifact_refs = {
        "neural_manifest_path": str(paths["neural_manifest"]),
        "loss_ledger_path": str(paths["loss_ledger"]),
        "training_manifest_path": str(paths["training_manifest"]),
        "training_rows_path": str(paths["training_rows"]),
        "dataset_contract_path": str(dataset_path),
        "model_config_path": str(model_config_path),
        "cpu_smoke_forward_path": str(cpu_smoke_path),
        "trainer_manifest_path": str(manifest_path),
        "markdown_path": str(markdown_path),
    }
    trainer_manifest = build_wm_transport_trainer_scaffold_manifest(
        neural_manifest=neural_manifest,
        loss_ledger=loss_ledger,
        dataset_contract=dataset_contract,
        model_config=model_config,
        cpu_smoke_forward=cpu_smoke,
        artifact_refs=artifact_refs,
    )
    save_wm_transport_trainer_dataset_contract(dataset_path, dataset_contract)
    _write_json(model_config_path, model_config)
    _write_json(cpu_smoke_path, cpu_smoke)
    save_wm_transport_trainer_scaffold_manifest(manifest_path, trainer_manifest)
    payload = trainer_manifest.to_dict()
    _write_markdown(markdown_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--neural-dir", default=str(DEFAULT_NEURAL_DIR))
    parser.add_argument("--scaffold-dir", default=str(DEFAULT_SCAFFOLD_DIR))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_train_wm_transport_bridge_v0_scaffold(
        output_dir=args.output_dir,
        neural_dir=args.neural_dir,
        scaffold_dir=args.scaffold_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["dataset_contract_ready"]
        and payload["losses_defined"]
        and payload["cpu_smoke_forward_passed"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["promotion_eligible"]
        and not payload["reward_math_mutation"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

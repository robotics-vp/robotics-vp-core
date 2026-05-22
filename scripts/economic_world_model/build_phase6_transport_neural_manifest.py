#!/usr/bin/env python3
"""Build Phase-6.3 transport neural manifest and loss ledger artifacts."""

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

from scripts.economic_world_model.prepare_phase6_transport_scaffold import (  # noqa: E402
    run_prepare_phase6_transport_scaffold,
)
from src.world_model.transport.bridge_contracts import (  # noqa: E402
    load_wm_transport_bridge_contracts,
    load_wm_transport_contract_pack,
)
from src.world_model.transport.losses import (  # noqa: E402
    build_wm_transport_loss_ledger,
    save_wm_transport_loss_ledger,
)
from src.world_model.transport.neural_manifest import (  # noqa: E402
    build_wm_transport_neural_architecture_manifest,
    save_wm_transport_neural_architecture_manifest,
)
from src.world_model.transport.training_rows import (  # noqa: E402
    load_wm_transport_training_manifest,
    load_wm_transport_training_rows,
)
from src.world_model.transport.wm_transformers import (  # noqa: E402
    load_per_wm_transformer_registry,
)

DEFAULT_SCAFFOLD_DIR = Path("artifacts/economic_world_model/phase6_transport_scaffold")
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/economic_world_model/phase6_transport_neural_manifest"
)


def _write_markdown(
    path: Path, manifest: Mapping[str, Any], ledger: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 6.3 Transport Neural Manifest",
        "",
        f"- Manifest: `{manifest['manifest_id']}`",
        f"- Architecture stage: `{manifest['architecture_stage']}`",
        f"- Components: `{len(manifest['components'])}`",
        f"- Loss definitions: `{ledger['loss_count']}`",
        f"- Ready for trainer scaffold: `{str(manifest['ready_for_trainer_scaffold']).lower()}`",
        f"- Ready for GPU training: `{str(manifest['ready_for_gpu_training']).lower()}`",
        f"- Training executed: `{str(manifest['training_executed']).lower()}`",
        f"- Weights written: `{str(manifest['weights_written']).lower()}`",
        f"- Promotion eligible: `{str(manifest['promotion_eligible']).lower()}`",
        "",
        "## Components",
        "",
    ]
    for component in manifest["components"]:
        lines.append(f"- `{component['component_key']}` — {component['model_family']}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This artifact defines neural topology and losses only. It does not train,",
            "write weights, invoke providers, run hardware, grant live authority, or",
            "mutate frozen reward/trust/`w_econ`/lambda math.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_scaffold(
    *, scaffold_dir: Path, output_dir: Path, run_dependencies_if_missing: bool
) -> dict[str, Path]:
    paths = {
        "contract_pack": scaffold_dir / "wm_transport_contract_pack_v1.json",
        "contracts": scaffold_dir / "wm_transport_bridge_contracts_v1.jsonl",
        "registry": scaffold_dir / "per_wm_transport_transformer_registry_v1.json",
        "training_manifest": scaffold_dir / "wm_transport_training_manifest_v1.json",
        "training_rows": scaffold_dir / "wm_transport_training_rows_v1.jsonl",
    }
    if all(path.exists() for path in paths.values()):
        return paths
    if not run_dependencies_if_missing:
        missing = [str(path) for path in paths.values() if not path.exists()]
        raise FileNotFoundError(
            "Missing Phase-6 scaffold inputs: " + ", ".join(missing)
        )
    run_prepare_phase6_transport_scaffold(
        output_dir=scaffold_dir,
        run_dependencies_if_missing=True,
    )
    if not all(path.exists() for path in paths.values()):
        raise FileNotFoundError(
            "Phase-6 scaffold builder did not materialize all inputs"
        )
    return paths


def run_build_phase6_transport_neural_manifest(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    scaffold_dir: str | Path = DEFAULT_SCAFFOLD_DIR,
    run_dependencies_if_missing: bool = True,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scaffold_paths = _resolve_scaffold(
        scaffold_dir=Path(scaffold_dir),
        output_dir=output,
        run_dependencies_if_missing=run_dependencies_if_missing,
    )
    contract_pack = load_wm_transport_contract_pack(scaffold_paths["contract_pack"])
    contracts = load_wm_transport_bridge_contracts(scaffold_paths["contracts"])
    registry = load_per_wm_transformer_registry(scaffold_paths["registry"])
    training_manifest = load_wm_transport_training_manifest(
        scaffold_paths["training_manifest"]
    )
    training_rows = load_wm_transport_training_rows(scaffold_paths["training_rows"])

    manifest_path = output / "wm_transport_neural_architecture_manifest_v1.json"
    loss_ledger_path = output / "wm_transport_loss_ledger_v1.json"
    markdown_path = output / "wm_transport_neural_architecture_manifest_v1.md"
    artifact_refs = {
        "contract_pack_path": str(scaffold_paths["contract_pack"]),
        "contracts_path": str(scaffold_paths["contracts"]),
        "registry_path": str(scaffold_paths["registry"]),
        "training_manifest_path": str(scaffold_paths["training_manifest"]),
        "training_rows_path": str(scaffold_paths["training_rows"]),
        "neural_manifest_path": str(manifest_path),
        "loss_ledger_path": str(loss_ledger_path),
        "markdown_path": str(markdown_path),
    }
    neural_manifest = build_wm_transport_neural_architecture_manifest(
        contract_pack=contract_pack,
        contracts=contracts,
        transformer_registry=registry,
        training_manifest=training_manifest,
        training_rows=training_rows,
        artifact_refs=artifact_refs,
    )
    loss_ledger = build_wm_transport_loss_ledger(
        neural_manifest=neural_manifest,
        training_manifest=training_manifest,
        training_rows=training_rows,
    )
    save_wm_transport_neural_architecture_manifest(manifest_path, neural_manifest)
    save_wm_transport_loss_ledger(loss_ledger_path, loss_ledger)
    _write_markdown(markdown_path, neural_manifest.to_dict(), loss_ledger.to_dict())
    payload = {
        "manifest_id": neural_manifest.manifest_id,
        "loss_ledger_id": loss_ledger.ledger_id,
        "component_count": len(neural_manifest.components),
        "loss_count": loss_ledger.loss_count,
        "ready_for_trainer_scaffold": neural_manifest.ready_for_trainer_scaffold
        and loss_ledger.ready_for_cpu_smoke_forward,
        "ready_for_gpu_training": False,
        "training_executed": False,
        "weights_written": False,
        "promotion_eligible": False,
        "reward_math_mutation": False,
        "artifact_refs": artifact_refs,
        "authority_class": "transport_neural_manifest_builder_only",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scaffold-dir", default=str(DEFAULT_SCAFFOLD_DIR))
    parser.add_argument("--no-run-dependencies", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = run_build_phase6_transport_neural_manifest(
        output_dir=args.output_dir,
        scaffold_dir=args.scaffold_dir,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    return (
        0
        if payload["ready_for_trainer_scaffold"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["promotion_eligible"]
        and not payload["reward_math_mutation"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

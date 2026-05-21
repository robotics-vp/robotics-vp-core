#!/usr/bin/env python3
"""Economic WM v0 trainer scaffold.

This script is intentionally non-training. It validates dataset/model/loss
shapes, runs deterministic CPU-only smoke forwards, emits manifests, and denies
promotion/training authority until GPU/provider/benchmark evidence exists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

if __package__ is None or __package__ == "":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_digest import sha256_json  # noqa: E402
from src.utils.json_safe import to_json_safe  # noqa: E402
from src.world_model.economic_world_model import (  # noqa: E402
    EconomicWMNeuralArchitectureManifest,
    EconomicWMPhase5LocalPrepManifest,
    load_economic_wm_counterfactual_value_join_rows,
    load_economic_wm_datapack_composition_rows,
    load_economic_wm_neural_architecture_manifest,
    load_economic_wm_phase5_local_prep_manifest,
    load_economic_wm_temporal_window_rows,
)

TRAINER_SCAFFOLD_VERSION = "economic_wm_trainer_scaffold_manifest_v1"
DATASET_CONTRACT_VERSION = "economic_wm_trainer_dataset_contract_v1"
MODEL_CONFIG_VERSION = "economic_wm_trainer_model_component_config_v1"
LOSS_DEFINITIONS_VERSION = "economic_wm_trainer_loss_definitions_v1"
CPU_SMOKE_REPORT_VERSION = "economic_wm_trainer_cpu_smoke_forward_v1"

TRAINER_BLOCKERS = [
    "gpu_training_not_run",
    "provider_bringup_not_run",
    "promotion_grade_benchmark_evidence_missing",
    "non_stub_teacher_runtime_not_verified",
    "weights_not_written",
]


def _mapping(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(to_json_safe(dict(payload or {})))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _unique(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _union_float_keys(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        for key, value in dict(row).items():
            try:
                float(value)
            except Exception:
                continue
            keys.add(str(key))
    return sorted(keys)


def _vector_from_keys(payload: Mapping[str, Any], keys: list[str]) -> list[float]:
    values: list[float] = []
    for key in keys:
        try:
            values.append(float(payload.get(key, 0.0)))
        except Exception:
            values.append(0.0)
    return values


def _resolve_phase5_prep(
    *,
    output_root: Path,
    phase5_prep_path: Optional[str | Path],
    run_if_missing: bool,
) -> Path:
    path = Path(
        phase5_prep_path
        or "artifacts/economic_world_model/economic_wm_phase5_local_prep/economic_wm_phase5_local_prep_manifest_v1.json"
    )
    if path.exists():
        return path
    if not run_if_missing:
        raise FileNotFoundError(path)
    from scripts.economic_world_model.prepare_economic_wm_phase5_local_prep import (
        run_prepare_economic_wm_phase5_local_prep,
    )

    payload = run_prepare_economic_wm_phase5_local_prep(
        output_dir=output_root / "phase5_local_prep"
    )
    return Path(payload["artifact_refs"]["manifest_path"])


def _resolve_neural_manifest(
    *,
    output_root: Path,
    neural_manifest_path: Optional[str | Path],
    run_if_missing: bool,
) -> Path:
    path = Path(
        neural_manifest_path
        or "artifacts/economic_world_model/economic_wm_neural_architecture_manifest/economic_wm_neural_architecture_manifest_v1.json"
    )
    if path.exists():
        return path
    if not run_if_missing:
        raise FileNotFoundError(path)
    from scripts.economic_world_model.build_economic_wm_neural_architecture_manifest import (  # noqa: E501
        run_build_economic_wm_neural_architecture_manifest,
    )

    payload = run_build_economic_wm_neural_architecture_manifest(
        output_dir=output_root / "neural_architecture_manifest"
    )
    return Path(payload["artifact_refs"]["manifest_path"])


def _load_phase5_row_families(
    phase5_manifest: EconomicWMPhase5LocalPrepManifest,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    compositions = [
        item.to_dict()
        for item in load_economic_wm_datapack_composition_rows(
            phase5_manifest.composition_rows_path
        )
    ]
    joins = [
        item.to_dict()
        for item in load_economic_wm_counterfactual_value_join_rows(
            phase5_manifest.counterfactual_value_joins_path
        )
    ]
    windows = [
        item.to_dict()
        for item in load_economic_wm_temporal_window_rows(
            phase5_manifest.temporal_windows_path
        )
    ]
    return compositions, joins, windows


def _build_dataset_contract(
    *,
    phase5_manifest: EconomicWMPhase5LocalPrepManifest,
    compositions: list[Mapping[str, Any]],
    joins: list[Mapping[str, Any]],
    windows: list[Mapping[str, Any]],
    neural_manifest: EconomicWMNeuralArchitectureManifest,
) -> Dict[str, Any]:
    composition_feature_keys = _union_float_keys(
        row.get("feature_vector", {}) for row in compositions
    )
    composition_target_keys = _union_float_keys(
        row.get("target_vector", {}) for row in compositions
    )
    temporal_feature_keys = _union_float_keys(
        row.get("aggregate_feature_vector", {}) for row in windows
    )
    temporal_target_keys = _union_float_keys(
        row.get("aggregate_target_vector", {}) for row in windows
    )
    required_surfaces = _unique(
        surface
        for component in neural_manifest.components
        for surface in component.input_surfaces
    )
    payload = {
        "version": DATASET_CONTRACT_VERSION,
        "dataset_contract_id": "",
        "phase5_manifest_id": phase5_manifest.manifest_id,
        "neural_manifest_id": neural_manifest.manifest_id,
        "row_families": {
            "datapack_composition_rows": len(compositions),
            "counterfactual_value_join_rows": len(joins),
            "temporal_window_rows": len(windows),
        },
        "shape_contracts": {
            "composition_feature_dim": len(composition_feature_keys),
            "composition_target_dim": len(composition_target_keys),
            "temporal_feature_dim": len(temporal_feature_keys),
            "temporal_target_dim": len(temporal_target_keys),
            "join_row_count": len(joins),
        },
        "composition_feature_keys": composition_feature_keys,
        "composition_target_keys": composition_target_keys,
        "temporal_feature_keys": temporal_feature_keys,
        "temporal_target_keys": temporal_target_keys,
        "required_input_surfaces": required_surfaces,
        "authority_class": "trainer_dataset_contract_only",
        "ready_for_cpu_smoke_forward": bool(compositions and windows),
        "ready_for_gpu_training": False,
        "training_executed": False,
        "weights_written": False,
        "promotion_eligible": False,
        "reward_math_mutation": False,
        "blockers": list(TRAINER_BLOCKERS),
    }
    payload["dataset_contract_id"] = f"ewm_dataset_contract_{sha256_json(payload)[:16]}"
    return payload


def _build_model_component_config(
    *,
    dataset_contract: Mapping[str, Any],
    neural_manifest: EconomicWMNeuralArchitectureManifest,
) -> Dict[str, Any]:
    shapes = dict(dataset_contract.get("shape_contracts", {}) or {})
    input_dim = int(shapes.get("composition_feature_dim", 0)) + int(
        shapes.get("temporal_feature_dim", 0)
    )
    input_dim = max(1, input_dim)
    components = []
    for component in neural_manifest.components:
        output_dim = max(1, len(component.output_surfaces))
        components.append(
            {
                "component_key": component.component_key,
                "model_family": component.model_family,
                "runtime_plane": component.runtime_plane,
                "input_dim": input_dim,
                "hidden_dims": [
                    min(128, max(8, input_dim * 2)),
                    min(64, max(4, input_dim)),
                ],
                "output_dim": output_dim,
                "input_surfaces": list(component.input_surfaces),
                "output_surfaces": list(component.output_surfaces),
                "training_enabled": False,
                "weights_initialized": False,
                "weights_written": False,
                "promotion_eligible": False,
                "authority_class": "model_component_config_only",
            }
        )
    payload = {
        "version": MODEL_CONFIG_VERSION,
        "model_config_id": "",
        "dataset_contract_id": dataset_contract["dataset_contract_id"],
        "component_count": len(components),
        "components": components,
        "training_executed": False,
        "weights_written": False,
        "ready_for_gpu_training": False,
        "promotion_eligible": False,
        "reward_math_mutation": False,
        "blockers": list(TRAINER_BLOCKERS),
    }
    payload["model_config_id"] = f"ewm_model_config_{sha256_json(payload)[:16]}"
    return payload


def _build_loss_definitions(
    model_config: Mapping[str, Any],
) -> Dict[str, Any]:
    loss_by_component: Dict[str, Dict[str, Any]] = {}
    for component in model_config.get("components", []):
        key = str(component.get("component_key", "component"))
        if "allocator" in key:
            loss_family = "shadow_regret_and_pareto_quality_proxy"
        elif "dynamics" in key:
            loss_family = "temporal_consistency_l1"
        elif "governance" in key:
            loss_family = "constraint_violation_penalty"
        elif "datapack" in key:
            loss_family = "composition_reconstruction_and_contrastive_proxy"
        else:
            loss_family = "masked_mse"
        loss_by_component[key] = {
            "loss_family": loss_family,
            "optimization_status": "defined_not_optimized",
            "requires_gpu_training_for_real_fit": True,
            "promotion_gate": "promotion_grade_shadow_benchmark_evidence",
        }
    payload = {
        "version": LOSS_DEFINITIONS_VERSION,
        "loss_definitions_id": "",
        "model_config_id": model_config["model_config_id"],
        "loss_by_component": loss_by_component,
        "training_executed": False,
        "weights_written": False,
        "promotion_eligible": False,
        "reward_math_mutation": False,
        "blockers": list(TRAINER_BLOCKERS),
    }
    payload["loss_definitions_id"] = f"ewm_loss_defs_{sha256_json(payload)[:16]}"
    return payload


def _smoke_forward_vector(
    *,
    dataset_contract: Mapping[str, Any],
    compositions: list[Mapping[str, Any]],
    windows: list[Mapping[str, Any]],
) -> list[float]:
    composition_keys = list(dataset_contract.get("composition_feature_keys", []))
    temporal_keys = list(dataset_contract.get("temporal_feature_keys", []))
    composition_source = (
        compositions[0].get("feature_vector", {}) if compositions else {}
    )
    temporal_source = windows[0].get("aggregate_feature_vector", {}) if windows else {}
    return [
        *_vector_from_keys(composition_source, composition_keys),
        *_vector_from_keys(temporal_source, temporal_keys),
    ] or [0.0]


def _component_forward(
    component: Mapping[str, Any], vector: list[float]
) -> list[float]:
    output_dim = int(component.get("output_dim", 1) or 1)
    key = str(component.get("component_key", "component"))
    scale = (sum(ord(char) for char in key) % 23 + 3) / 100.0
    weighted_sum = sum((idx + 1) * value for idx, value in enumerate(vector))
    denom = max(1.0, float(len(vector)))
    return [weighted_sum * scale / (denom * (idx + 1)) for idx in range(output_dim)]


def _build_cpu_smoke_forward_report(
    *,
    dataset_contract: Mapping[str, Any],
    model_config: Mapping[str, Any],
    compositions: list[Mapping[str, Any]],
    windows: list[Mapping[str, Any]],
) -> Dict[str, Any]:
    vector = _smoke_forward_vector(
        dataset_contract=dataset_contract,
        compositions=compositions,
        windows=windows,
    )
    component_reports = []
    finite = True
    for component in model_config.get("components", []):
        output = _component_forward(component, vector)
        finite = finite and all(math.isfinite(value) for value in output)
        component_reports.append(
            {
                "component_key": component["component_key"],
                "input_dim": int(component["input_dim"]),
                "output_dim": int(component["output_dim"]),
                "observed_input_dim": len(vector),
                "observed_output_dim": len(output),
                "output_sample": output[: min(4, len(output))],
                "shape_check_passed": len(vector) == int(component["input_dim"])
                and len(output) == int(component["output_dim"]),
                "finite_check_passed": all(math.isfinite(value) for value in output),
            }
        )
    shape_passed = all(item["shape_check_passed"] for item in component_reports)
    payload = {
        "version": CPU_SMOKE_REPORT_VERSION,
        "cpu_smoke_forward_id": "",
        "dataset_contract_id": dataset_contract["dataset_contract_id"],
        "model_config_id": model_config["model_config_id"],
        "component_reports": component_reports,
        "input_vector_dim": len(vector),
        "component_count": len(component_reports),
        "cpu_smoke_forward_passed": bool(component_reports and finite and shape_passed),
        "training_executed": False,
        "weights_written": False,
        "ready_for_gpu_training": False,
        "promotion_eligible": False,
        "reward_math_mutation": False,
        "blockers": list(TRAINER_BLOCKERS),
    }
    payload["cpu_smoke_forward_id"] = f"ewm_cpu_smoke_{sha256_json(payload)[:16]}"
    return payload


def _trainer_manifest(
    *,
    phase5_manifest: EconomicWMPhase5LocalPrepManifest,
    neural_manifest: EconomicWMNeuralArchitectureManifest,
    dataset_contract: Mapping[str, Any],
    model_config: Mapping[str, Any],
    loss_definitions: Mapping[str, Any],
    smoke_forward: Mapping[str, Any],
    artifact_refs: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = {
        "trainer_scaffold_id": "",
        "version": TRAINER_SCAFFOLD_VERSION,
        "phase5_manifest_id": phase5_manifest.manifest_id,
        "neural_manifest_id": neural_manifest.manifest_id,
        "dataset_contract_id": dataset_contract["dataset_contract_id"],
        "model_config_id": model_config["model_config_id"],
        "loss_definitions_id": loss_definitions["loss_definitions_id"],
        "cpu_smoke_forward_id": smoke_forward["cpu_smoke_forward_id"],
        "authority_class": "trainer_scaffold_only",
        "dataset_contract_ready": bool(
            dataset_contract.get("ready_for_cpu_smoke_forward", False)
        ),
        "cpu_smoke_forward_passed": bool(
            smoke_forward.get("cpu_smoke_forward_passed", False)
        ),
        "losses_defined": bool(loss_definitions.get("loss_by_component")),
        "training_executed": False,
        "weights_written": False,
        "ready_for_gpu_training": False,
        "ready_for_training": False,
        "promotion_eligible": False,
        "reward_math_mutation": False,
        "blockers": list(TRAINER_BLOCKERS),
        "artifact_refs": _mapping(artifact_refs),
        "metadata": {
            "boundary": "shape-checked trainer scaffold only",
            "cpu_smoke_only": True,
        },
    }
    payload["trainer_scaffold_id"] = f"ewm_trainer_scaffold_{sha256_json(payload)[:16]}"
    return payload


def _write_markdown(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# Economic WM Trainer Scaffold v0",
        "",
        f"- Trainer scaffold ID: `{manifest['trainer_scaffold_id']}`",
        f"- Phase-5 manifest ID: `{manifest['phase5_manifest_id']}`",
        f"- Neural manifest ID: `{manifest['neural_manifest_id']}`",
        f"- Authority class: `{manifest['authority_class']}`",
        f"- Dataset contract ready: `{str(manifest['dataset_contract_ready']).lower()}`",
        f"- CPU smoke forward passed: `{str(manifest['cpu_smoke_forward_passed']).lower()}`",
        f"- Losses defined: `{str(manifest['losses_defined']).lower()}`",
        f"- Training executed: `{str(manifest['training_executed']).lower()}`",
        f"- Weights written: `{str(manifest['weights_written']).lower()}`",
        f"- Ready for GPU training: `{str(manifest['ready_for_gpu_training']).lower()}`",
        f"- Promotion eligible: `{str(manifest['promotion_eligible']).lower()}`",
        f"- Reward math mutation: `{str(manifest['reward_math_mutation']).lower()}`",
        "",
        "## Blockers",
    ]
    lines.extend(f"- `{blocker}`" for blocker in manifest["blockers"])
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This script emits a dataset contract, model component configs, loss definitions, and deterministic CPU smoke forwards only. It does not train, initialize or write real weights, run providers, promote a model, or mutate reward/trust/`w_econ`/lambda math.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_train_economic_world_model_v0_scaffold(
    *,
    output_dir: str | Path,
    phase5_prep_path: Optional[str | Path] = None,
    neural_manifest_path: Optional[str | Path] = None,
    run_dependencies_if_missing: bool = True,
) -> Dict[str, Any]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_phase5_path = _resolve_phase5_prep(
        output_root=output_root,
        phase5_prep_path=phase5_prep_path,
        run_if_missing=run_dependencies_if_missing,
    )
    resolved_neural_path = _resolve_neural_manifest(
        output_root=output_root,
        neural_manifest_path=neural_manifest_path,
        run_if_missing=run_dependencies_if_missing,
    )
    phase5_manifest = load_economic_wm_phase5_local_prep_manifest(resolved_phase5_path)
    neural_manifest = load_economic_wm_neural_architecture_manifest(
        resolved_neural_path
    )
    compositions, joins, windows = _load_phase5_row_families(phase5_manifest)

    dataset_contract = _build_dataset_contract(
        phase5_manifest=phase5_manifest,
        compositions=compositions,
        joins=joins,
        windows=windows,
        neural_manifest=neural_manifest,
    )
    model_config = _build_model_component_config(
        dataset_contract=dataset_contract,
        neural_manifest=neural_manifest,
    )
    loss_definitions = _build_loss_definitions(model_config)
    smoke_forward = _build_cpu_smoke_forward_report(
        dataset_contract=dataset_contract,
        model_config=model_config,
        compositions=compositions,
        windows=windows,
    )

    dataset_contract_path = output_root / "economic_wm_trainer_dataset_contract_v1.json"
    model_config_path = (
        output_root / "economic_wm_trainer_model_component_config_v1.json"
    )
    loss_definitions_path = output_root / "economic_wm_trainer_loss_definitions_v1.json"
    smoke_forward_path = output_root / "economic_wm_trainer_cpu_smoke_forward_v1.json"
    manifest_path = output_root / "economic_wm_trainer_scaffold_manifest_v1.json"
    markdown_path = output_root / "economic_wm_trainer_scaffold_v1.md"

    artifact_refs = {
        "phase5_prep_path": str(resolved_phase5_path),
        "neural_manifest_path": str(resolved_neural_path),
        "dataset_contract_path": str(dataset_contract_path),
        "model_component_config_path": str(model_config_path),
        "loss_definitions_path": str(loss_definitions_path),
        "cpu_smoke_forward_path": str(smoke_forward_path),
        "manifest_path": str(manifest_path),
        "markdown_path": str(markdown_path),
    }
    manifest = _trainer_manifest(
        phase5_manifest=phase5_manifest,
        neural_manifest=neural_manifest,
        dataset_contract=dataset_contract,
        model_config=model_config,
        loss_definitions=loss_definitions,
        smoke_forward=smoke_forward,
        artifact_refs=artifact_refs,
    )
    _write_json(dataset_contract_path, dataset_contract)
    _write_json(model_config_path, model_config)
    _write_json(loss_definitions_path, loss_definitions)
    _write_json(smoke_forward_path, smoke_forward)
    _write_json(manifest_path, manifest)
    _write_markdown(markdown_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/economic_world_model/economic_wm_trainer_scaffold",
        help="Directory for trainer scaffold artifacts.",
    )
    parser.add_argument("--phase5-prep", default=None)
    parser.add_argument("--neural-manifest", default=None)
    parser.add_argument(
        "--no-run-dependencies",
        action="store_true",
        help="Do not run Phase-5 prep or neural manifest builders if missing.",
    )
    args = parser.parse_args()
    payload = run_train_economic_world_model_v0_scaffold(
        output_dir=args.output_dir,
        phase5_prep_path=args.phase5_prep,
        neural_manifest_path=args.neural_manifest,
        run_dependencies_if_missing=not args.no_run_dependencies,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if payload["authority_class"] == "trainer_scaffold_only"
        and payload["dataset_contract_ready"]
        and payload["cpu_smoke_forward_passed"]
        and payload["losses_defined"]
        and not payload["training_executed"]
        and not payload["weights_written"]
        and not payload["promotion_eligible"]
        and not payload["reward_math_mutation"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

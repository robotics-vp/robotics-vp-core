import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.train_sim_synth_backend_selector import _run_training
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality


pytest.importorskip("torch")


def _write_receipt_bundles(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundles = []
    for idx in range(6):
        bundles.append(
            {
                "bundle_id": f"backend_bundle_{idx}",
                "world_state": {
                    "state_id": f"state_{idx}",
                    "simulation_agenda": {
                        "jobs": [
                            {
                                "job_id": f"job_{idx}",
                                "coverage_gap_score": 0.35 + (0.03 * (idx % 3)),
                                "economic_priority": 0.45 + (0.05 * (idx % 4)),
                                "trust_priority": 0.2 + (0.02 * (idx % 4)),
                                "readiness": 0.3 + (0.05 * (idx % 4)),
                                "data_collection_intent": ["explore", "exploit", "validate"][idx % 3],
                                "risk_family": "collision" if idx % 2 == 0 else "",
                                "object_family": "drawer" if idx % 2 == 1 else "",
                            }
                        ]
                    },
                    "physics_context": {
                        "backend": ["pybullet", "isaac", "holosoma"][idx % 3],
                        "fidelity_tier": ["fast_scan", "branch_balanced", "high_fidelity"][idx % 3],
                        "domain_randomization_regime": [
                            "steady_state",
                            "coverage_exploration",
                            "benchmark_focus",
                        ][idx % 3],
                        "metadata": {
                            "heuristic_backend": "pybullet",
                            "heuristic_fidelity_tier": "branch_balanced",
                            "heuristic_domain_randomization_regime": "coverage_exploration",
                            "backend_helper_status": {"promotion_stage": "shadow_candidate"},
                            "benchmark_signals": {"ready": idx % 2 == 0},
                        },
                    },
                },
                "physics_calibration_receipt": {
                    "receipt_id": f"calibration_{idx}",
                    "backend": ["isaac", "holosoma", "pybullet"][idx % 3],
                    "fidelity_tier": ["high_fidelity", "branch_balanced", "fast_scan"][idx % 3],
                    "domain_randomization_regime": [
                        "benchmark_focus",
                        "coverage_exploration",
                        "steady_state",
                    ][idx % 3],
                    "quality_score": 0.7 + (0.03 * idx),
                },
                "runtime_receipt_manifest": {
                    "version": "sim_synth_runtime_receipt_manifest_v1",
                    "manifest_id": f"runtime_manifest_backend_{idx}",
                    "manifest_status": "complete",
                    "receipt_family_counts": {"physics_calibration_receipt_v1": 1},
                    "missing_required_families": [],
                },
            }
        )
    path.write_text(json.dumps({"bundles": bundles}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_live_receipt_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    world_state = {
        "state_id": "live_state_backend",
        "simulation_agenda": {
            "jobs": [
                {
                    "job_id": "live_job_backend",
                    "coverage_gap_score": 0.41,
                    "economic_priority": 0.58,
                    "trust_priority": 0.26,
                    "readiness": 0.61,
                    "data_collection_intent": "validate",
                }
            ]
        },
        "physics_context": {
            "backend": "pybullet",
            "fidelity_tier": "branch_balanced",
            "domain_randomization_regime": "coverage_exploration",
            "metadata": {
                "heuristic_backend": "pybullet",
                "heuristic_fidelity_tier": "branch_balanced",
                "heuristic_domain_randomization_regime": "coverage_exploration",
                "backend_helper_status": {"promotion_stage": "shadow_candidate"},
                "benchmark_signals": {"ready": False},
            },
        },
        "version": "sim_synth_physics_world_state_v1",
    }
    calibration = {
        "receipt_id": "live_calibration_backend",
        "backend": "isaac",
        "fidelity_tier": "high_fidelity",
        "calibration_profile": "default",
        "quality_score": 0.82,
        "metadata": {},
        "version": "physics_calibration_receipt_v1",
    }
    manifest = {
        "version": "sim_synth_runtime_receipt_manifest_v1",
        "manifest_id": "live_runtime_manifest_backend",
        "manifest_status": "complete",
        "receipt_family_counts": {"physics_calibration_receipt_v1": 1},
        "missing_required_families": [],
    }
    (path / "episode_sim_synth_world_state_v1.json").write_text(
        json.dumps(world_state, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (path / "episode_physics_calibration_receipt_v1.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (path / "runtime_receipt_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(8):
        rows.append(
            {
                "jobs": [
                    {
                        "coverage_gap_score": 0.2 + (0.05 * (idx % 3)),
                        "economic_priority": 0.4 + (0.05 * (idx % 4)),
                        "trust_priority": 0.2 + (0.03 * (idx % 5)),
                        "readiness": 0.3 + (0.04 * (idx % 4)),
                        "data_collection_intent": ["explore", "exploit", "validate"][idx % 3],
                        "risk_family": "collision" if idx % 2 == 0 else "",
                        "object_family": "drawer" if idx % 2 == 1 else "",
                    }
                ],
                "benchmark_signals": {"ready": idx % 4 == 0},
                "heuristic_backend": ["pybullet", "isaac", "holosoma", "other"][idx % 4],
                "heuristic_fidelity_tier": ["fast_scan", "branch_balanced", "high_fidelity"][idx % 3],
                "heuristic_domain_randomization_regime": [
                    "steady_state",
                    "coverage_exploration",
                    "calibration_focus",
                    "benchmark_focus",
                ][idx % 4],
                "target_backend": ["pybullet", "isaac", "holosoma", "pybullet"][idx % 4],
                "target_fidelity_tier": ["fast_scan", "branch_balanced", "high_fidelity"][idx % 3],
                "target_domain_randomization_regime": [
                    "steady_state",
                    "coverage_exploration",
                    "calibration_focus",
                    "benchmark_focus",
                ][idx % 4],
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return path


def test_train_sim_synth_backend_selector_emits_runtime_package(tmp_path: Path) -> None:
    receipt_path = _write_receipt_bundles(tmp_path / "backend_selector_receipts.json")
    output_dir = tmp_path / "results"
    args = Namespace(
        dataset=None,
        receipt_path=str(receipt_path),
        epochs=2,
        lr=1e-3,
        hidden_dim=16,
        save_dir=str(output_dir),
        run_name="sim_synth_backend_selector_unit",
        seed=19,
        skip_regal_runner=True,
    )

    result = _run_training(args, runner=None)

    assert Path(result["checkpoint"]).exists()
    assert Path(result["compiled_dataset"]).exists()
    assert Path(result["dataset_summary"]).exists()
    assert Path(result["model_config"]).exists()
    assert Path(result["preconditions"]).exists()
    assert Path(result["training_summary"]).exists()
    assert Path(result["runtime_package"]).exists()
    assert (output_dir / "training_job_result.json").exists()
    assert result["benchmark_gate_ready"] is False

    runtime_package = json.loads(Path(result["runtime_package"]).read_text(encoding="utf-8"))
    dataset_summary = json.loads(Path(result["dataset_summary"]).read_text(encoding="utf-8"))
    assert runtime_package["promotion_stage"] == "shadow_candidate"
    assert runtime_package["inference_contract"]["helper_blend_policy"] == "bounded_backend_selector_helper_v1"
    assert runtime_package["metadata"]["target_hardware_class"] == "unitree_g1_r1_class"
    assert runtime_package["checkpoint_path"] == "sim_synth_backend_selector.pt"
    assert dataset_summary["runtime_receipt_rows"] == 6
    assert dataset_summary["source_row_count"] == 6
    assert dataset_summary["admissibility_summary"]["positive_training_row_count"] == 6
    assert dataset_summary["admissibility_summary"]["excluded_row_count"] == 0
    assert dataset_summary["input_sources"]["receipt_path"] == str(receipt_path)


def test_train_sim_synth_backend_selector_runner_emits_runtime_manifest(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "backend_selector_rows.jsonl")
    output_dir = tmp_path / "runner_results"
    args = Namespace(
        dataset=str(dataset_path),
        receipt_path=None,
        epochs=2,
        lr=1e-3,
        hidden_dim=16,
        save_dir=str(output_dir),
        run_name="sim_synth_backend_selector_runner",
        seed=23,
        skip_regal_runner=False,
    )

    holder = {}

    def _wrapped(runner) -> None:
        holder["payload"] = _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=23,
            num_episodes=1,
            training_steps=2,
            fail_on_verify_error=False,
        ),
        plan_sha="plan_sha",
        plan_id="sim_synth_backend_selector_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    checkpoint_registry = json.loads((output_dir / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert manifest["training_kind"] == "sim_synth_backend_selector"
    assert manifest["artifact_paths"]["sim_synth_backend_selector_runtime_package"].endswith(
        "sim_synth_backend_selector_package.json"
    )
    assert checkpoint_registry["checkpoints"][0]["model_family"] == "sim_synth_backend_selector"
    assert holder["payload"]["benchmark_gate_ready"] is False
    assert holder["payload"]["admissibility_summary"]["legacy_dataset_row_count"] == 8


def test_train_sim_synth_backend_selector_harvests_receipt_dir(tmp_path: Path) -> None:
    receipt_dir = _write_live_receipt_dir(tmp_path / "live_receipts")
    output_dir = tmp_path / "harvest_results"
    args = Namespace(
        dataset=None,
        receipt_path=None,
        receipt_dir=[str(receipt_dir)],
        epochs=2,
        lr=1e-3,
        hidden_dim=16,
        save_dir=str(output_dir),
        run_name="sim_synth_backend_selector_harvest",
        seed=37,
        skip_regal_runner=True,
    )

    result = _run_training(args, runner=None)
    dataset_summary = json.loads(Path(result["dataset_summary"]).read_text(encoding="utf-8"))

    assert dataset_summary["input_sources"]["receipt_source_kind"] == "harvested_runtime_receipts"
    assert dataset_summary["input_sources"]["receipt_dirs"] == [str(receipt_dir)]
    assert dataset_summary["input_sources"]["receipt_bundle_count"] == 1
    assert dataset_summary["admissibility_summary"]["positive_training_row_count"] == 1
    assert dataset_summary["admissibility_summary"]["excluded_row_count"] == 0

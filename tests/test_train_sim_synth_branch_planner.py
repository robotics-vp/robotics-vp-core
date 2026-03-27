import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.train_sim_synth_branch_planner import _run_training
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality


pytest.importorskip("torch")


def _write_receipt_bundles(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundles = []
    for idx in range(6):
        bundles.append(
            {
                "bundle_id": f"branch_bundle_{idx}",
                "world_state": {
                    "state_id": f"branch_state_{idx}",
                    "simulation_agenda": {
                        "jobs": [
                            {
                                "job_id": f"job_{idx}",
                                "coverage_gap_score": 0.32 + (0.04 * (idx % 4)),
                                "economic_priority": 0.45 + (0.03 * (idx % 5)),
                                "trust_priority": 0.2 + (0.02 * (idx % 4)),
                                "readiness": 0.35 + (0.05 * (idx % 4)),
                                "data_collection_intent": ["explore", "exploit", "validate"][idx % 3],
                                "risk_family": "collision" if idx % 2 == 0 else "",
                                "object_family": "drawer" if idx % 2 == 1 else "",
                                "objective_preset": ["balanced", "throughput", "safety"][idx % 3],
                            }
                        ]
                    },
                    "physics_context": {
                        "backend": ["pybullet", "isaac", "holosoma"][idx % 3],
                        "fidelity_tier": ["fast_scan", "branch_balanced", "high_fidelity"][idx % 3],
                    },
                    "gen2sim_admission": {
                        "metadata": {"benchmark_signals": {"ready": idx % 2 == 0}}
                    },
                    "synthetic_branch_plans": [
                        {
                            "plan_id": f"plan_{idx}",
                            "source_job_id": f"job_{idx}",
                            "generation_mode": [
                                "coverage_branch",
                                "targeted_synth_rollout",
                                "physics_probe",
                                "geometry_guarded_rollout",
                                "neural_branch_candidate",
                            ][idx % 5],
                            "expected_yield_score": 0.4 + (0.05 * (idx % 5)),
                            "metadata": {
                                "heuristic_generation_mode": [
                                    "coverage_branch",
                                    "targeted_synth_rollout",
                                    "physics_probe",
                                    "geometry_guarded_rollout",
                                    "neural_branch_candidate",
                                ][(idx + 1) % 5],
                                "branch_helper_status": {"promotion_stage": "shadow_candidate"},
                            },
                        }
                    ],
                },
                "simulation_outcome_receipts": [
                    {
                        "receipt_id": f"outcome_{idx}",
                        "branch_plan_id": f"plan_{idx}",
                        "status": "completed" if idx % 2 == 0 else "failed",
                        "metadata": {
                            "realized_yield_score": 0.75 if idx % 2 == 0 else 0.15,
                            "executed_generation_mode": [
                                "coverage_branch",
                                "targeted_synth_rollout",
                                "physics_probe",
                                "geometry_guarded_rollout",
                                "neural_branch_candidate",
                            ][idx % 5],
                        },
                    }
                ],
            }
        )
    path.write_text(json.dumps({"bundles": bundles}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _write_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(10):
        rows.append(
            {
                "job": {
                    "coverage_gap_score": 0.3 + (0.04 * (idx % 4)),
                    "economic_priority": 0.45 + (0.03 * (idx % 5)),
                    "trust_priority": 0.2 + (0.02 * (idx % 4)),
                    "readiness": 0.35 + (0.05 * (idx % 4)),
                    "data_collection_intent": ["explore", "exploit", "validate"][idx % 3],
                    "risk_family": "collision" if idx % 2 == 0 else "",
                    "object_family": "drawer" if idx % 2 == 1 else "",
                    "objective_preset": ["balanced", "throughput", "safety", "energy_saver"][idx % 4],
                },
                "context": {
                    "heuristic_generation_mode": [
                        "coverage_branch",
                        "targeted_synth_rollout",
                        "physics_probe",
                        "geometry_guarded_rollout",
                        "neural_branch_candidate",
                    ][idx % 5],
                    "physics_context": {
                        "backend": ["pybullet", "isaac", "holosoma", "other"][idx % 4],
                        "fidelity_tier": ["fast_scan", "branch_balanced", "high_fidelity"][idx % 3],
                    },
                },
                "target_generation_mode": [
                    "coverage_branch",
                    "targeted_synth_rollout",
                    "physics_probe",
                    "geometry_guarded_rollout",
                    "neural_branch_candidate",
                ][idx % 5],
                "target_expected_yield_score": 0.35 + (0.05 * (idx % 6)),
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    return path


def test_train_sim_synth_branch_planner_emits_runtime_package(tmp_path: Path) -> None:
    receipt_path = _write_receipt_bundles(tmp_path / "branch_planner_receipts.json")
    output_dir = tmp_path / "results"
    args = Namespace(
        dataset=None,
        receipt_path=str(receipt_path),
        epochs=2,
        lr=1e-3,
        hidden_dim=16,
        save_dir=str(output_dir),
        run_name="sim_synth_branch_planner_unit",
        seed=29,
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
    assert runtime_package["inference_contract"]["helper_blend_policy"] == "bounded_branch_planner_helper_v1"
    assert runtime_package["metadata"]["target_hardware_class"] == "unitree_g1_r1_class"
    assert runtime_package["checkpoint_path"] == "sim_synth_branch_planner.pt"
    assert dataset_summary["runtime_receipt_rows"] == 6
    assert dataset_summary["input_sources"]["receipt_path"] == str(receipt_path)


def test_train_sim_synth_branch_planner_runner_emits_runtime_manifest(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "branch_planner_rows.jsonl")
    output_dir = tmp_path / "runner_results"
    args = Namespace(
        dataset=str(dataset_path),
        receipt_path=None,
        epochs=2,
        lr=1e-3,
        hidden_dim=16,
        save_dir=str(output_dir),
        run_name="sim_synth_branch_planner_runner",
        seed=31,
        skip_regal_runner=False,
    )

    holder = {}

    def _wrapped(runner) -> None:
        holder["payload"] = _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=31,
            num_episodes=1,
            training_steps=2,
            fail_on_verify_error=False,
        ),
        plan_sha="plan_sha",
        plan_id="sim_synth_branch_planner_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    checkpoint_registry = json.loads((output_dir / "checkpoint_registry.json").read_text(encoding="utf-8"))

    assert manifest["training_kind"] == "sim_synth_branch_planner"
    assert manifest["artifact_paths"]["sim_synth_branch_planner_runtime_package"].endswith(
        "sim_synth_branch_planner_package.json"
    )
    assert checkpoint_registry["checkpoints"][0]["model_family"] == "sim_synth_branch_planner"
    assert holder["payload"]["benchmark_gate_ready"] is False

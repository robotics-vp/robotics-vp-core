from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts.train_sim_synth_backend_selector import _run_training
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality
from src.utils.config_digest import sha256_json
from src.world_model.sim_synth_physics import compile_sim_synth_physics_world_state
from src.world_model.semantic_coverage_graph import CoverageEdge, CoverageNode, SemanticCoverageGraph


def _backend_rows() -> list[dict]:
    return [
        {
            "jobs": [
                {
                    "coverage_gap_score": 0.9,
                    "economic_priority": 0.85,
                    "trust_priority": 0.4,
                    "readiness": 0.7,
                    "data_collection_intent": "exploit",
                    "risk_family": "",
                    "object_family": "drawer",
                }
            ],
            "heuristic_backend": "pybullet",
            "heuristic_fidelity_tier": "branch_balanced",
            "heuristic_domain_randomization_regime": "coverage_exploration",
            "benchmark_signals": {"ready": False, "benchmark_eligible": False},
            "target_backend": "isaac",
            "target_fidelity_tier": "high_fidelity",
            "target_domain_randomization_regime": "benchmark_focus",
        },
        {
            "jobs": [
                {
                    "coverage_gap_score": 0.4,
                    "economic_priority": 0.3,
                    "trust_priority": 0.7,
                    "readiness": 0.5,
                    "data_collection_intent": "explore",
                    "risk_family": "collision",
                    "object_family": "",
                }
            ],
            "heuristic_backend": "pybullet",
            "heuristic_fidelity_tier": "fast_scan",
            "heuristic_domain_randomization_regime": "steady_state",
            "benchmark_signals": {"ready": False, "benchmark_eligible": False},
            "target_backend": "pybullet",
            "target_fidelity_tier": "fast_scan",
            "target_domain_randomization_regime": "coverage_exploration",
        },
        {
            "jobs": [
                {
                    "coverage_gap_score": 0.75,
                    "economic_priority": 0.8,
                    "trust_priority": 0.35,
                    "readiness": 0.65,
                    "data_collection_intent": "validate",
                    "risk_family": "precision",
                    "object_family": "",
                }
            ],
            "heuristic_backend": "holosoma",
            "heuristic_fidelity_tier": "high_fidelity",
            "heuristic_domain_randomization_regime": "calibration_focus",
            "benchmark_signals": {"ready": True, "benchmark_eligible": True},
            "target_backend": "holosoma",
            "target_fidelity_tier": "high_fidelity",
            "target_domain_randomization_regime": "calibration_focus",
        },
    ]


def _write_dataset(path: Path) -> Path:
    rows = _backend_rows()
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _graph() -> SemanticCoverageGraph:
    return SemanticCoverageGraph(
        nodes=[
            CoverageNode("task:drawer_vase", "task", "drawer_vase"),
            CoverageNode("hrl:grasp_handle", "skill", "Grasp Handle"),
            CoverageNode("prim:locate_handle", "env_primitive", "Locate Handle"),
        ],
        edges=[
            CoverageEdge(
                "hrl:grasp_handle",
                "prim:locate_handle",
                "requires",
                evidence_count=0,
                economic_priority=0.8,
                trust_priority=0.4,
                promotion_readiness=0.6,
            )
        ],
    )


def test_train_sim_synth_backend_selector_emits_runtime_package(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "backend_rows.jsonl")
    output_dir = tmp_path / "out"
    args = type(
        "Args",
        (),
        {
            "dataset": str(dataset_path),
            "epochs": 1,
            "lr": 1e-3,
            "hidden_dim": 8,
            "save_dir": str(output_dir),
            "run_name": "backend_selector_unit",
            "seed": 7,
            "skip_regal_runner": True,
        },
    )()

    result = _run_training(args, runner=None)

    runtime_package = json.loads(Path(result["runtime_package"]).read_text(encoding="utf-8"))
    assert Path(result["checkpoint"]).exists()
    assert runtime_package["promotion_stage"] == "shadow_candidate"
    assert runtime_package["inference_contract"]["helper_blend_policy"] == "bounded_backend_selector_helper_v1"

    world_state = compile_sim_synth_physics_world_state(
        _graph(),
        backend_selector=result["runtime_package"],
        backend_selector_mode="auto",
    )
    helper_status = world_state.physics_context.metadata["backend_helper_status"]
    assert helper_status["status"] == "loaded"
    assert helper_status["package_path"].endswith("sim_synth_backend_selector_package.json")
    assert world_state.physics_context.selection_policy == "heuristic_plus_learned_backend_selector"


def test_train_sim_synth_backend_selector_runner_emits_runtime_manifest(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "backend_rows.jsonl")
    output_dir = tmp_path / "runner"

    class Args:
        dataset = str(dataset_path)
        epochs = 1
        lr = 1e-3
        hidden_dim = 8
        save_dir = str(output_dir)
        run_name = "backend_selector_runner"
        seed = 11
        skip_regal_runner = False

    holder = {}

    def _wrapped(runner) -> None:
        holder["payload"] = _run_training(Args(), runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=11,
            num_episodes=1,
            training_steps=1,
            fail_on_verify_error=False,
        ),
        plan_sha=sha256_json({"plan": "backend_selector_test"}),
        plan_id="backend_selector_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_kind"] == "sim_synth_backend_selector"
    assert manifest["artifact_paths"]["sim_synth_backend_selector_runtime_package"].endswith(
        "sim_synth_backend_selector_package.json"
    )
    assert holder["payload"]["benchmark_gate_ready"] is False


def test_backend_selector_required_mode_rejects_shadow_package(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "backend_rows.jsonl")
    args = type(
        "Args",
        (),
        {
            "dataset": str(dataset_path),
            "epochs": 1,
            "lr": 1e-3,
            "hidden_dim": 8,
            "save_dir": str(tmp_path / "out"),
            "run_name": "backend_selector_required",
            "seed": 17,
            "skip_regal_runner": True,
        },
    )()
    result = _run_training(args, runner=None)

    with pytest.raises(ValueError, match="required"):
        compile_sim_synth_physics_world_state(
            _graph(),
            backend_selector=result["runtime_package"],
            backend_selector_mode="required",
        )

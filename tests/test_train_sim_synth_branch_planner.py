from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from scripts.train_sim_synth_branch_planner import _run_training
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality
from src.utils.config_digest import sha256_json
from src.world_model.semantic_coverage_graph import CoverageEdge, CoverageNode, SemanticCoverageGraph
from src.world_model.sim_synth_physics import compile_sim_synth_physics_world_state


def _branch_rows() -> list[dict]:
    return [
        {
            "job": {
                "coverage_gap_score": 0.85,
                "economic_priority": 0.8,
                "trust_priority": 0.35,
                "readiness": 0.55,
                "data_collection_intent": "exploit",
                "risk_family": "",
                "object_family": "drawer",
                "objective_preset": "balanced",
            },
            "context": {
                "physics_context": {"backend": "pybullet", "fidelity_tier": "branch_balanced"},
                "heuristic_generation_mode": "targeted_synth_rollout",
            },
            "target_generation_mode": "targeted_synth_rollout",
            "target_expected_yield_score": 0.8,
        },
        {
            "job": {
                "coverage_gap_score": 0.45,
                "economic_priority": 0.3,
                "trust_priority": 0.8,
                "readiness": 0.7,
                "data_collection_intent": "validate",
                "risk_family": "collision",
                "object_family": "",
                "objective_preset": "safety",
            },
            "context": {
                "physics_context": {"backend": "isaac", "fidelity_tier": "high_fidelity"},
                "heuristic_generation_mode": "physics_probe",
            },
            "target_generation_mode": "physics_probe",
            "target_expected_yield_score": 0.65,
        },
        {
            "job": {
                "coverage_gap_score": 0.6,
                "economic_priority": 0.55,
                "trust_priority": 0.45,
                "readiness": 0.5,
                "data_collection_intent": "explore",
                "risk_family": "",
                "object_family": "",
                "objective_preset": "throughput",
            },
            "context": {
                "physics_context": {"backend": "holosoma", "fidelity_tier": "fast_scan"},
                "heuristic_generation_mode": "coverage_branch",
            },
            "target_generation_mode": "coverage_branch",
            "target_expected_yield_score": 0.4,
        },
    ]


def _write_dataset(path: Path) -> Path:
    rows = _branch_rows()
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return path


def _graph() -> SemanticCoverageGraph:
    return SemanticCoverageGraph(
        nodes=[
            CoverageNode("task:drawer_vase", "task", "drawer_vase"),
            CoverageNode("hrl:grasp_handle", "skill", "Grasp Handle"),
            CoverageNode("risk:collision", "risk_family", "collision"),
        ],
        edges=[
            CoverageEdge(
                "hrl:grasp_handle",
                "risk:collision",
                "requires",
                evidence_count=0,
                economic_priority=0.4,
                trust_priority=0.2,
                promotion_readiness=0.9,
            )
        ],
    )


def test_train_sim_synth_branch_planner_emits_runtime_package(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "branch_rows.jsonl")
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
            "run_name": "branch_planner_unit",
            "seed": 5,
            "skip_regal_runner": True,
        },
    )()

    result = _run_training(args, runner=None)

    runtime_package = json.loads(Path(result["runtime_package"]).read_text(encoding="utf-8"))
    assert Path(result["checkpoint"]).exists()
    assert runtime_package["promotion_stage"] == "shadow_candidate"
    assert runtime_package["inference_contract"]["helper_blend_policy"] == "bounded_branch_planner_helper_v1"

    world_state = compile_sim_synth_physics_world_state(
        _graph(),
        branch_planner=result["runtime_package"],
        branch_planner_mode="auto",
    )
    helper_status = world_state.synthetic_branch_plans[0].metadata["branch_helper_status"]
    assert helper_status["status"] == "loaded"
    assert helper_status["package_path"].endswith("sim_synth_branch_planner_package.json")
    assert world_state.synthetic_branch_plans[0].selection_policy == "heuristic_plus_learned_branch_planner"


def test_train_sim_synth_branch_planner_runner_emits_runtime_manifest(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "branch_rows.jsonl")
    output_dir = tmp_path / "runner"

    class Args:
        dataset = str(dataset_path)
        epochs = 1
        lr = 1e-3
        hidden_dim = 8
        save_dir = str(output_dir)
        run_name = "branch_planner_runner"
        seed = 13
        skip_regal_runner = False

    holder = {}

    def _wrapped(runner) -> None:
        holder["payload"] = _run_training(Args(), runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=13,
            num_episodes=1,
            training_steps=1,
            fail_on_verify_error=False,
        ),
        plan_sha=sha256_json({"plan": "branch_planner_test"}),
        plan_id="branch_planner_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_kind"] == "sim_synth_branch_planner"
    assert manifest["artifact_paths"]["sim_synth_branch_planner_runtime_package"].endswith(
        "sim_synth_branch_planner_package.json"
    )
    assert holder["payload"]["benchmark_gate_ready"] is False


def test_branch_planner_required_mode_rejects_shadow_package(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path / "branch_rows.jsonl")
    args = type(
        "Args",
        (),
        {
            "dataset": str(dataset_path),
            "epochs": 1,
            "lr": 1e-3,
            "hidden_dim": 8,
            "save_dir": str(tmp_path / "out"),
            "run_name": "branch_planner_required",
            "seed": 19,
            "skip_regal_runner": True,
        },
    )()
    result = _run_training(args, runner=None)

    with pytest.raises(ValueError, match="required"):
        compile_sim_synth_physics_world_state(
            _graph(),
            branch_planner=result["runtime_package"],
            branch_planner_mode="required",
        )

from __future__ import annotations

import json
from pathlib import Path

from scripts.train_semantic_feedback_adapters import _run_training, parse_args
from src.envs.primitive_inventory import for_env
from src.hrl.skill_graph import SkillGraph
from src.orchestrator.coverage_loop import run_coverage_loop
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality
from src.utils.config_digest import sha256_json
from src.world_model.semantic_coverage_graph import SemanticCoverageGraph


def _coverage_graph_path(tmp_path: Path) -> Path:
    graph = SemanticCoverageGraph.build(
        skill_graph=SkillGraph.build_from_registry(hrl_skills=True),
        env_inventories=[for_env("drawer_vase")],
    )
    for index, edge in enumerate(graph.edges[:12]):
        edge.economic_priority = 0.2 + (0.03 * index)
        edge.trust_priority = 0.25 + (0.03 * index)
        edge.promotion_readiness = 0.3 + (0.03 * index)
        edge.metadata["quality_score"] = 0.4 + (0.02 * index)
        edge.metadata["process_reward_delta"] = 0.15
        edge.metadata["policy_eval_delta"] = 0.08
        edge.metadata["backend_health_score"] = 0.85
        edge.metadata["wm_validation_pressure"] = 0.2 if index % 2 == 0 else 0.0
    path = tmp_path / "coverage_graph.json"
    path.write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")
    return path


def test_train_semantic_feedback_adapters_emits_runtime_package(tmp_path: Path) -> None:
    graph_path = _coverage_graph_path(tmp_path)
    args = parse_args(
        [
            "--coverage-graph",
            str(graph_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-regal-runner",
            "--epochs",
            "2",
        ]
    )

    result = _run_training(args, runner=None)
    package = json.loads(Path(result["runtime_package"]).read_text(encoding="utf-8"))

    assert package["promotion_stage"] == "shadow_candidate"
    assert package["inference_contract"]["target_contract"] == "semantic_feedback_adapter_v1"


def test_regality_wrapper_registers_semantic_feedback_adapter_artifacts(tmp_path: Path) -> None:
    graph_path = _coverage_graph_path(tmp_path)
    output_dir = tmp_path / "runner"

    def _wrapped(runner) -> None:
        args = parse_args(
            [
                "--coverage-graph",
                str(graph_path),
                "--output-dir",
                str(output_dir),
                "--epochs",
                "2",
            ]
        )
        _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=7,
            num_episodes=1,
            training_steps=2,
            fail_on_verify_error=False,
        ),
        plan_sha=sha256_json({"plan": "semantic_feedback_adapter_test"}),
        plan_id="semantic_feedback_adapter_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_kind"] == "semantic_feedback_adapters"
    assert manifest["artifact_paths"]["semantic_feedback_adapter_runtime_package"].endswith(
        "semantic_feedback_adapter_runtime_package.json"
    )


def test_coverage_loop_loads_semantic_feedback_adapter_runtime_package(tmp_path: Path) -> None:
    graph_path = _coverage_graph_path(tmp_path)
    args = parse_args(
        [
            "--coverage-graph",
            str(graph_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-regal-runner",
            "--epochs",
            "2",
        ]
    )
    result = _run_training(args, runner=None)

    coverage = run_coverage_loop(
        [
            {
                "task_id": "open_drawer",
                "env_id": "drawer_vase",
                "semantic_tokens": ["skill:locate_drawer", "skill:grasp_handle"],
            }
        ],
        feedback_adapter_package=result["runtime_package"],
        feedback_adapter_mode="auto",
        shadow_fit_feedback_adapter=False,
    )

    helper_status = coverage.coverage_summary["feedback_loop"]["feedback_adapter_helper_status"]
    assert helper_status["status"] == "loaded"

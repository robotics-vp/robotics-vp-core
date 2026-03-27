from __future__ import annotations

import json
from pathlib import Path

from scripts.train_semantic_wm_refiner import _run_training, parse_args
from src.orchestrator.coverage_loop import run_coverage_loop
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality
from src.utils.config_digest import sha256_json
from src.world_model.semantic_feedback_packets import GraphMutationProposal
from src.world_model.semantic_wm_correction import SemanticWMCorrectionOverlay
from src.world_model.semantic_world_model import (
    SemanticMetaNode,
    SemanticObjectState,
    SemanticRelationState,
    SemanticWorldModelState,
)


def _world_model() -> SemanticWorldModelState:
    return SemanticWorldModelState(
        world_model_id="wm_refiner_test",
        episode_id="episode_test",
        task_id="drawer_vase",
        objective_preset="balanced",
        semantic_tags=["drawer", "vase", "risk:fragility"],
        objects=[
            SemanticObjectState(
                object_id="object_drawer",
                label="drawer",
                category="container",
                confidence=0.9,
                salience=0.8,
                affordances=["open", "grasp_handle"],
                track_refs=["track_1"],
            ),
            SemanticObjectState(
                object_id="object_vase",
                label="vase",
                category="fragile_object",
                confidence=0.92,
                salience=0.95,
                affordances=["avoid_contact"],
                risk_tags=["fragility"],
                track_refs=["track_2"],
            ),
        ],
        relations=[
            SemanticRelationState(
                relation_id="rel_drawer_vase",
                subject_id="object_drawer",
                relation_type="near",
                object_id="object_vase",
                confidence=0.7,
            )
        ],
        meta_nodes=[
            SemanticMetaNode(
                node_id="meta_ontology",
                node_type="ontology_router",
                priority="high",
                score=0.72,
                rationale="novel semantics detected",
                target_refs=["skill:novel_recovery"],
            )
        ],
        capability_scores={
            "object_memory": 0.8,
            "affordance_grounding": 0.72,
            "meta_node_orchestration": 0.6,
            "risk_reasoning": 0.68,
        },
        topology={"grounded_track_object_count": 2},
    )


def _artifact_dir(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "coverage_artifact"
    artifact_dir.mkdir()
    (artifact_dir / "input_semantic_world_model.json").write_text(
        json.dumps(_world_model().to_dict()),
        encoding="utf-8",
    )
    (artifact_dir / "semantic_wm_correction_overlay.json").write_text(
        json.dumps(
            SemanticWMCorrectionOverlay(
                object_confidence_adjustments={"object_vase": -0.15},
                relation_confidence_adjustments={"rel_drawer_vase": -0.1},
                capability_adjustments={"object_memory": -0.08},
                meta_node_pressure=0.5,
                metadata={"packet_count": 1},
            ).to_dict()
        ),
        encoding="utf-8",
    )
    (artifact_dir / "coverage_feedback_summary.json").write_text(
        json.dumps({"gap_return_mean": 0.5, "process_reward_mean": 0.2}),
        encoding="utf-8",
    )
    (artifact_dir / "graph_mutation_proposals.json").write_text(
        json.dumps(
            [
                GraphMutationProposal(
                    proposal_id="p1",
                    action="add_provisional_skill",
                    target_ref="skill:novel_recovery",
                    confidence=0.82,
                    rationale="novel recovery",
                ).to_dict()
            ]
        ),
        encoding="utf-8",
    )
    (artifact_dir / "graph_mutation_execution.json").write_text(
        json.dumps({"records": [{"proposal_id": "p1", "status": "applied"}]}),
        encoding="utf-8",
    )
    (artifact_dir / "wm_validation_packets.json").write_text(
        json.dumps(
            [
                {
                    "target_ref": "object_vase",
                    "validation_kind": "state_mismatch",
                    "error_score": 0.8,
                }
            ]
        ),
        encoding="utf-8",
    )
    return artifact_dir


def test_train_semantic_wm_refiner_emits_runtime_package(tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    args = parse_args(
        [
            "--artifact-dir",
            str(artifact_dir),
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
    assert package["inference_contract"]["target_contract"] == "semantic_wm_refiner_v1"


def test_regality_wrapper_registers_semantic_wm_refiner_artifacts(tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    output_dir = tmp_path / "runner"

    def _wrapped(runner) -> None:
        args = parse_args(
            [
                "--artifact-dir",
                str(artifact_dir),
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
        plan_sha=sha256_json({"plan": "semantic_wm_refiner_test"}),
        plan_id="semantic_wm_refiner_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_kind"] == "semantic_wm_refiner"
    assert manifest["artifact_paths"]["semantic_wm_refiner_runtime_package"].endswith(
        "semantic_wm_refiner_runtime_package.json"
    )


def test_coverage_loop_loads_semantic_wm_refiner_runtime_package(tmp_path: Path) -> None:
    artifact_dir = _artifact_dir(tmp_path)
    args = parse_args(
        [
            "--artifact-dir",
            str(artifact_dir),
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
        semantic_world_model=_world_model().to_dict(),
        wm_validation_packets=[
            {
                "target_ref": "object_vase",
                "validation_kind": "state_mismatch",
                "error_score": 0.8,
            }
        ],
        semantic_wm_refiner_package=result["runtime_package"],
        semantic_wm_refiner_mode="auto",
        shadow_fit_semantic_wm_refiner=False,
    )

    helper_status = coverage.semantic_wm_refiner_summary["helper_status"]
    assert helper_status["status"] == "loaded"
    assert coverage.semantic_wm_refiner_summary["active"] is True

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from scripts.train_meta_transformer_synthetic import _run_training
from src.orchestrator.meta_transformer import MetaTransformer
from src.orchestrator.meta_transformer_training import (
    generate_meta_transformer_dataset,
    save_meta_transformer_dataset,
)
from src.world_model.semantic_world_model import (
    SemanticMetaNode,
    SemanticObjectState,
    SemanticRelationState,
    SemanticWorldModelState,
)


def _write_runtime_export(root: Path, sample_count: int = 8) -> Path:
    export_dir = root / "runtime_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = export_dir / "meta_transformer_runtime_dataset.json"
    summary_path = export_dir / "semantic_runtime_learning_summary.json"
    save_meta_transformer_dataset(generate_meta_transformer_dataset(sample_count), str(dataset_path))
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "semantic_runtime_learning_summary_v1",
                "total_rows": sample_count,
                "bounded_ready_count": sample_count,
                "semantic_grounded_count": sample_count,
                "route_success_count": sample_count // 2,
                "authority_success_count": sample_count // 2,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return export_dir


def _train_runtime_package(tmp_path: Path) -> str:
    export_dir = _write_runtime_export(tmp_path, sample_count=6)
    output_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"
    args = Namespace(
        runtime_export_dir=str(export_dir),
        dataset_json=None,
        runtime_summary_json=None,
        synthetic_samples=0,
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        run_name="meta_runtime",
        hidden_dim=32,
        num_heads=2,
        num_layers=1,
        max_semantic_tokens=8,
        batch_size=2,
        epochs=1,
        lr=1e-3,
        val_fraction=0.25,
        seed=17,
        skip_regal_runner=True,
    )
    return _run_training(args, runner=None)["runtime_package"]


def _semantic_world_model() -> SemanticWorldModelState:
    return SemanticWorldModelState(
        world_model_id="wm_meta_runtime",
        episode_id="episode_meta_runtime",
        task_id="drawer_vase_task",
        objective_preset="balanced",
        semantic_tags=["drawer", "vase", "fragile"],
        objects=[
            SemanticObjectState(
                object_id="drawer",
                label="drawer",
                category="container",
                confidence=0.92,
                salience=0.78,
                affordances=["open", "close"],
                state_tags=["occluding"],
                risk_tags=[],
            ),
            SemanticObjectState(
                object_id="vase",
                label="vase",
                category="fragile_object",
                confidence=0.95,
                salience=0.91,
                affordances=["avoid_contact"],
                state_tags=["fragile"],
                risk_tags=["fragility"],
            ),
        ],
        relations=[
            SemanticRelationState(
                relation_id="rel_meta_runtime",
                subject_id="drawer",
                relation_type="near",
                object_id="vase",
                confidence=0.7,
            )
        ],
        meta_nodes=[
            SemanticMetaNode(
                node_id="node_risk",
                node_type="risk_triage",
                priority="critical",
                score=0.84,
                rationale="fragile object present",
            ),
            SemanticMetaNode(
                node_id="node_efficiency",
                node_type="efficiency_router",
                priority="medium",
                score=0.36,
                rationale="energy still matters",
            ),
        ],
        capability_scores={
            "risk_reasoning": 0.81,
            "object_memory": 0.76,
            "affordance_grounding": 0.63,
            "fusion_bridge": 0.58,
            "stage2_bridge": 0.47,
            "meta_node_orchestration": 0.72,
        },
        topology={
            "grounded_track_object_count": 2,
            "object_count": 2,
            "relation_count": 1,
            "meta_node_count": 2,
        },
        metadata={"grounded_scene": {"grounding_mode": "scene_tracks"}},
    )


def test_meta_transformer_auto_mode_consumes_runtime_package(tmp_path: Path) -> None:
    runtime_package = _train_runtime_package(tmp_path)
    transformer = MetaTransformer(
        d_shared=24,
        helper_package_path=runtime_package,
        helper_mode="auto",
    )

    output = transformer.forward(
        dino_features=np.zeros(256, dtype=np.float32),
        vla_features=np.zeros(128, dtype=np.float32),
    )

    helper = output.metadata["learned_helper"]
    assert helper["status"] == "loaded"
    assert helper["promotion_stage"] == "shadow_candidate"
    assert helper["benchmark_gate_ready"] is False
    assert helper["helper_weight"] == pytest.approx(0.2)
    assert isinstance(helper["predicted_ontology_tokens"], list)
    assert helper["planning_heads_available"] is True
    assert "objective_preset" in helper
    assert "planning_trace" in helper
    assert "objective_distribution" in helper["planning_trace"]


def test_meta_transformer_required_mode_rejects_unready_package(tmp_path: Path) -> None:
    runtime_package = _train_runtime_package(tmp_path)
    transformer = MetaTransformer(
        d_shared=24,
        helper_package_path=runtime_package,
        helper_mode="required",
    )

    with pytest.raises(ValueError, match="benchmark-gated package"):
        transformer.forward(
            dino_features=np.zeros(256, dtype=np.float32),
            vla_features=np.zeros(128, dtype=np.float32),
        )


def test_meta_transformer_propose_plan_records_planning_application(tmp_path: Path) -> None:
    runtime_package = _train_runtime_package(tmp_path)
    transformer = MetaTransformer(
        d_shared=24,
        helper_package_path=runtime_package,
        helper_mode="auto",
    )

    output = transformer.propose_plan(
        econ_signals={"mpl_urgency": 0.2, "error_urgency": 0.75, "energy_urgency": 0.18},
        datapack_signals={
            "data_coverage_score": 0.54,
            "embedding_diversity": 0.42,
            "vla_annotation_fraction": 0.8,
            "guidance_annotation_fraction": 0.7,
        },
        semantic_world_model=_semantic_world_model(),
        selection_summary={
            "selection_policy": "heuristic_plus_learned_helper",
            "selected_ids": ["dp_meta_runtime"],
            "selection_meta_choice": {
                "selected_datapack_id": "dp_meta_runtime",
                "candidate_count": 2,
                "selected_execution_ready": True,
                "selected_non_heuristic_grounding": True,
                "selected_benchmark_eligible": True,
                "top_score": 2.1,
                "margin_to_runner_up": 0.3,
                "selected_quality_score": 0.82,
            },
        },
    )

    helper = output.metadata["learned_helper"]
    assert output.objective_preset in {"balanced", "safety", "energy_saver", "throughput"}
    assert helper["planning_application"]["planning_available"] is True
    assert helper["planning_application"]["objective_prior"] == "safety"
    assert helper["planning_application"]["energy_profile_source"] == "blended"
    assert output.metadata["planning_context"]["selection_summary_available"] is True

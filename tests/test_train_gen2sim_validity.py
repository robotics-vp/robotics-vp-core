import json
from argparse import Namespace

import numpy as np

from scripts.train_gen2sim_validity import _run_training
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality


def _write_branch_corpus(root) -> tuple[str, str, str, str]:
    corpus_path = root / "branches.npz"
    np.savez(
        corpus_path,
        n_branches=np.array(2),
        objective_dim=np.array(4),
        branch_0_z_sequence=np.array([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]], dtype=np.float32),
        branch_0_actions=np.array([[0.1], [0.2]], dtype=np.float32),
        branch_0_source_episode=np.array(0),
        branch_0_source_timestep=np.array(1),
        branch_0_trust_score=np.array(0.95),
        branch_0_std_ratio=np.array(1.0),
        branch_0_brick_id=np.array(0),
        branch_0_objective_vector=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        branch_0_branch_value=np.array(0.9),
        branch_1_z_sequence=np.array([[0.2, 0.2], [0.25, 0.3], [0.3, 0.4]], dtype=np.float32),
        branch_1_actions=np.array([[0.2], [0.15]], dtype=np.float32),
        branch_1_source_episode=np.array(1),
        branch_1_source_timestep=np.array(0),
        branch_1_trust_score=np.array(0.88),
        branch_1_std_ratio=np.array(0.94),
        branch_1_brick_id=np.array(1),
        branch_1_objective_vector=np.array([0.5, 0.5, 0.0, 1.0], dtype=np.float32),
        branch_1_branch_value=np.array(0.5),
    )
    metadata_path = root / "branches_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "synthetic_branch_corpus_metadata_v1",
                "scene_tracks_backend": "real",
                "vision_backbone_selected": "real",
                "semantic_grounding_mode": "non_heuristic",
                "semantic_memory_grounded": True,
            }
        ),
        encoding="utf-8",
    )
    gap_labels_path = root / "branches_gap_labels.json"
    gap_labels_path.write_text(
        json.dumps(
            [
                {
                    "branch_idx": 0,
                    "coverage_gap_contribution": 0.5,
                    "economic_priority": 0.4,
                },
                {
                    "branch_idx": 1,
                    "coverage_gap_contribution": 0.2,
                    "economic_priority": 0.1,
                },
            ]
        ),
        encoding="utf-8",
    )
    gen2sim_validity_path = root / "branches_gen2sim_validity.json"
    gen2sim_validity_path.write_text(
        json.dumps(
            [
                {
                    "branch_idx": 0,
                    "assessment_id": "gen2sim_branch_0",
                    "subject_id": "branch_0",
                    "subject_kind": "synthetic_branch",
                    "validity_score": 0.85,
                    "value_support_score": 0.75,
                    "admission_score": 0.7978125,
                    "promotion_stage": "shadow_candidate",
                    "benchmark_gate": {"ready": False},
                    "execution_preconditions": {"ready": True},
                    "component_scores": {"dynamics_score": 0.9},
                    "reason_codes": ["gen2sim_validity_ok"],
                    "metadata": {
                        "benchmark_signals": {"benchmark_eligible": True},
                    },
                },
                {
                    "branch_idx": 1,
                    "assessment_id": "gen2sim_branch_1",
                    "subject_id": "branch_1",
                    "subject_kind": "synthetic_branch",
                    "validity_score": 0.58,
                    "value_support_score": 0.35,
                    "admission_score": 0.48575,
                    "promotion_stage": "shadow_candidate",
                    "benchmark_gate": {"ready": False},
                    "execution_preconditions": {"ready": True},
                    "component_scores": {"dynamics_score": 0.6},
                    "reason_codes": ["benchmark_gate_not_ready"],
                    "metadata": {
                        "benchmark_signals": {"benchmark_eligible": False},
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    return (
        str(corpus_path),
        str(metadata_path),
        str(gap_labels_path),
        str(gen2sim_validity_path),
    )


def test_train_gen2sim_validity_emits_runtime_package(tmp_path) -> None:
    branch_corpus, metadata_path, gap_labels_path, gen2sim_validity_path = _write_branch_corpus(
        tmp_path
    )
    output_dir = tmp_path / "gen2sim_training"
    args = Namespace(
        branch_corpus=branch_corpus,
        metadata=metadata_path,
        gap_labels=gap_labels_path,
        gen2sim_validity=gen2sim_validity_path,
        epochs=2,
        lr=1e-3,
        hidden_dim=16,
        save_dir=str(output_dir),
        run_name="gen2sim_validity_test",
        seed=7,
        skip_regal_runner=False,
    )

    holder = {}

    def _wrapped(runner) -> None:
        holder["payload"] = _run_training(args, runner)

    run_training_with_regality(
        training_fn=_wrapped,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=7,
            num_episodes=2,
            training_steps=2,
            fail_on_verify_error=False,
        ),
        plan_sha="plan_sha",
        plan_id="gen2sim_validity_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    package = json.loads((output_dir / "gen2sim_validity_package.json").read_text(encoding="utf-8"))

    assert manifest["training_kind"] == "gen2sim_validity_helper"
    assert manifest["artifact_paths"]["gen2sim_validity_runtime_package"].endswith(
        "gen2sim_validity_package.json"
    )
    assert manifest["artifact_paths"]["gen2sim_validity_dataset_summary"].endswith(
        "gen2sim_validity_dataset_summary.json"
    )
    assert package["promotion_stage"] == "shadow_candidate"
    assert package["inference_contract"]["targets"] == [
        "validity_score",
        "value_support_score",
    ]
    assert holder["payload"]["training_summary"]["row_count"] == 2

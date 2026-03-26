import json
from argparse import Namespace

import numpy as np

from scripts.train_offline_with_local_synth import _run_training
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality


def _write_real_dataset(path) -> None:
    np.savez(
        path,
        n_episodes=np.array(2),
        ep_0_z_sequence=np.array([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]], dtype=np.float32),
        ep_0_actions=np.array([[0.1], [0.2]], dtype=np.float32),
        ep_0_rewards=np.array([0.4, 0.5], dtype=np.float32),
        ep_1_z_sequence=np.array([[0.3, 0.1], [0.4, 0.2], [0.5, 0.3]], dtype=np.float32),
        ep_1_actions=np.array([[0.3], [0.1]], dtype=np.float32),
        ep_1_rewards=np.array([0.3, 0.2], dtype=np.float32),
    )


def _write_synth_dataset(root) -> tuple[str, str, str]:
    corpus_path = root / "branches.npz"
    np.savez(
        corpus_path,
        n_branches=np.array(1),
        objective_dim=np.array(4),
        branch_0_z_sequence=np.array([[0.2, 0.2], [0.25, 0.3], [0.3, 0.4]], dtype=np.float32),
        branch_0_actions=np.array([[0.2], [0.15]], dtype=np.float32),
        branch_0_source_episode=np.array(0),
        branch_0_source_timestep=np.array(0),
        branch_0_trust_score=np.array(0.95),
        branch_0_std_ratio=np.array(1.0),
        branch_0_brick_id=np.array(0),
        branch_0_objective_vector=np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        branch_0_branch_value=np.array(0.9),
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
                "future_training_signals": {"scene_tracks_non_stub": True},
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
                    "skill_edge": "drawer -> place",
                    "branch_value": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )
    return str(corpus_path), str(metadata_path), str(gap_labels_path)


def test_local_synth_training_emits_runtime_artifacts(tmp_path) -> None:
    real_data = tmp_path / "real_rollouts.npz"
    _write_real_dataset(real_data)
    synth_data, synth_metadata, synth_gap_labels = _write_synth_dataset(tmp_path)
    output_dir = tmp_path / "training_run"

    args = Namespace(
        real_data=str(real_data),
        synth_data=synth_data,
        synth_metadata=synth_metadata,
        synth_gap_labels=synth_gap_labels,
        w_econ_lattice=str(tmp_path / "missing_lattice.pt"),
        output_dir=str(output_dir),
        seed=7,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        synth_weight=0.3,
        econ_weight_scale=1.0,
        use_lambda_controller=False,
        lambda_controller=str(tmp_path / "missing_lambda.pt"),
        eval_every=1,
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
            num_episodes=1,
            training_steps=1,
            fail_on_verify_error=False,
        ),
        plan_sha="plan_sha",
        plan_id="offline_local_synth_test",
    )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text(encoding="utf-8"))
    payload = holder["payload"]

    assert manifest["training_kind"] == "offline_local_synth"
    assert manifest["artifact_paths"]["synthetic_branch_summary"].endswith("synthetic_branch_summary.json")
    assert manifest["artifact_paths"]["offline_local_synth_eval"].endswith("offline_local_synth_eval.json")
    assert manifest["metadata"]["scene_tracks_backend"] == "real"
    assert payload["results"]["synthetic_branch_policy"]["benchmark_gate_ready"] is True
    assert (output_dir / "synthetic_branch_execution_preconditions.json").exists()
    assert (output_dir / "offline_baseline_actor.pt").exists()

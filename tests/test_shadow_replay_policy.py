import json
from pathlib import Path

import yaml

from src.learning.replay_policy_trainer import evaluate_replay_policy, train_replay_policy
from src.replay.dataset import ReplayDatasetBuilder
from src.shadow_runtime.control_plane import run_shadow_control_plane


def test_shadow_replay_policy_training_and_eval_smoke(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    policy_dir = tmp_path / "replay_policy"
    eval_dir = tmp_path / "replay_policy_eval"
    config_path = tmp_path / "cpu_smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "hidden_dim": 64,
                    "head_hidden_dim": 32,
                    "vision_dim": 16,
                    "use_condition_film": True,
                    "use_condition_vector_for_policy": True,
                    "condition_fusion_mode": "film",
                    "default_skill_mode": "efficiency_throughput",
                    "enable_value_head": True,
                },
                "training": {
                    "seed": 42,
                    "device": "cpu",
                    "batch_size": 4,
                    "epochs": 2,
                    "lr": 1e-3,
                    "val_fraction": 0.25,
                    "grad_clip": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=42,
        episodes=3,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(dataset_dir)

    train_result = train_replay_policy(
        dataset_dir=dataset_dir,
        config_path=config_path,
        output_dir=policy_dir,
    )
    eval_result = evaluate_replay_policy(
        dataset_dir=dataset_dir,
        checkpoint_path=train_result.best_checkpoint_path,
        output_dir=eval_dir,
    )

    assert Path(train_result.best_checkpoint_path).exists()
    assert Path(train_result.metrics_path).exists()
    assert eval_result["metrics"]["count"] > 0
    predictions = (eval_dir / "policy_predictions.jsonl").read_text().strip().splitlines()
    assert len(predictions) >= 1

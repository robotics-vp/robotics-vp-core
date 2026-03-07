import json
from pathlib import Path

import yaml

from src.learning.offline_rl import train_offline_rl
from src.replay.dataset import ReplayDatasetBuilder
from src.shadow_runtime.control_plane import run_shadow_control_plane


def test_shadow_offline_rl_training_smoke(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    output_dir = tmp_path / "offline_rl"
    config_path = tmp_path / "offline_rl.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "algorithm": "td3_bc_shadow",
                "device": "cpu",
                "training": {
                    "seed": 42,
                    "epochs": 2,
                    "batch_size": 4,
                    "gamma": 0.98,
                    "tau": 0.02,
                    "policy_delay": 2,
                    "actor_lr": 1e-3,
                    "critic_lr": 1e-3,
                    "bc_weight": 2.0,
                    "val_fraction": 0.25,
                },
                "model": {
                    "hidden_dim": 64,
                    "head_hidden_dim": 32,
                    "vision_dim": 16,
                    "use_condition_film": True,
                    "use_condition_vector_for_policy": True,
                    "condition_fusion_mode": "film",
                    "default_skill_mode": "efficiency_throughput",
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

    result = train_offline_rl(dataset_dir=dataset_dir, config_path=config_path, output_dir=output_dir)

    assert Path(result.actor_checkpoint_path).exists()
    assert Path(result.critic_checkpoint_path).exists()
    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["legacy_default_untouched"] is True

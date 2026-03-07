import json
from pathlib import Path

import pytest

from src.training.checkpoint_registry import build_checkpoint_record
from src.training.regal_training_runner import TrainingRunConfig, run_training_with_regality
from src.valuation.trajectory_audit import create_trajectory_audit


def test_regal_training_runner_emits_training_runtime_manifest(tmp_path):
    output_dir = tmp_path / "training_run"
    artifact_path = output_dir / "artifact.json"
    checkpoint_path = output_dir / "checkpoint.pt"

    def _train(runner):
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps({"artifact": True}), encoding="utf-8")
        checkpoint_path.write_bytes(b"checkpoint")
        runner.set_eligible_datapacks(["ep_001"])
        runner.set_sampler_config(seed=42, config_sha="cfg_123")
        runner.record_sample("shadow_task", datapack_id="ep_001", slice_id="ep_001")
        runner.add_trajectory_audit(
            create_trajectory_audit(
                episode_id="ep_001",
                num_steps=2,
                actions=[[0.1, 0.2], [0.2, 0.3]],
                rewards=[0.5, 0.6],
                reward_components={"throughput": [0.5, 0.6]},
            )
        )
        runner.update_step(2)
        runner.configure_training_runtime(
            training_kind="unit_test_training",
            config_digest="cfg_123",
            replay_dataset_summary={"num_episodes": 1},
            source_domain_coverage={"source_domain_counts": {"synthetic": 1}},
            receipt_label_coverage={"total_labels": 1},
            objective_profile_snapshot={"profile_id": "balanced_contract"},
            promotion_policy_snapshot={"policy_name": "promotion_default"},
        )
        runner.set_regal_result({"overall_status": "pass"}, context_sha="ctx_123")
        runner.register_artifact("unit_artifact", artifact_path)
        runner.register_checkpoint(
            build_checkpoint_record(
                checkpoint_id="unit_checkpoint",
                model_family="unit_test",
                model_version="unit_v1",
                path=checkpoint_path,
                step=2,
                epoch=1,
            )
        )

    result = run_training_with_regality(
        training_fn=_train,
        config=TrainingRunConfig(
            output_dir=str(output_dir),
            seed=42,
            num_episodes=1,
            training_steps=2,
            fail_on_verify_error=False,
        ),
        plan_sha="plan_sha",
        plan_id="unit_test_training",
    )

    assert result.training_runtime_manifest_sha is not None
    assert result.checkpoint_registry_sha is not None
    assert (output_dir / "training_runtime_manifest.json").exists()
    assert (output_dir / "checkpoint_registry.json").exists()
    assert (output_dir / "training_runtime_summary.md").exists()


def test_regal_training_runner_writes_failed_runtime_manifest_on_exception(tmp_path):
    output_dir = tmp_path / "failed_training_run"

    def _train(runner):
        runner.configure_training_runtime(
            training_kind="failing_training",
            config_digest="cfg_123",
            source_domain_coverage={"source_domain_counts": {"synthetic": 1}},
            receipt_label_coverage={"total_labels": 0},
        )
        raise RuntimeError("intentional failure")

    with pytest.raises(RuntimeError):
        run_training_with_regality(
            training_fn=_train,
            config=TrainingRunConfig(output_dir=str(output_dir), seed=0),
            plan_sha="plan_sha",
            plan_id="failing_training",
        )

    manifest = json.loads((output_dir / "training_runtime_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "intentional failure" in manifest["failure_reason"]


def test_shadow_training_entrypoints_use_canonical_runner():
    for path in (
        "scripts/train_shadow_replay_policy.py",
        "scripts/train_shadow_pricing_models.py",
        "scripts/train_shadow_offline_rl.py",
    ):
        content = Path(path).read_text(encoding="utf-8")
        assert "run_training_with_regality" in content

from src.replay.compatibility import check_artifact_schema_versions
from src.training.checkpoint_registry import (
    CheckpointRegistry,
    CheckpointRecord,
    check_checkpoint_registry_compatibility,
)
from src.training.training_manifest import (
    TrainingRuntimeManifest,
    check_training_runtime_manifest_compatibility,
)


def test_artifact_schema_compatibility_helpers_detect_mismatch(tmp_path):
    results = check_artifact_schema_versions(
        {"artifact_a": {"schema_version": "v1", "config_digest": "cfg", "dataset_digest": "ds"}},
        required_versions={"artifact_a": "v2"},
    )
    assert results[0].compatible is False
    assert "schema_version_mismatch" in results[0].reasons


def test_training_runtime_and_checkpoint_registry_compatibility_roundtrip(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"ok")
    registry = CheckpointRegistry(
        schema_version="training_checkpoint_registry_v1",
        run_id="run_compat",
        training_kind="shadow_offline_rl",
        created_at="2026-01-01T00:00:00+00:00",
        checkpoints=[
            CheckpointRecord(
                checkpoint_id="ckpt",
                model_family="shadow_offline_rl_actor",
                model_version="offline_td3_bc_shadow_actor_v1",
                path=str(checkpoint_path),
                file_name=checkpoint_path.name,
                artifact_digest="digest",
            )
        ],
    )
    assert check_checkpoint_registry_compatibility(registry).compatible is True

    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{}", encoding="utf-8")
    manifest = TrainingRuntimeManifest(
        schema_version="training_runtime_manifest_v1",
        run_id="run_compat",
        training_kind="shadow_offline_rl",
        status="completed",
        seed=0,
        plan_id="shadow_offline_rl",
        plan_sha="sha",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:01:00+00:00",
        config_path=None,
        config_digest="cfg",
        replay_dataset_dir=None,
        replay_manifest_digest=None,
        replay_dataset_summary={},
        objective_profile_snapshot={},
        promotion_policy_snapshot={},
        source_domain_coverage={},
        receipt_label_coverage={},
        artifact_paths={"artifact": str(artifact_path)},
        checkpoint_registry_path=str(checkpoint_path),
    )
    assert check_training_runtime_manifest_compatibility(manifest).compatible is True

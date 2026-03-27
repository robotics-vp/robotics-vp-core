import json

from src.training.checkpoint_registry import (
    build_checkpoint_record,
    check_checkpoint_registry_compatibility,
    create_checkpoint_registry,
    write_checkpoint_registry,
)
from src.training.training_manifest import (
    TrainingRuntimeManifest,
    check_training_runtime_manifest_compatibility,
    load_training_runtime_manifest,
    write_training_runtime_manifest,
)


def test_training_manifest_and_checkpoint_registry_roundtrip(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"shadow-checkpoint")
    checkpoint = build_checkpoint_record(
        checkpoint_id="shadow_policy_best",
        model_family="shadow_replay_policy",
        model_version="replay_policy_bc_v1",
        path=checkpoint_path,
        step=12,
        epoch=3,
        is_best=True,
        metadata={"dataset_digest": "ds_123"},
    )
    registry = create_checkpoint_registry(
        run_id="run_123",
        training_kind="shadow_replay_policy",
        checkpoints=[checkpoint],
    )
    registry_path = tmp_path / "checkpoint_registry.json"
    registry_sha = write_checkpoint_registry(registry_path, registry)
    assert registry_sha
    assert check_checkpoint_registry_compatibility(registry).compatible is True

    artifact_path = tmp_path / "summary.json"
    artifact_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    manifest = TrainingRuntimeManifest(
        schema_version="training_runtime_manifest_v1",
        run_id="run_123",
        training_kind="shadow_replay_policy",
        status="completed",
        seed=42,
        plan_id="shadow_replay_policy",
        plan_sha="plan_sha",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:05:00+00:00",
        config_path="configs/replay_policy/cpu_smoke.yaml",
        config_digest="cfg_123",
        replay_dataset_dir="artifacts/replay_dataset",
        replay_manifest_digest="manifest_123",
        replay_dataset_summary={"num_episodes": 3},
        objective_profile_snapshot={"profile_id": "balanced_contract"},
        promotion_policy_snapshot={"policy_name": "promotion_default"},
        source_domain_coverage={"source_domain_counts": {"synthetic": 3}},
        receipt_label_coverage={"total_labels": 3},
        artifact_paths={"training_summary": str(artifact_path)},
        inferential_learnability_summary={"contract_count": 3, "benchmark_receipt_backed_count": 1},
        inferential_admission_summary={"decision_count": 3, "decision_counts": {"adapt_now": 2}},
        inferential_work_order_summary={"work_orders": 2},
        checkpoint_registry_path=str(registry_path),
        checkpoint_registry_digest=registry_sha,
        promotion_evidence_path=str(artifact_path),
        promotion_evidence_digest="evidence_123",
        metadata={"unit_test": True},
    )
    manifest_path = tmp_path / "training_runtime_manifest.json"
    manifest_sha = write_training_runtime_manifest(manifest_path, manifest)
    assert manifest_sha
    loaded = load_training_runtime_manifest(manifest_path)
    assert loaded.inferential_admission_summary["decision_count"] == 3
    assert check_training_runtime_manifest_compatibility(manifest).compatible is True


def test_training_manifest_compatibility_detects_missing_artifact(tmp_path):
    manifest = TrainingRuntimeManifest(
        schema_version="training_runtime_manifest_v1",
        run_id="run_456",
        training_kind="shadow_pricing_models",
        status="completed",
        seed=1,
        plan_id="shadow_pricing_models",
        plan_sha="plan_sha",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:05:00+00:00",
        config_path=None,
        config_digest="cfg",
        replay_dataset_dir=None,
        replay_manifest_digest=None,
        replay_dataset_summary={},
        objective_profile_snapshot={},
        promotion_policy_snapshot={},
        source_domain_coverage={},
        receipt_label_coverage={},
        artifact_paths={"missing": str(tmp_path / "missing.json")},
    )
    compatibility = check_training_runtime_manifest_compatibility(manifest)
    assert compatibility.compatible is False
    assert "artifact_path_missing" in compatibility.reasons

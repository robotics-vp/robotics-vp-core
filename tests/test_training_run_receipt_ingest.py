import json

from src.replay.dataset import ReplayDatasetBuilder
from src.replay.receipt_ingest import (
    build_training_run_receipt_label_bundle,
    resolve_receipt_label_bundle,
)
from src.shadow_runtime.control_plane import run_shadow_control_plane
from src.training.training_manifest import TrainingRuntimeManifest, write_training_runtime_manifest


def test_training_run_receipt_ingest_prefers_online_training_artifacts(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    training_run_dir = tmp_path / "training_run"
    episode_logs_dir = training_run_dir / "online_episode_logs"
    dataset_dir = training_run_dir / "online_replay_dataset"
    receipts_path = training_run_dir / "online_episode_receipts.jsonl"
    promotion_ledger_path = training_run_dir / "promotion_ledger_v1.json"
    budget_settlement_path = training_run_dir / "budget_settlement_v1.json"

    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=11,
        episodes=1,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    trace_payload = json.loads((shadow_dir / "shadow_episode_traces.json").read_text(encoding="utf-8"))
    episode_log = trace_payload["episodes"][0]["episode_log"]
    episode_id = episode_log["metadata"]["episode_id"]

    episode_logs_dir.mkdir(parents=True, exist_ok=True)
    (episode_logs_dir / f"{episode_id}.json").write_text(json.dumps(episode_log), encoding="utf-8")
    dataset = ReplayDatasetBuilder().add_workcell_episode_log(
        episode_logs_dir / f"{episode_id}.json",
        run_id="training_run_001",
        source_domain="training_run",
    ).write(dataset_dir)

    receipts_path.write_text(
        json.dumps(
            {
                "run_id": "training_run_001",
                "episode_id": episode_id,
                "source_domain": "training_run",
                "predicted_value": 4.0,
                "realized_value": 3.25,
                "quoted_rate": 4.0,
                "accepted_rate": 3.25,
                "pricing_accepted": True,
                "task_success": True,
                "objective_satisfied": True,
                "realized_reward": 2.75,
                "failure_events": [],
                "risk_events": [],
                "incident_events": [],
                "expected_adaptation_benefit": 0.5,
                "realized_adaptation_benefit": 0.35,
                "adaptation_compute_cost": 0.1,
                "adaptation_risk_cost": 0.0,
                "adaptation_review_required": False,
                "marginal_frontier_gain_predicted": 0.3,
                "marginal_frontier_gain_realized": 0.22,
                "data_share_credit_predicted": 0.5,
                "data_share_credit_realized": 0.4,
                "downweight_recommended": False,
                "human_review_label": "pass",
                "override_label": None,
                "scene_tracks_non_stub": True,
                "scene_tracks_backend": "real",
                "teacher_runtime_backend_selected": "real",
                "semantic_memory_grounded": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    promotion_ledger_path.write_text(
        json.dumps(
            {
                "schema_version": "promotion_ledger_v1",
                "run_id": "training_run_001",
                "summary": {"eligible_nodes": 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    budget_settlement_path.write_text(
        json.dumps(
            {
                "schema_version": "budget_settlement_v1",
                "run_id": "training_run_001",
                "budget_settlement_live": True,
                "observed_receipts_ref": str(receipts_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = TrainingRuntimeManifest(
        schema_version="training_runtime_manifest_v1",
        run_id="training_run_001",
        training_kind="online_sac",
        status="completed",
        seed=11,
        plan_id="online_sac",
        plan_sha="plan_sha",
        started_at="2026-01-01T00:00:00+00:00",
        ended_at="2026-01-01T00:05:00+00:00",
        config_path=None,
        config_digest="cfg_123",
        replay_dataset_dir=str(dataset_dir),
        replay_manifest_digest=dataset.manifest.manifest_hash,
        replay_dataset_summary=dataset.to_summary(),
        objective_profile_snapshot={"profile_id": "balanced_contract"},
        promotion_policy_snapshot={"policy_name": "promotion_default"},
        source_domain_coverage={"source_domain_counts": {"training_run": 1}},
        receipt_label_coverage={},
        artifact_paths={
            "online_episode_logs": str(episode_logs_dir),
            "online_episode_receipts": str(receipts_path),
            "online_replay_dataset_manifest": str(dataset_dir / "manifest.json"),
            "promotion_ledger_ref": str(promotion_ledger_path),
            "budget_settlement_report": str(budget_settlement_path),
        },
        promotion_ledger_path=str(promotion_ledger_path),
        budget_settlement_path=str(budget_settlement_path),
        budget_settlement_live=True,
    )
    write_training_runtime_manifest(training_run_dir / "training_runtime_manifest.json", manifest)

    bundle = build_training_run_receipt_label_bundle(training_run_dir)
    assert bundle.label_mode == "training_run"
    assert bundle.coverage_summary()["source_domain_counts"]["training_run"] >= 4
    assert bundle.deployment_receipts[0].task_success is True
    assert bundle.deployment_receipts[0].realized_reward == 2.75
    assert bundle.metadata["promotion_ledger_ref"].endswith("promotion_ledger_v1.json")
    assert bundle.metadata["budget_settlement_live"] is True
    execution_summary = bundle.metadata["execution_precondition_summary"]
    assert execution_summary["satisfied_preconditions"]["artifact::training_runtime_manifest"] == 1
    assert execution_summary["satisfied_preconditions"]["artifact::promotion_ledger_ref"] == 1
    assert execution_summary["satisfied_preconditions"]["signal_bool::budget_settlement_live"] == 1
    assert execution_summary["satisfied_preconditions"]["signal_bool::scene_tracks_non_stub"] == 1
    assert execution_summary["satisfied_preconditions"]["signal_bool::teacher_runtime_real"] == 1

    resolved = resolve_receipt_label_bundle(
        dataset=dataset,
        receipt_label_dir=training_run_dir,
        allow_synthetic=False,
        label_mode="training_run",
    )
    assert resolved.coverage_summary()["deployment_receipts"] == 1

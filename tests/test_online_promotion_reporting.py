import json

from src.regality.promotion_policy import load_regal_promotion_policy
from src.regality.promotion_reporting import (
    build_promotion_evidence_report,
    write_promotion_evidence_report,
)
from src.orchestrator.shadow_advisory import build_shadow_advisory_output
from src.replay.dataset import ReplayDatasetBuilder
from src.replay.receipt_ingest import build_training_run_receipt_label_bundle
from src.shadow_runtime.control_plane import run_shadow_control_plane


def test_online_promotion_reporting_includes_coverage_and_error_summaries(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    training_run_dir = tmp_path / "training_run"
    episode_logs_dir = training_run_dir / "online_episode_logs"
    dataset_dir = training_run_dir / "online_replay_dataset"
    receipts_path = training_run_dir / "online_episode_receipts.jsonl"
    output_dir = tmp_path / "promotion_eval"

    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=7,
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
        run_id="online_run_001",
        source_domain="training_run",
    ).write(dataset_dir)
    receipts_path.write_text(
        json.dumps(
            {
                "run_id": "online_run_001",
                "episode_id": episode_id,
                "source_domain": "training_run",
                "predicted_value": 2.0,
                "realized_value": 1.0,
                "quoted_rate": 2.0,
                "accepted_rate": 1.0,
                "pricing_accepted": False,
                "task_success": False,
                "objective_satisfied": False,
                "realized_reward": -0.25,
                "failure_events": ["sla_violation"],
                "risk_events": ["high_error_rate"],
                "incident_events": ["sla_violation", "high_error_rate"],
                "expected_adaptation_benefit": 0.2,
                "realized_adaptation_benefit": 0.05,
                "adaptation_compute_cost": 0.1,
                "adaptation_risk_cost": 0.15,
                "adaptation_review_required": True,
                "marginal_frontier_gain_predicted": 0.15,
                "marginal_frontier_gain_realized": 0.01,
                "data_share_credit_predicted": 0.1,
                "data_share_credit_realized": 0.02,
                "downweight_recommended": True,
                "human_review_label": "needs_review",
                "override_label": "operator_hold",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    receipt_bundle = build_training_run_receipt_label_bundle(
        training_run_dir,
        replay_dataset_dir=dataset.root_dir,
        label_mode="training_run",
    )
    report = build_promotion_evidence_report(
        dataset=dataset,
        promotion_policy=load_regal_promotion_policy("configs/regality/promotion_default.yaml"),
        receipt_bundle=receipt_bundle,
        work_orders=build_shadow_advisory_output(replay_dataset_dir=str(dataset.root_dir))["collection_work_orders"],
    )
    paths = write_promotion_evidence_report(output_dir, report)

    assert report.receipt_label_coverage["source_domain_counts"]["training_run"] >= 4
    assert report.node_reports[0].coverage["episode_count"] == 1
    assert "trace_ready_episode_count" in report.node_reports[0].coverage
    assert "by_source_domain" in report.node_reports[0].disagreement_slices
    assert "count" in report.node_reports[0].false_positive_summary
    assert "count" in report.node_reports[0].false_negative_summary
    assert paths["json"]
    markdown = (output_dir / "regal_promotion_eval.md").read_text(encoding="utf-8")
    assert "Coverage:" in markdown

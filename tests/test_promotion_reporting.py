from src.regality.promotion_policy import load_regal_promotion_policy
from src.regality.promotion_reporting import (
    build_promotion_evidence_report,
    write_promotion_evidence_report,
)
from src.orchestrator.shadow_advisory import build_shadow_advisory_output
from src.replay.dataset import ReplayDatasetBuilder, load_replay_dataset
from src.replay.receipt_ingest import build_synthetic_receipt_label_bundle
from src.shadow_runtime.control_plane import run_shadow_control_plane


def test_promotion_reporting_emits_sidecars(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    output_dir = tmp_path / "promotion_eval"
    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=42,
        episodes=2,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(dataset_dir)
    dataset = load_replay_dataset(dataset_dir)
    bundle = build_synthetic_receipt_label_bundle(dataset)
    policy = load_regal_promotion_policy("configs/regality/promotion_default.yaml")

    report = build_promotion_evidence_report(
        dataset=dataset,
        promotion_policy=policy,
        receipt_bundle=bundle,
    )
    paths = write_promotion_evidence_report(output_dir, report)

    assert report.summary["node_count"] >= 5
    assert report.receipt_label_coverage["total_labels"] > 0
    assert report.summary["trace_ready_episode_count"] == dataset.manifest.num_episodes
    assert report.summary["inferential_learnability_summary"]["contract_count"] == dataset.manifest.num_episodes
    assert report.node_reports[0].coverage["trace_ready_episode_count"] == dataset.manifest.num_episodes
    assert paths["json"]
    assert paths["markdown"]
    assert any(key.startswith("sidecar::") for key in paths)


def test_promotion_reporting_accepts_work_order_evidence(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    run_shadow_control_plane(
        output_dir=shadow_dir,
        seed=42,
        episodes=2,
        objective_profile_id="balanced_contract",
        include_regal=True,
        timestamp_base="2026-01-01T00:00:00+00:00",
    )
    ReplayDatasetBuilder().add_shadow_run(shadow_dir).write(dataset_dir)
    dataset = load_replay_dataset(dataset_dir)
    advisory = build_shadow_advisory_output(replay_dataset_dir=str(dataset_dir))
    work_orders = (
        advisory["adaptation_work_orders"]
        + advisory["collection_work_orders"]
        + advisory["review_work_orders"]
    )
    report = build_promotion_evidence_report(
        dataset=dataset,
        promotion_policy=load_regal_promotion_policy("configs/regality/promotion_default.yaml"),
        receipt_bundle=build_synthetic_receipt_label_bundle(dataset),
        work_orders=work_orders,
    )

    assert report.summary["work_order_count"] == len(work_orders)
    assert report.node_reports[0].downstream_usefulness["executable_work_orders"] >= 0

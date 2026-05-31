from pathlib import Path

from src.world_model.economic_world_model import (
    build_post_gap_readiness_bundle,
    load_benchmark_gate_specs,
    load_corpus_prep_artifact_plans,
    load_external_dataset_corpus_plans,
    load_gpu_day_one_runbooks,
    load_post_gap_readiness_report,
    load_readiness_specs,
    save_post_gap_readiness_bundle,
)


def test_post_gap_readiness_manifest_covers_all_lanes() -> None:
    bundle = build_post_gap_readiness_bundle()
    report = bundle["report"]
    dataset_ids = {dataset.dataset_id for dataset in bundle["datasets"]}
    gate_keys = {gate.gate_key for gate in bundle["benchmark_gates"]}
    purchase_keys = {spec.key for spec in bundle["purchase_readiness"]}

    assert report.all_post_gap_items_manifested is True
    assert report.ready_for_august_gpu_window is True
    assert report.launch_authority_granted is False
    assert report.provider_executed is False
    assert report.gpu_training_executed is False
    assert report.external_download_executed is False
    assert report.promotion_eligible is False
    assert report.phase7_constraint_honored is True
    assert report.gpu_day_one_runbook_count >= 5
    assert report.external_dataset_count >= 8
    assert report.corpus_prep_artifact_count >= report.external_dataset_count * 6
    assert report.benchmark_gate_count >= 6
    assert report.provider_runtime_packaging_count >= 6
    assert report.replay_loop_count >= 5
    assert report.g1_r1_purchase_readiness_count >= 10
    assert report.evidence_hygiene_count >= 7
    assert {
        "open_x_embodiment_oxe",
        "droid",
        "bridgedata_v2",
        "lerobot_hub_curated",
        "robomind_v2",
        "rh20t",
        "ego4d_ego_exo4d",
        "agibot_world_watchlist",
        "local_robotics_vp_artifacts",
    }.issubset(dataset_ids)
    assert {
        "transport_eval_acceptance",
        "perception_replay_consistency",
        "command_timing_safety_benchmark",
        "economic_allocation_shadow_benchmark",
        "phase7_governance_outcome_scoring",
        "promotion_gate_fail_closed",
    }.issubset(gate_keys)
    assert {
        "variant_decision_criteria",
        "workspace_safety_plan",
        "estop_and_recovery_plan",
        "network_dds_plan",
        "companion_compute_assumptions",
        "camera_sensor_mounting_plan",
        "storage_logging_plan",
        "calibration_checklist",
        "first_week_bringup_runbook",
        "do_not_run_until_safety_gates",
    }.issubset(purchase_keys)


def test_post_gap_readiness_dataset_corpus_plans_are_fail_closed() -> None:
    bundle = build_post_gap_readiness_bundle()
    for dataset in bundle["datasets"]:
        assert dataset.repo_schema_targets
        assert dataset.normalization_steps
        assert dataset.split_manifest_plan
        assert dataset.replay_indexer_plan
        assert dataset.data_quality_receipt_plan
        assert dataset.label_gap_ledger_plan
        assert dataset.governance_label_spec
        assert dataset.transport_meta_node_plan
        assert dataset.download_executed is False
        assert dataset.ready_for_training is False

    for row in bundle["corpus_prep"]:
        assert row.launch_allowed is False
        assert row.required_fields
        assert row.acceptance_checks
        assert row.output_template.startswith(
            "artifacts/economic_world_model/post_gap_readiness/"
        )

    for runbook in bundle["runbooks"]:
        assert runbook.launch_allowed is False
        assert runbook.provider_bringup_ready is False
        assert runbook.gpu_training_ready is False
        assert runbook.expected_artifacts
        assert runbook.failure_receipts
        assert runbook.stop_conditions

    for gate in bundle["benchmark_gates"]:
        assert gate.status == "fail_closed_missing_evidence"
        assert gate.promotion_gate is True
        assert gate.promotion_eligible is False
        assert gate.fail_closed_reasons


def test_post_gap_readiness_save_and_load_round_trip(tmp_path: Path) -> None:
    report_payload = save_post_gap_readiness_bundle(output_dir=tmp_path)
    report = load_post_gap_readiness_report(report_payload["artifact_refs"]["report_path"])
    runbooks = load_gpu_day_one_runbooks(
        report_payload["artifact_refs"]["gpu_day_one_runbooks_path"]
    )
    datasets = load_external_dataset_corpus_plans(
        report_payload["artifact_refs"]["external_dataset_corpus_plan_path"]
    )
    corpus_prep = load_corpus_prep_artifact_plans(
        report_payload["artifact_refs"]["corpus_prep_artifact_plans_path"]
    )
    gates = load_benchmark_gate_specs(
        report_payload["artifact_refs"]["benchmark_gate_specs_path"]
    )
    provider_specs = load_readiness_specs(
        report_payload["artifact_refs"]["provider_runtime_packaging_specs_path"]
    )
    replay_specs = load_readiness_specs(
        report_payload["artifact_refs"]["perception_embodiment_replay_loop_specs_path"]
    )
    purchase_specs = load_readiness_specs(
        report_payload["artifact_refs"]["g1_r1_purchase_readiness_path"]
    )
    hygiene_specs = load_readiness_specs(
        report_payload["artifact_refs"]["evidence_hygiene_specs_path"]
    )

    assert report.report_id == report_payload["report_id"]
    assert len(runbooks) == report.gpu_day_one_runbook_count
    assert len(datasets) == report.external_dataset_count
    assert len(corpus_prep) == report.corpus_prep_artifact_count
    assert len(gates) == report.benchmark_gate_count
    assert len(provider_specs) == report.provider_runtime_packaging_count
    assert len(replay_specs) == report.replay_loop_count
    assert len(purchase_specs) == report.g1_r1_purchase_readiness_count
    assert len(hygiene_specs) == report.evidence_hygiene_count
    assert (tmp_path / "post_gap_readiness_v1.md").exists()

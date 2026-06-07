from __future__ import annotations

import json
from pathlib import Path

from src.dataset_bridges.lerobot_video_receipt_adapter import (
    adapt_lerobot_video_receipts_for_perception,
    build_fixture_lerobot_video_receipts,
    replay_episodes_from_lerobot_video_receipts,
    write_lerobot_video_receipt_bridge_artifacts,
)


def test_lerobot_video_receipts_preserve_replay_refs_and_posture() -> None:
    receipts = build_fixture_lerobot_video_receipts()

    episodes, steps, rows = replay_episodes_from_lerobot_video_receipts(
        receipts,
        dataset_id="fixture/lerobot_video",
    )

    assert len(rows) == 3
    assert len(episodes) == 1
    assert len(steps) == 3
    assert steps[0].step_idx == 0
    assert steps[0].timestamp == "0.000000"
    assert steps[0].task_id == "g1_shadow_pick_place"
    assert steps[0].env_id == "bipedal_whole_body_unitree_g1_shadow_replay"
    assert steps[0].metadata["camera_keys"] == ["front", "wrist"]
    assert steps[0].metadata["source_receipt_id"] == "fixture_video_receipt_001"
    assert steps[0].provenance["runtime_packet_ref"] == "runtime_packet_fixture.json"
    assert steps[0].provenance["event_spine_ref"] == "event_spine_fixture.json"
    assert steps[0].provenance["decision_ledger_ref"] == "decision_ledger_fixture.json"
    assert steps[0].objective_tensor_ref == "objective_tensor_fixture.json"
    assert steps[0].econ_tensor_ref == "econ_tensor_fixture.json"
    assert steps[0].metadata["video_receipt_bridge"]["provider_executed"] is False
    assert steps[0].metadata["future_training_signals"]["promotion_eligible"] is False
    assert episodes[0].objective_tensor_ref == "objective_tensor_fixture.json"
    assert episodes[0].econ_tensor_ref == "econ_tensor_fixture.json"


def test_lerobot_video_receipts_build_perception_samples_without_provider_claims() -> None:
    bundle = adapt_lerobot_video_receipts_for_perception(
        build_fixture_lerobot_video_receipts(),
        dataset_id="fixture/lerobot_video",
    )

    assert bundle.report.status == "ok_local_video_receipts_replay_perception_schema_only"
    assert bundle.report.provider_executed is False
    assert bundle.report.gpu_training_executed is False
    assert bundle.report.video_decoding_executed is False
    assert bundle.report.weights_downloaded is False
    assert bundle.report.unitree_hardware_truth is False
    assert bundle.report.promotion_eligible is False
    assert bundle.report.phase7_authority_granted is False
    assert bundle.report.evidence_fusion_sample_count == 3
    assert bundle.report.vjepa_temporal_sample_count == 2
    assert bundle.report.vision_backbone_projection_sample_count == 3
    assert bundle.report.unavailable_posture_count == 1

    first_sample = bundle.evidence_fusion_samples[0]
    assert first_sample.metadata["camera_keys"] == ["front", "wrist"]
    assert first_sample.metadata["video_receipt_bridge"]["promotion_eligible"] is False
    assert first_sample.metadata["provenance"]["event_spine_ref"] == "event_spine_fixture.json"
    assert {provider.truth_class for provider in first_sample.providers} == {
        "advisory_evidence"
    }
    assert all(
        provider.metadata["feature_posture"] == "cpu_placeholder_schema_verification"
        for provider in first_sample.providers
    )

    final_sample = bundle.evidence_fusion_samples[-1]
    providers_by_id = {provider.provider_id: provider for provider in final_sample.providers}
    assert providers_by_id["front"].truth_class == "advisory_evidence"
    assert providers_by_id["wrist"].availability_status == "unavailable"
    assert providers_by_id["wrist"].truth_class == "unavailable"


def test_lerobot_video_receipt_bridge_writes_receipts(tmp_path: Path) -> None:
    payload = write_lerobot_video_receipt_bridge_artifacts(
        build_fixture_lerobot_video_receipts(),
        tmp_path,
        dataset_id="fixture/lerobot_video",
    )

    assert payload["promotion_eligible"] is False
    assert payload["artifact_refs"]["report_path"].endswith(
        "lerobot_video_receipt_bridge_report_v1.json"
    )
    report = json.loads(Path(payload["artifact_refs"]["report_path"]).read_text())
    assert report["replay_step_count"] == 3
    assert Path(payload["artifact_refs"]["lerobot_rows_path"]).exists()
    assert Path(payload["artifact_refs"]["replay_steps_path"]).exists()

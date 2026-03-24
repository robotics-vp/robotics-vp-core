import json

from src.orchestrator.shadow_advisory import build_shadow_advisory_output
from src.replay.dataset import ReplayDatasetBuilder, load_replay_dataset
from src.replay.receipt_ingest import (
    build_synthetic_receipt_label_bundle,
    load_receipt_label_bundle,
    write_receipt_label_bundle,
)
from src.shadow_runtime.control_plane import run_shadow_control_plane


def test_receipt_ingest_roundtrip_and_shadow_advisory_consumption(tmp_path):
    shadow_dir = tmp_path / "shadow_run"
    dataset_dir = tmp_path / "replay_dataset"
    receipt_dir = tmp_path / "receipt_labels"
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
    paths = write_receipt_label_bundle(bundle, receipt_dir)
    restored = load_receipt_label_bundle(receipt_dir)

    assert restored.coverage_summary()["deployment_receipts"] == len(bundle.deployment_receipts)
    assert restored.coverage_summary()["covered_episode_count"] == dataset.manifest.num_episodes
    assert paths["bundle"]

    overlay_path = tmp_path / "epiplexity_overlays.jsonl"
    overlay_path.write_text(
        json.dumps(
            {
                "pack_id": dataset.episodes[0].datapack_summary["datapack_id"],
                "epiplexity_summary": {
                    "canonical_tokens": {
                        "steps_5_bs_4": {
                            "mean": {"delta_epi_vs_baseline": 0.25, "epi_per_flop": 0.4},
                            "confidence": 0.8,
                        }
                    },
                    "_default": {"repr_id": "canonical_tokens", "budget_id": "steps_5_bs_4"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    advisory = build_shadow_advisory_output(
        replay_dataset_dir=str(dataset_dir),
        receipt_label_dir=str(receipt_dir),
        epiplexity_overlay_path=str(overlay_path),
    )
    assert advisory["summary"]["receipt_label_coverage"]["total_labels"] > 0
    assert advisory["episodes"][0]["receipt_feedback"]["deployment_outcome"] is not None
    assert advisory["summary"]["epiplexity_overlay_joins"] >= 1
    assert advisory["episodes"][0]["epiplexity_evidence"]["overlay_joined"] is True
    assert "execution_preconditions" in advisory["episodes"][0]
    assert advisory["adaptation_budget"]["summary"]["work_orders"] >= 1

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.world_model.economic_world_model import (
    import_lerobot_corpus_slice,
    load_external_corpus_quality_receipts,
    load_external_lerobot_corpus_import_report,
)


def _write_fixture_source(root: Path) -> None:
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "README.md").write_text("# fixture\n", encoding="utf-8")
    (root / ".gitattributes").write_text("*.parquet filter=lfs\n", encoding="utf-8")
    (root / "meta/info.json").write_text(
        '{"codebase_version": "v3.0", "fps": 10}', encoding="utf-8"
    )
    (root / "meta/stats.json").write_text("{}", encoding="utf-8")
    pq.write_table(
        pa.table({"task_index": [0], "task": ["Push a block."]}),
        root / "meta/tasks.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 0],
                "dataset_from_index": [0, 3],
                "dataset_to_index": [3, 6],
                "tasks": [["Push a block."], ["Push a block."]],
                "length": [3, 3],
                "meta/episodes/chunk_index": [0, 0],
                "meta/episodes/file_index": [0, 0],
            }
        ),
        root / "meta/episodes/chunk-000/file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "observation.state": [
                    [0.0, 0.1],
                    [0.2, 0.3],
                    [0.4, 0.5],
                    [1.0, 1.1],
                    [1.2, 1.3],
                    [1.4, 1.5],
                ],
                "observation.environment_state": [
                    [0.0, 0.0],
                    [0.1, 0.1],
                    [0.2, 0.2],
                    [1.0, 1.0],
                    [1.1, 1.1],
                    [1.2, 1.2],
                ],
                "action": [
                    [0.0, 0.0],
                    [0.1, 0.1],
                    [0.2, 0.2],
                    [1.0, 1.0],
                    [1.1, 1.1],
                    [1.2, 1.2],
                ],
                "episode_index": [0, 0, 0, 1, 1, 1],
                "frame_index": [0, 1, 2, 0, 1, 2],
                "timestamp": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
                "next.reward": [0.0, 0.1, 1.0, 0.0, 0.2, 1.0],
                "next.done": [False, False, True, False, False, True],
                "next.success": [False, False, True, False, False, True],
                "index": [0, 1, 2, 3, 4, 5],
                "task_index": [0, 0, 0, 0, 0, 0],
            }
        ),
        root / "data/chunk-000/file-000.parquet",
    )


def test_external_lerobot_corpus_import_writes_real_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "out"
    _write_fixture_source(source)

    payload = import_lerobot_corpus_slice(
        repo_id="fixture/lerobot",
        output_dir=output,
        source_root=source,
        max_episodes=2,
        max_steps_per_episode=3,
    )

    assert payload["status"] == "ok_external_corpus_slice_imported_shadow_only"
    assert payload["download_executed"] is False
    assert payload["selected_episode_count"] == 2
    assert payload["selected_step_count"] == 6
    assert payload["replay_episode_count"] == 2
    assert payload["replay_step_count"] == 6
    assert payload["ready_for_shadow_eval"] is True
    assert payload["ready_for_training"] is False
    assert payload["provider_executed"] is False
    assert payload["gpu_training_executed"] is False
    assert payload["unitree_hardware_truth"] is False
    assert payload["promotion_eligible"] is False
    assert payload["phase7_authority_granted"] is False

    report = load_external_lerobot_corpus_import_report(
        payload["artifact_refs"]["report_path"]
    )
    receipts = load_external_corpus_quality_receipts(
        payload["artifact_refs"]["data_quality_receipts_path"]
    )
    assert report.report_id == payload["report_id"]
    assert report.quality_passed_count == report.quality_receipt_count
    assert {receipt.check_key for receipt in receipts} >= {
        "source_files_downloaded_or_present",
        "selected_episode_count_nonzero",
        "selected_step_count_nonzero",
        "action_schema_present",
        "timestamp_monotonic_per_episode",
        "promotion_gate_fail_closed",
    }
    assert Path(payload["artifact_refs"]["replay_manifest_path"]).exists()
    assert Path(payload["artifact_refs"]["replay_index_path"]).exists()
    assert Path(payload["artifact_refs"]["label_gap_ledger_path"]).exists()
    assert Path(payload["artifact_refs"]["governance_label_specs_path"]).exists()
    assert Path(
        payload["artifact_refs"]["economic_wm_external_corpus_ingestion_rows_path"]
    ).exists()

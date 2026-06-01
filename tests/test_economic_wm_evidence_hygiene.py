import json
from pathlib import Path

from src.world_model.economic_world_model import (
    load_evidence_hygiene_report,
    run_economic_wm_evidence_hygiene,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evidence_hygiene_passes_false_claims_and_existing_refs(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    output_dir = tmp_path / "out"
    receipt = artifact_root / "provider_receipt.json"
    _write_json(receipt, {"status": "ok"})
    _write_json(
        artifact_root / "report.json",
        {
            "provider_executed": True,
            "gpu_training_executed": False,
            "artifact_refs": {"provider_receipt_path": str(receipt)},
        },
    )

    report = run_economic_wm_evidence_hygiene(
        artifact_root=artifact_root,
        output_dir=output_dir,
    )

    assert report["status"] == "ok_evidence_hygiene_passed"
    assert report["blocking_issue_count"] == 0
    assert report["provider_gpu_hardware_claims_blocked"] is True
    loaded = load_evidence_hygiene_report(report["output_paths"]["report_path"])
    assert loaded.report_id == report["report_id"]


def test_evidence_hygiene_blocks_unevidenced_gpu_claim(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    output_dir = tmp_path / "out"
    _write_json(
        artifact_root / "report.json",
        {
            "gpu_training_executed": True,
            "ready_for_training": True,
        },
    )

    report = run_economic_wm_evidence_hygiene(
        artifact_root=artifact_root,
        output_dir=output_dir,
    )

    assert report["status"] == "blocked_evidence_hygiene_failed"
    assert report["blocking_issue_count"] >= 2
    assert report["provider_gpu_hardware_claims_blocked"] is False


def test_evidence_hygiene_blocks_missing_artifact_ref(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    output_dir = tmp_path / "out"
    _write_json(
        artifact_root / "report.json",
        {"artifact_refs": {"report_path": "missing_report.json"}},
    )

    report = run_economic_wm_evidence_hygiene(
        artifact_root=artifact_root,
        output_dir=output_dir,
    )

    assert report["status"] == "blocked_evidence_hygiene_failed"
    assert report["artifact_refs_resolved"] is False


def test_evidence_hygiene_blocks_retention_over_limit(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    output_dir = tmp_path / "out"
    large = artifact_root / "large.json"
    _write_json(large, {"status": "ok", "padding": "x" * 256})

    report = run_economic_wm_evidence_hygiene(
        artifact_root=artifact_root,
        output_dir=output_dir,
        max_local_artifact_bytes=128,
        advisory_local_artifact_bytes=64,
    )

    assert report["status"] == "blocked_evidence_hygiene_failed"
    assert report["retention_policy_passed"] is False

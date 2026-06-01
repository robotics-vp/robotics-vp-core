import json
from pathlib import Path

from src.runpod.provider_readiness_ledger import (
    build_provider_readiness_report,
    default_provider_readiness_entries,
    write_provider_readiness_report,
)


def test_provider_readiness_entries_cover_named_unwired_provider_families() -> None:
    entries = default_provider_readiness_entries()
    families = {entry.provider_family for entry in entries}

    assert {
        "SAM/SAM3D",
        "DINO/SigLIP",
        "V-JEPA2",
        "OpenVLA",
        "Isaac/Unitree",
        "Holosoma",
    }.issubset(families)
    assert all(entry.expected_receipts for entry in entries)
    assert all(entry.promotion_eligible is False for entry in entries)
    assert all(entry.authority_class == "provider_readiness_ledger_only" for entry in entries)


def test_provider_readiness_report_is_honest_about_local_prerequisites(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.runpod.provider_readiness_ledger.shutil.which",
        lambda name: "/usr/local/bin/runpodctl" if name == "runpodctl" else None,
    )

    report = build_provider_readiness_report(
        api_key="present",
        volume_id="vol-test",
    )

    assert report.status == "ready_to_prepare_provider_manifests"
    assert report.provider_execution_attempted is False
    assert report.provider_or_hardware_proof is False
    assert report.promotion_eligible is False
    assert report.local_prerequisite_status["runpodctl_installed"] is True
    assert report.local_prerequisite_status["RUNPOD_API_KEY_set"] is True
    assert report.local_prerequisite_status["RUNPOD_VOLUME_ID_set"] is True
    assert report.metadata["provider_bringup_manifest_preview"]["profile_id"] == "provider_bringup"


def test_write_provider_readiness_report_materializes_json_and_markdown(tmp_path: Path) -> None:
    summary = write_provider_readiness_report(tmp_path, api_key="", volume_id="")

    json_path = Path(summary["json_path"])
    markdown_path = Path(summary["markdown_path"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_path.exists()
    assert markdown_path.exists()
    assert payload["provider_execution_attempted"] is False
    assert payload["promotion_eligible"] is False
    assert payload["entry_count"] >= 6
    assert "Provider Readiness Ledger" in markdown_path.read_text(encoding="utf-8")

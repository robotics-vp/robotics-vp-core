from __future__ import annotations

import json
from pathlib import Path

from scripts.economic_world_model.compile_provider_bringup_readiness_ledger import (
    run_compile_provider_bringup_readiness_ledger,
)
from src.world_model.economic_world_model import (
    build_provider_bringup_ledger,
    load_provider_bringup_ledger_entries,
    load_provider_bringup_ledger_report,
    validate_provider_bringup_ledger,
)


def test_provider_bringup_ledger_covers_required_families_fail_closed() -> None:
    report, entries = build_provider_bringup_ledger()
    by_family = {entry.provider_family for entry in entries}
    by_key = {entry.provider_key: entry for entry in entries}

    assert report.status == "ok"
    assert report.all_entries_fail_closed is True
    assert report.entry_count == len(entries)
    assert report.launch_allowed_count == 0
    assert report.provider_bringup_ready_count == 0
    assert report.provider_executed is False
    assert report.gpu_executed is False
    assert report.runpod_launched is False
    assert report.weights_downloaded is False
    assert report.training_executed is False
    assert report.hardware_executed is False
    assert report.promotion_eligible is False
    assert not any(report.denied_gates.values())
    assert {
        "sam_sam3d",
        "dino_siglip",
        "vjepa2",
        "openvla",
        "isaac_unitree",
        "holosoma",
    } <= by_family
    assert report.missing_required_families == []

    assert by_key["dino_siglip_vision_backbone"].source_backlog_ids == [
        "vision_backbone_stub_replacement"
    ]
    assert by_key["isaac_unitree_runtime"].owner_wm == (
        "sim_synth_physics_and_embodiment_actuation"
    )
    assert by_key["holosoma_runtime"].run_class == "loop"
    assert by_key["openvla_semantic_teacher"].provider_family == "openvla"

    for entry in entries:
        assert entry.launch_allowed is False
        assert entry.provider_bringup_ready is False
        assert entry.provider_executed is False
        assert entry.runpod_launched is False
        assert entry.weights_downloaded is False
        assert entry.weights_written is False
        assert entry.training_executed is False
        assert entry.hardware_executed is False
        assert entry.promotion_eligible is False
        assert not any(entry.denied_gates.values())
        assert entry.expected_receipts
        assert entry.blocker_codes
        assert entry.surface_roles
        assert entry.unavailable_posture
        assert any(
            "TEMPLATE_ONLY_PROVIDER_LEDGER" in command
            for command in entry.command_templates
        )
        assert entry.manifest_stub["status"] == "pending"
        assert entry.manifest_stub["pod_id"] is None
        assert entry.manifest_stub["promotion_eligible"] is False

    validation = validate_provider_bringup_ledger(report=report, entries=entries)
    assert validation["status"] == "ok"
    assert validation["safe_for_template_storage"] is True
    assert validation["safe_for_launch"] is False
    assert validation["error_count"] == 0


def test_compile_provider_bringup_ledger_writes_artifacts(tmp_path: Path) -> None:
    payload = run_compile_provider_bringup_readiness_ledger(output_dir=tmp_path)

    assert payload["status"] == "ok"
    assert payload["all_entries_fail_closed"] is True
    assert payload["launch_allowed_count"] == 0
    assert payload["provider_bringup_ready_count"] == 0
    assert payload["provider_executed"] is False
    assert payload["promotion_eligible"] is False
    assert payload["validation"]["safe_for_template_storage"] is True
    assert payload["validation"]["safe_for_launch"] is False

    refs = payload["artifact_refs"]
    report = load_provider_bringup_ledger_report(refs["report_path"])
    entries = load_provider_bringup_ledger_entries(refs["entries_path"])
    assert report.report_id == payload["report_id"]
    assert len(entries) == payload["entry_count"]
    assert Path(refs["markdown_path"]).exists()
    assert Path(refs["validation_path"]).exists()
    assert Path(refs["validation_markdown_path"]).exists()

    manifest_dir = Path(refs["manifest_template_dir"])
    manifests = sorted(manifest_dir.glob("*.manifest_template.json"))
    assert len(manifests) == payload["entry_count"]
    first = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert first["task"].startswith("[TEMPLATE ONLY]")
    assert first["status"] == "pending"
    assert first["pod_id"] is None
    assert any("TEMPLATE_ONLY_PROVIDER_LEDGER" in command for command in first["commands"])

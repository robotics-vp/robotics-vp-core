from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.training.training_manifest import (
    TRAINING_RUNTIME_MANIFEST_SCHEMA_VERSION,
    TrainingRuntimeManifest,
    write_training_runtime_manifest,
)
from src.world_model.perception_grounding.benchmark_evidence import (
    load_perception_benchmark_evidence,
)
from src.world_model.perception_grounding.benchmark_evidence_emitter import (
    emit_provider_adapter_benchmark_evidence,
)
from src.world_model.perception_grounding.receipts import ProviderInvocationReceipt


def _receipt(
    *,
    provider_kind: str,
    status: str = "success",
    quality: float = 0.8,
    fallback: bool = False,
    token_count: int = 3,
) -> dict[str, object]:
    return ProviderInvocationReceipt(
        receipt_id=f"receipt_{provider_kind}_{status}_{int(fallback)}",
        provider_id={
            "sam_calibration": "sam_3_1",
            "vision_backbone_projection": "dinov2_vit_l_14",
            "depth_metric_calibration": "depth_anything_v2",
            "vjepa_temporal_alignment": "vjepa2",
        }.get(provider_kind, "provider"),
        provider_kind=provider_kind,
        invocation_status=status,
        output_quality_score=quality,
        latency_ms=4.0,
        output_token_count=token_count,
        fallback_used=fallback,
        fallback_reason="fallback" if fallback else "",
        metadata={"runtime_provider_backed": status == "success" and not fallback},
    ).to_dict()


def _write_receipt_payload(tmp_path: Path, receipts: list[dict[str, object]]) -> Path:
    path = tmp_path / "provider_receipts.json"
    path.write_text(
        json.dumps({"metadata": {"provider_adapter_receipts": receipts}}, indent=2),
        encoding="utf-8",
    )
    return path


def _write_training_manifest(tmp_path: Path) -> Path:
    artifact_path = tmp_path / "training_summary.json"
    artifact_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    manifest = TrainingRuntimeManifest(
        schema_version=TRAINING_RUNTIME_MANIFEST_SCHEMA_VERSION,
        run_id="perception_provider_run",
        training_kind="perception_provider_adapter",
        status="completed",
        seed=7,
        plan_id="vision_backbone_projection",
        plan_sha="plan_sha",
        started_at="2026-05-11T00:00:00+00:00",
        ended_at="2026-05-11T00:05:00+00:00",
        config_path=None,
        config_digest="cfg",
        replay_dataset_dir=None,
        replay_manifest_digest=None,
        replay_dataset_summary={"num_records": 2},
        objective_profile_snapshot={},
        promotion_policy_snapshot={"policy": "benchmark_gated"},
        source_domain_coverage={"source_domain_counts": {"external": 2}},
        receipt_label_coverage={"provider_invocation_receipts": 2},
        artifact_paths={"training_summary": str(artifact_path)},
    )
    manifest_path = tmp_path / "training_runtime_manifest.json"
    write_training_runtime_manifest(manifest_path, manifest)
    return manifest_path


def test_provider_adapter_emitter_writes_receipt_backed_provisional_evidence(
    tmp_path,
) -> None:
    receipts_path = _write_receipt_payload(
        tmp_path,
        [
            _receipt(provider_kind="vision_backbone_projection", quality=0.86),
            _receipt(
                provider_kind="vision_backbone_projection",
                status="error",
                quality=0.0,
                fallback=True,
                token_count=0,
            ),
            _receipt(provider_kind="vjepa_temporal_alignment", quality=0.9),
        ],
    )
    manifest_path = _write_training_manifest(tmp_path)
    output_path = tmp_path / "vision_provider_benchmark_evidence.json"

    emission = emit_provider_adapter_benchmark_evidence(
        provider_receipts_path=receipts_path,
        provider_kind="vision_backbone_projection",
        output_path=output_path,
        training_manifest_path=manifest_path,
    )

    assert output_path.exists()
    assert emission.provider_kind == "vision_backbone_projection"
    assert emission.matched_receipt_count == 2
    assert emission.success_count == 1
    assert emission.fallback_count == 1
    assert emission.training_manifest_ref_status == "present"

    evidence = load_perception_benchmark_evidence(output_path).to_dict()
    assert evidence["subsystem_key"] == "vision_backbone_projection"
    assert evidence["benchmark_evidence_present"] is True
    assert evidence["evidence_source_provisional"] is True
    assert evidence["promotion_eligible"] is False
    assert evidence["evidence_truth_class"] == "provider_backed"
    assert evidence["token_source_kind"] == "vision_backbone_projection"
    assert evidence["metadata"]["promotion_claim"] == "not_implied_by_emitter"
    assert evidence["metadata"]["training_manifest_run_id"] == "perception_provider_run"


def test_provider_adapter_emitter_honors_nonprovisional_metric_report(tmp_path) -> None:
    receipts_path = _write_receipt_payload(
        tmp_path,
        [_receipt(provider_kind="sam_calibration", quality=0.92)],
    )
    metric_report = tmp_path / "sam_metric_report.json"
    metric_report.write_text(
        json.dumps(
            {
                "metrics": {
                    "benchmark_evidence_present": True,
                    "evidence_source_provisional": False,
                    "annotation_supervision_score": 0.91,
                    "held_out_label_agreement": 0.88,
                    "downstream_usefulness_score": 0.84,
                    "receipt_consistency": 0.93,
                    "gate_score": 0.89,
                    "promotion_eligible": True,
                }
            }
        ),
        encoding="utf-8",
    )

    emission = emit_provider_adapter_benchmark_evidence(
        provider_receipts_path=receipts_path,
        provider_kind="sam_calibration",
        metric_report_path=metric_report,
    )

    evidence = emission.evidence.to_dict()
    assert emission.metric_report_ref_status == "present"
    assert evidence["evidence_source_provisional"] is False
    assert evidence["promotion_eligible"] is True
    assert evidence["gate_score"] == 0.89
    assert evidence["metadata"]["metric_report_digest"]


def test_provider_adapter_benchmark_evidence_cli(tmp_path) -> None:
    receipts_path = _write_receipt_payload(
        tmp_path,
        [_receipt(provider_kind="vjepa_temporal_alignment", quality=0.83)],
    )
    output_path = tmp_path / "vjepa_benchmark_evidence.json"
    summary_path = tmp_path / "emission_summary.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/emit_perception_provider_adapter_benchmark_evidence.py",
            "--provider-receipts",
            str(receipts_path),
            "--provider-kind",
            "vjepa_temporal_alignment",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        capture_output=True,
    )

    assert output_path.exists()
    assert summary_path.exists()
    cli_summary = json.loads(result.stdout)
    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert cli_summary["benchmark_evidence_present"] is True
    assert cli_summary["provider_kind"] == "vjepa_temporal_alignment"
    assert cli_summary["success_count"] == 1
    assert cli_summary["evidence_source_provisional"] is True
    assert persisted_summary["evidence"]["subsystem_key"] == "vjepa_temporal_alignment"

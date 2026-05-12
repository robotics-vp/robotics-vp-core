"""Artifact emitters for Perception / Grounding benchmark evidence.

These helpers turn persisted Phase 2 annotation exports into typed
``PerceptionBenchmarkEvidence`` artifacts that promotion resolvers can consume
later.  The emitter is deliberately narrow: producing an artifact proves that
the benchmark path ran, not that a seam is promoted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.training.training_manifest import load_training_runtime_manifest
from src.utils.config_digest import sha256_file

from .annotation_export import load_annotation_export_json
from .benchmark_evidence import (
    PerceptionBenchmarkEvidence,
    build_perception_benchmark_evidence,
    write_perception_benchmark_evidence,
)
from .seam_registry import PerceptionSeamRegistry


ANNOTATION_BENCHMARK_SEAM_TYPES = {
    "annotation_bridge_projection",
    "scene_graph_transformer",
}
PROVIDER_ADAPTER_BENCHMARK_SEAM_TYPES = {
    "sam_calibration",
    "vision_backbone_projection",
    "depth_metric_calibration",
    "vjepa_temporal_alignment",
}

PROVIDER_ADAPTER_LATENCY_BUDGET_MS = {
    "sam_calibration": 50.0,
    "vision_backbone_projection": 20.0,
    "depth_metric_calibration": 40.0,
    "vjepa_temporal_alignment": 120.0,
}


@dataclass(frozen=True)
class AnnotationBenchmarkEvidenceEmission:
    """Summary of one persisted annotation-export benchmark-evidence run."""

    evidence: PerceptionBenchmarkEvidence
    evidence_digest: str
    seam_type: str
    seam_id: str
    source_annotation_export_path: str
    output_path: Optional[str] = None
    loaded_annotation_record_count: int = 0
    checkpoint_path: Optional[str] = None
    checkpoint_ref_status: str = "not_supplied"
    device: str = "cpu"
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "annotation_benchmark_evidence_emission_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seam_type": self.seam_type,
            "seam_id": self.seam_id,
            "source_annotation_export_path": self.source_annotation_export_path,
            "output_path": self.output_path,
            "loaded_annotation_record_count": int(self.loaded_annotation_record_count),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_ref_status": self.checkpoint_ref_status,
            "device": self.device,
            "evidence_digest": self.evidence_digest,
            "evidence": self.evidence.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProviderAdapterBenchmarkEvidenceEmission:
    """Summary of one provider-adapter benchmark-evidence emission."""

    evidence: PerceptionBenchmarkEvidence
    evidence_digest: str
    provider_kind: str
    source_provider_receipts_path: str
    output_path: Optional[str] = None
    matched_receipt_count: int = 0
    success_count: int = 0
    fallback_count: int = 0
    checkpoint_path: Optional[str] = None
    checkpoint_ref_status: str = "not_supplied"
    training_manifest_path: Optional[str] = None
    training_manifest_ref_status: str = "not_supplied"
    metric_report_path: Optional[str] = None
    metric_report_ref_status: str = "not_supplied"
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "provider_adapter_benchmark_evidence_emission_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_kind": self.provider_kind,
            "source_provider_receipts_path": self.source_provider_receipts_path,
            "output_path": self.output_path,
            "matched_receipt_count": int(self.matched_receipt_count),
            "success_count": int(self.success_count),
            "fallback_count": int(self.fallback_count),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_ref_status": self.checkpoint_ref_status,
            "training_manifest_path": self.training_manifest_path,
            "training_manifest_ref_status": self.training_manifest_ref_status,
            "metric_report_path": self.metric_report_path,
            "metric_report_ref_status": self.metric_report_ref_status,
            "evidence_digest": self.evidence_digest,
            "evidence": self.evidence.to_dict(),
            "metadata": dict(self.metadata),
        }


def _checkpoint_ref_status(path: Optional[str | Path]) -> tuple[Optional[Path], str]:
    if path is None:
        return None, "not_supplied"
    resolved = Path(path).resolve()
    return resolved, "present" if resolved.exists() else "missing_fresh_init"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_provider_invocation_receipts(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        receipts: list[dict[str, Any]] = []
        for item in payload:
            receipts.extend(_extract_provider_invocation_receipts(item))
        return receipts
    if not isinstance(payload, Mapping):
        return []

    if "provider_kind" in payload and "invocation_status" in payload:
        return [dict(payload)]

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(
        metadata.get("provider_adapter_receipts"),
        list,
    ):
        return [
            dict(item)
            for item in metadata.get("provider_adapter_receipts", [])
            if isinstance(item, Mapping)
        ]

    for key in (
        "provider_adapter_receipts",
        "provider_invocation_receipts",
        "receipts",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            receipts = []
            for item in value:
                receipts.extend(_extract_provider_invocation_receipts(item))
            return receipts
    return []


def _training_manifest_metadata(
    training_manifest_path: Optional[str | Path],
) -> tuple[Optional[str], str, dict[str, Any]]:
    if training_manifest_path is None:
        return None, "not_supplied", {}
    path = Path(training_manifest_path).resolve()
    if not path.exists():
        return str(path), "missing", {"training_manifest_path": str(path)}
    try:
        manifest = load_training_runtime_manifest(path)
        return (
            str(path),
            "present",
            {
                "training_manifest_path": str(path),
                "training_manifest_digest": sha256_file(path),
                "training_manifest_run_id": manifest.run_id,
                "training_manifest_training_kind": manifest.training_kind,
                "training_manifest_status": manifest.status,
                "training_manifest_promotion_evidence_path": (
                    manifest.promotion_evidence_path
                ),
                "training_manifest_checkpoint_registry_path": (
                    manifest.checkpoint_registry_path
                ),
                "training_manifest_artifact_keys": sorted(
                    manifest.artifact_paths.keys()
                ),
            },
        )
    except Exception as exc:
        return (
            str(path),
            "unreadable",
            {
                "training_manifest_path": str(path),
                "training_manifest_digest": sha256_file(path),
                "training_manifest_parse_error": str(exc)[:200],
            },
        )


def _metric_report_metadata(
    metric_report_path: Optional[str | Path],
) -> tuple[Optional[str], str, dict[str, Any], dict[str, Any]]:
    if metric_report_path is None:
        return None, "not_supplied", {}, {}
    path = Path(metric_report_path).resolve()
    if not path.exists():
        return str(path), "missing", {}, {"metric_report_path": str(path)}
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError("metric report must be a JSON object")
    metrics = dict(payload.get("metrics", payload))
    metadata = {
        "metric_report_path": str(path),
        "metric_report_digest": sha256_file(path),
        "metric_report_keys": sorted(metrics.keys()),
    }
    return str(path), "present", metrics, metadata


def _clip01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    return max(0.0, min(1.0, number))


def _provider_metrics_from_receipts(
    *,
    provider_kind: str,
    receipts: Sequence[Mapping[str, Any]],
    metric_report: Optional[Mapping[str, Any]] = None,
    evidence_source_provisional: Optional[bool] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    total = len(receipts)
    success_receipts = [
        receipt
        for receipt in receipts
        if str(receipt.get("invocation_status", "")) == "success"
        and not bool(receipt.get("fallback_used", False))
    ]
    fallback_count = sum(
        1 for receipt in receipts if bool(receipt.get("fallback_used", False))
    )
    success_count = len(success_receipts)
    success_rate = success_count / float(total or 1)
    fallback_rate = fallback_count / float(total or 1)
    output_quality_scores = [
        _clip01(receipt.get("output_quality_score", 0.0))
        for receipt in success_receipts
    ]
    mean_output_quality = (
        sum(output_quality_scores) / float(len(output_quality_scores))
        if output_quality_scores
        else 0.0
    )
    token_counts = [
        int(receipt.get("output_token_count", 0) or 0) for receipt in success_receipts
    ]
    token_presence_score = 1.0 if any(count > 0 for count in token_counts) else 0.0
    latency_values = [
        max(0.0, float(receipt.get("latency_ms", 0.0) or 0.0))
        for receipt in success_receipts
    ]
    mean_latency_ms = (
        sum(latency_values) / float(len(latency_values)) if latency_values else 0.0
    )
    latency_budget = PROVIDER_ADAPTER_LATENCY_BUDGET_MS.get(provider_kind, 50.0)
    latency_budget_score = (
        max(0.0, 1.0 - mean_latency_ms / max(latency_budget, 1e-6))
        if success_receipts
        else 0.0
    )

    report = dict(metric_report or {})
    if evidence_source_provisional is None:
        evidence_source_provisional = bool(
            report.get(
                "evidence_source_provisional",
                True,
            )
        )
    metrics = {
        "benchmark_evidence_present": total > 0,
        "evidence_source_provisional": bool(evidence_source_provisional),
        "evidence_truth_class": (
            "provider_backed" if success_count > 0 else "unavailable"
        ),
        "token_source_kind": provider_kind,
        "source_record_count": total,
        "annotation_supervision_score": mean_output_quality,
        "held_out_label_agreement": success_rate,
        "downstream_usefulness_score": (
            0.5 * mean_output_quality
            + 0.3 * token_presence_score
            + 0.2 * latency_budget_score
        ),
        "receipt_consistency": max(0.0, success_rate * (1.0 - fallback_rate)),
    }
    for key in (
        "benchmark_evidence_present",
        "evidence_source_provisional",
        "evidence_truth_class",
        "token_source_kind",
        "source_record_count",
        "annotation_supervision_score",
        "held_out_label_agreement",
        "downstream_usefulness_score",
        "receipt_consistency",
        "gate_score",
        "promotion_eligible",
    ):
        if key in report:
            metrics[key] = report[key]

    summary = {
        "receipt_count": total,
        "success_count": success_count,
        "fallback_count": fallback_count,
        "success_rate": success_rate,
        "fallback_rate": fallback_rate,
        "mean_output_quality": mean_output_quality,
        "mean_latency_ms": mean_latency_ms,
        "latency_budget_ms": latency_budget,
        "latency_budget_score": latency_budget_score,
        "token_presence_score": token_presence_score,
    }
    return metrics, summary


def emit_annotation_benchmark_evidence(
    *,
    annotation_export_path: str | Path,
    seam_type: str = "scene_graph_transformer",
    output_path: Optional[str | Path] = None,
    seam_id: Optional[str] = None,
    checkpoint_path: Optional[str | Path] = None,
    checkpoint_dir: Optional[str | Path] = None,
    hyperparams: Optional[Mapping[str, Any]] = None,
    device: str = "cpu",
    evidence_source_provisional: Optional[bool] = None,
    held_out_fraction: float = 0.2,
    d_token: int = 128,
    d_edge: int = 64,
    n_categories: int = 16,
    evidence_metadata: Optional[Mapping[str, Any]] = None,
) -> AnnotationBenchmarkEvidenceEmission:
    """Emit benchmark evidence from a persisted annotation export.

    ``annotation_export_path`` is expected to point at the JSON produced by
    ``save_annotation_export_json``.  The function may evaluate a fresh seam
    instance when no checkpoint is supplied; the output remains a benchmark
    artifact only and does not imply promotion.
    """
    from src.training.perception_seam_data import evaluate_seam_on_annotations

    if seam_type not in ANNOTATION_BENCHMARK_SEAM_TYPES:
        supported = ", ".join(sorted(ANNOTATION_BENCHMARK_SEAM_TYPES))
        raise ValueError(
            f"Unsupported annotation benchmark seam_type={seam_type!r}. "
            f"Supported: {supported}"
        )

    source_path = Path(annotation_export_path).resolve()
    records = load_annotation_export_json(source_path)

    ckpt_path, checkpoint_ref_status = _checkpoint_ref_status(checkpoint_path)

    registry = PerceptionSeamRegistry(
        checkpoint_dir=checkpoint_dir,
        default_device=device,
    )
    resolved_seam_id = seam_id or f"{seam_type}_annotation_benchmark"
    descriptor = registry.register_seam(
        seam_type=seam_type,
        seam_id=resolved_seam_id,
        posture="auto",
        hyperparams=dict(hyperparams or {}),
        checkpoint_path=str(ckpt_path) if ckpt_path is not None else None,
        device=device,
    )
    seam = registry.load_seam(resolved_seam_id)
    descriptor = registry.get_descriptor(resolved_seam_id) or descriptor

    metrics = evaluate_seam_on_annotations(
        seam=seam,
        seam_type=seam_type,
        annotation_records=records,
        d_token=d_token,
        d_edge=d_edge,
        n_categories=n_categories,
        held_out_fraction=held_out_fraction,
        evidence_source_provisional=evidence_source_provisional,
    )
    metadata = {
        "emitter": "annotation_export_benchmark_evidence",
        "source_annotation_export_path": str(source_path),
        "loaded_annotation_record_count": len(records),
        "seam_descriptor": descriptor.to_dict(),
        "checkpoint_ref_status": checkpoint_ref_status,
        "promotion_claim": "not_implied_by_emitter",
        **dict(evidence_metadata or {}),
    }
    evidence = build_perception_benchmark_evidence(
        subsystem_key=seam_type,
        metrics=metrics,
        source_record_count=int(metrics.get("source_record_count", len(records))),
        source_artifact_path=source_path,
        metadata=metadata,
    )

    output_ref: Optional[str] = None
    if output_path is not None:
        output = Path(output_path).resolve()
        evidence_digest = write_perception_benchmark_evidence(output, evidence)
        output_ref = str(output)
    else:
        evidence_digest = evidence.evidence_digest

    return AnnotationBenchmarkEvidenceEmission(
        evidence=evidence,
        evidence_digest=evidence_digest,
        seam_type=seam_type,
        seam_id=resolved_seam_id,
        source_annotation_export_path=str(source_path),
        output_path=output_ref,
        loaded_annotation_record_count=len(records),
        checkpoint_path=str(ckpt_path) if ckpt_path is not None else None,
        checkpoint_ref_status=checkpoint_ref_status,
        device=device,
        metadata={
            "registry_summary": registry.summary(),
            "output_sha256": (
                sha256_file(Path(output_ref)) if output_ref is not None else None
            ),
        },
    )


def emit_provider_adapter_benchmark_evidence(
    *,
    provider_receipts_path: str | Path,
    provider_kind: str,
    output_path: Optional[str | Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    training_manifest_path: Optional[str | Path] = None,
    metric_report_path: Optional[str | Path] = None,
    evidence_source_provisional: Optional[bool] = None,
    evidence_metadata: Optional[Mapping[str, Any]] = None,
) -> ProviderAdapterBenchmarkEvidenceEmission:
    """Emit provider-adapter benchmark evidence from invocation receipts.

    Receipt-only evidence is provisional by default.  A metric report can carry
    non-provisional held-out scores when GPU/provider benchmark runs exist.
    """
    if provider_kind not in PROVIDER_ADAPTER_BENCHMARK_SEAM_TYPES:
        supported = ", ".join(sorted(PROVIDER_ADAPTER_BENCHMARK_SEAM_TYPES))
        raise ValueError(
            f"Unsupported provider benchmark provider_kind={provider_kind!r}. "
            f"Supported: {supported}"
        )

    source_path = Path(provider_receipts_path).resolve()
    raw_payload = _load_json(source_path)
    receipts = [
        receipt
        for receipt in _extract_provider_invocation_receipts(raw_payload)
        if str(receipt.get("provider_kind", "")) == provider_kind
    ]
    if not receipts:
        raise ValueError(
            f"No provider invocation receipts found for provider_kind={provider_kind!r}"
        )

    ckpt_path, checkpoint_ref_status = _checkpoint_ref_status(checkpoint_path)
    (
        manifest_path_ref,
        training_manifest_ref_status,
        manifest_metadata,
    ) = _training_manifest_metadata(training_manifest_path)
    (
        metric_report_path_ref,
        metric_report_ref_status,
        metric_report,
        metric_metadata,
    ) = _metric_report_metadata(metric_report_path)

    metrics, receipt_summary = _provider_metrics_from_receipts(
        provider_kind=provider_kind,
        receipts=receipts,
        metric_report=metric_report,
        evidence_source_provisional=evidence_source_provisional,
    )
    metadata = {
        "emitter": "provider_adapter_benchmark_evidence",
        "source_provider_receipts_path": str(source_path),
        "source_provider_receipts_digest": sha256_file(source_path),
        "provider_kind": provider_kind,
        "receipt_summary": receipt_summary,
        "checkpoint_ref_status": checkpoint_ref_status,
        "training_manifest_ref_status": training_manifest_ref_status,
        "metric_report_ref_status": metric_report_ref_status,
        "promotion_claim": "not_implied_by_emitter",
        **manifest_metadata,
        **metric_metadata,
        **dict(evidence_metadata or {}),
    }
    evidence = build_perception_benchmark_evidence(
        subsystem_key=provider_kind,
        metrics=metrics,
        source_record_count=int(metrics.get("source_record_count", len(receipts))),
        source_artifact_path=source_path,
        metadata=metadata,
    )

    output_ref: Optional[str] = None
    if output_path is not None:
        output = Path(output_path).resolve()
        evidence_digest = write_perception_benchmark_evidence(output, evidence)
        output_ref = str(output)
    else:
        evidence_digest = evidence.evidence_digest

    return ProviderAdapterBenchmarkEvidenceEmission(
        evidence=evidence,
        evidence_digest=evidence_digest,
        provider_kind=provider_kind,
        source_provider_receipts_path=str(source_path),
        output_path=output_ref,
        matched_receipt_count=len(receipts),
        success_count=int(receipt_summary["success_count"]),
        fallback_count=int(receipt_summary["fallback_count"]),
        checkpoint_path=str(ckpt_path) if ckpt_path is not None else None,
        checkpoint_ref_status=checkpoint_ref_status,
        training_manifest_path=manifest_path_ref,
        training_manifest_ref_status=training_manifest_ref_status,
        metric_report_path=metric_report_path_ref,
        metric_report_ref_status=metric_report_ref_status,
        metadata={
            "output_sha256": (
                sha256_file(Path(output_ref)) if output_ref is not None else None
            ),
            "receipt_summary": receipt_summary,
        },
    )


__all__ = [
    "ANNOTATION_BENCHMARK_SEAM_TYPES",
    "AnnotationBenchmarkEvidenceEmission",
    "PROVIDER_ADAPTER_BENCHMARK_SEAM_TYPES",
    "ProviderAdapterBenchmarkEvidenceEmission",
    "emit_annotation_benchmark_evidence",
    "emit_provider_adapter_benchmark_evidence",
]

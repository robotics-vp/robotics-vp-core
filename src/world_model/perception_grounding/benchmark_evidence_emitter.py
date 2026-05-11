"""Artifact emitters for Perception / Grounding benchmark evidence.

These helpers turn persisted Phase 2 annotation exports into typed
``PerceptionBenchmarkEvidence`` artifacts that promotion resolvers can consume
later.  The emitter is deliberately narrow: producing an artifact proves that
the benchmark path ran, not that a seam is promoted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.training.perception_seam_data import evaluate_seam_on_annotations
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
            "loaded_annotation_record_count": int(
                self.loaded_annotation_record_count
            ),
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_ref_status": self.checkpoint_ref_status,
            "device": self.device,
            "evidence_digest": self.evidence_digest,
            "evidence": self.evidence.to_dict(),
            "metadata": dict(self.metadata),
        }


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
    if seam_type not in ANNOTATION_BENCHMARK_SEAM_TYPES:
        supported = ", ".join(sorted(ANNOTATION_BENCHMARK_SEAM_TYPES))
        raise ValueError(
            f"Unsupported annotation benchmark seam_type={seam_type!r}. "
            f"Supported: {supported}"
        )

    source_path = Path(annotation_export_path).resolve()
    records = load_annotation_export_json(source_path)

    ckpt_path = Path(checkpoint_path).resolve() if checkpoint_path else None
    if ckpt_path is None:
        checkpoint_ref_status = "not_supplied"
    elif ckpt_path.exists():
        checkpoint_ref_status = "present"
    else:
        checkpoint_ref_status = "missing_fresh_init"

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


__all__ = [
    "ANNOTATION_BENCHMARK_SEAM_TYPES",
    "AnnotationBenchmarkEvidenceEmission",
    "emit_annotation_benchmark_evidence",
]

"""Artifact serialization for Map-First pseudo-supervision."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


MAP_FIRST_VERSION = "v1"
MAP_FIRST_PREFIX = "map_first_supervision_v1/"


@dataclass
class MapFirstSummary:
    """Summary metrics for Map-First pseudo-supervision."""

    dynamic_pct: float = 0.0
    static_map_coverage: float = 0.0
    depth_coverage_pct: float = 0.0
    residual_p50: float = 0.0
    residual_p90: float = 0.0
    semantic_stability_score: float = 0.0
    usable_pct_after_gating: float = 0.0
    map_first_quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dynamic_pct": self.dynamic_pct,
            "static_map_coverage": self.static_map_coverage,
            "depth_coverage_pct": self.depth_coverage_pct,
            "residual_p50": self.residual_p50,
            "residual_p90": self.residual_p90,
            "semantic_stability_score": self.semantic_stability_score,
            "usable_pct_after_gating": self.usable_pct_after_gating,
            "map_first_quality_score": self.map_first_quality_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MapFirstSummary":
        return cls(
            dynamic_pct=float(data.get("dynamic_pct", 0.0)),
            static_map_coverage=float(data.get("static_map_coverage", 0.0)),
            depth_coverage_pct=float(data.get("depth_coverage_pct", 0.0)),
            residual_p50=float(data.get("residual_p50", 0.0)),
            residual_p90=float(data.get("residual_p90", 0.0)),
            semantic_stability_score=float(data.get("semantic_stability_score", 0.0)),
            usable_pct_after_gating=float(data.get("usable_pct_after_gating", 0.0)),
            map_first_quality_score=float(data.get("map_first_quality_score", 0.0)),
        )


@dataclass
class MapFirstArtifact:
    """Container for Map-First supervision outputs."""

    dynamic_evidence: np.ndarray
    dynamic_mask: np.ndarray
    residual_mean: np.ndarray
    boxes3d: np.ndarray
    confidence: np.ndarray
    densify_depth: Optional[np.ndarray] = None
    densify_mask: Optional[np.ndarray] = None
    densify_world_points: Optional[np.ndarray] = None
    densify_world_mask: Optional[np.ndarray] = None
    semantics_stable: Optional[np.ndarray] = None
    semantics_stability: Optional[np.ndarray] = None
    static_map_centroids: Optional[np.ndarray] = None
    static_map_counts: Optional[np.ndarray] = None
    static_map_semantics: Optional[np.ndarray] = None
    evidence_occlusion: Optional[np.ndarray] = None
    vla_class_probs: Optional[np.ndarray] = None
    vla_confidence: Optional[np.ndarray] = None
    vla_embed: Optional[np.ndarray] = None
    vla_provenance_json: Optional[str] = None

    def to_npz(self, summary: Optional[MapFirstSummary] = None, export_float16: bool = True) -> Dict[str, np.ndarray]:
        """Serialize to numpy arrays with versioned prefix."""
        data: Dict[str, np.ndarray] = {
            f"{MAP_FIRST_PREFIX}version": np.array([MAP_FIRST_VERSION], dtype="U8"),
            f"{MAP_FIRST_PREFIX}dynamic_evidence": self.dynamic_evidence.astype(np.float32),
            f"{MAP_FIRST_PREFIX}dynamic_mask": self.dynamic_mask.astype(bool),
            f"{MAP_FIRST_PREFIX}residual_mean": self.residual_mean.astype(np.float32),
            f"{MAP_FIRST_PREFIX}boxes3d": self.boxes3d.astype(np.float32),
            f"{MAP_FIRST_PREFIX}confidence": self.confidence.astype(np.float32),
        }

        if self.densify_depth is not None:
            depth = self.densify_depth.astype(np.float16 if export_float16 else np.float32)
            data[f"{MAP_FIRST_PREFIX}densify_depth"] = depth
        if self.densify_mask is not None:
            data[f"{MAP_FIRST_PREFIX}densify_mask"] = self.densify_mask.astype(np.uint8)
        if self.densify_world_points is not None:
            pts_dtype = np.float16 if export_float16 else np.float32
            data[f"{MAP_FIRST_PREFIX}densify_world_points"] = self.densify_world_points.astype(pts_dtype)
        if self.densify_world_mask is not None:
            data[f"{MAP_FIRST_PREFIX}densify_world_mask"] = self.densify_world_mask.astype(np.uint8)
        if self.semantics_stable is not None:
            sem_dtype = np.float16 if export_float16 else np.float32
            data[f"{MAP_FIRST_PREFIX}semantics_stable"] = self.semantics_stable.astype(sem_dtype)
        if self.semantics_stability is not None:
            data[f"{MAP_FIRST_PREFIX}meta_semantics_stability"] = self.semantics_stability.astype(np.float32)
        if self.static_map_centroids is not None:
            data[f"{MAP_FIRST_PREFIX}static_map_centroids"] = self.static_map_centroids.astype(np.float32)
        if self.static_map_counts is not None:
            data[f"{MAP_FIRST_PREFIX}static_map_counts"] = self.static_map_counts.astype(np.float32)
        if self.static_map_semantics is not None:
            data[f"{MAP_FIRST_PREFIX}static_map_semantics"] = self.static_map_semantics.astype(np.float32)
        if self.evidence_occlusion is not None:
            data[f"{MAP_FIRST_PREFIX}evidence_occlusion"] = self.evidence_occlusion.astype(np.float32)

        data[f"{MAP_FIRST_PREFIX}evidence_dynamics_score"] = self.dynamic_evidence.astype(np.float32)
        data[f"{MAP_FIRST_PREFIX}evidence_geom_residual"] = self.residual_mean.astype(np.float32)
        if self.semantics_stable is not None:
            sem_dtype = np.float16 if export_float16 else np.float32
            data[f"{MAP_FIRST_PREFIX}evidence_map_semantics"] = self.semantics_stable.astype(sem_dtype)
        if self.semantics_stability is not None:
            data[f"{MAP_FIRST_PREFIX}evidence_map_stability"] = self.semantics_stability.astype(np.float32)
        if self.vla_class_probs is not None:
            sem_dtype = np.float16 if export_float16 else np.float32
            data[f"{MAP_FIRST_PREFIX}vla_class_probs"] = self.vla_class_probs.astype(sem_dtype)
        if self.vla_confidence is not None:
            data[f"{MAP_FIRST_PREFIX}vla_confidence"] = self.vla_confidence.astype(np.float32)
        if self.vla_embed is not None:
            emb_dtype = np.float16 if export_float16 else np.float32
            data[f"{MAP_FIRST_PREFIX}vla_embed"] = self.vla_embed.astype(emb_dtype)
        if self.vla_provenance_json is not None:
            data[f"{MAP_FIRST_PREFIX}vla_provenance_json"] = np.array([self.vla_provenance_json], dtype="U2048")

        if summary is not None:
            summary_json = json.dumps(summary.to_dict())
            data[f"{MAP_FIRST_PREFIX}summary_json"] = np.array([summary_json], dtype="U2048")

        return data


def compute_map_first_summary(
    residual_mean: np.ndarray,
    dynamic_mask: np.ndarray,
    coverage: np.ndarray,
    visibility_weight: np.ndarray,
    confidence: np.ndarray,
    densify_mask: Optional[np.ndarray] = None,
    semantic_stability: Optional[np.ndarray] = None,
) -> MapFirstSummary:
    """Compute summary metrics from outputs."""
    residuals = residual_mean[np.isfinite(residual_mean)]
    residual_p50 = float(np.percentile(residuals, 50)) if residuals.size > 0 else 0.0
    residual_p90 = float(np.percentile(residuals, 90)) if residuals.size > 0 else 0.0

    visible = visibility_weight >= 0.2
    if np.any(visible):
        dynamic_pct = float(np.mean(dynamic_mask[visible]))
        static_map_coverage = float(np.mean(coverage[visible]))
    else:
        dynamic_pct = float(np.mean(dynamic_mask)) if dynamic_mask.size > 0 else 0.0
        static_map_coverage = float(np.mean(coverage)) if coverage.size > 0 else 0.0

    depth_coverage_pct = 0.0
    if densify_mask is not None and densify_mask.size > 0:
        depth_coverage_pct = float(np.mean(densify_mask.astype(np.float32)))

    semantic_stability_score = 0.0
    if semantic_stability is not None and semantic_stability.size > 0:
        semantic_stability_score = float(np.mean(semantic_stability))

    usable_mask = (~dynamic_mask) & (confidence >= 0.2)
    usable_pct = float(np.mean(usable_mask)) if usable_mask.size > 0 else 0.0

    components = [static_map_coverage, 1.0 - dynamic_pct]
    if depth_coverage_pct > 0.0:
        components.append(depth_coverage_pct)
    if semantic_stability_score > 0.0:
        components.append(semantic_stability_score)
    quality = float(np.clip(np.mean(components), 0.0, 1.0)) if components else 0.0

    return MapFirstSummary(
        dynamic_pct=dynamic_pct,
        static_map_coverage=static_map_coverage,
        depth_coverage_pct=depth_coverage_pct,
        residual_p50=residual_p50,
        residual_p90=residual_p90,
        semantic_stability_score=semantic_stability_score,
        usable_pct_after_gating=usable_pct,
        map_first_quality_score=quality,
    )

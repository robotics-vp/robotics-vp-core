"""Helpers to attach embodiment outputs to DataPackMeta."""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.valuation.datapack_schema import EmbodimentProfileSummary


def embodiment_profile_from_summary(
    summary: Dict[str, Any],
    artifact_paths: Optional[Dict[str, str]] = None,
    cost_breakdown: Optional[Dict[str, Any]] = None,
    value_attribution: Optional[Dict[str, Any]] = None,
) -> EmbodimentProfileSummary:
    """Build EmbodimentProfileSummary from runner outputs."""
    artifact_paths = artifact_paths or {}
    cost_summary = cost_breakdown.get("episode") if isinstance(cost_breakdown, dict) else None
    value_summary = value_attribution.get("totals") if isinstance(value_attribution, dict) else None

    return EmbodimentProfileSummary(
        w_embodiment=float(summary.get("w_embodiment", 1.0)),
        embodiment_quality_score=float(summary.get("embodiment_quality_score", summary.get("w_embodiment", 1.0))),
        trust_override_candidate=bool(summary.get("trust_override_candidate", False)),
        physically_impossible_contacts=int(summary.get("physically_impossible_contacts", 0)),
        contact_coverage_pct=float(summary.get("contact_coverage_pct", 0.0)),
        semantic_confidence_mean=float(summary.get("semantic_confidence_mean", 0.0)),
        drift_score=float(summary.get("drift_score", 0.0)),
        embodiment_profile_npz=artifact_paths.get("embodiment_profile_path"),
        affordance_graph_npz=artifact_paths.get("affordance_graph_path"),
        skill_segments_npz=artifact_paths.get("skill_segments_path"),
        cost_breakdown_json=artifact_paths.get("cost_breakdown_path"),
        value_attribution_json=artifact_paths.get("value_attribution_path"),
        drift_report_json=artifact_paths.get("drift_report_path"),
        calibration_targets_json=artifact_paths.get("calibration_targets_path"),
        summary_jsonl=artifact_paths.get("summary_jsonl"),
        cost_summary=cost_summary,
        value_summary=value_summary,
        diagnostics=summary.get("diagnostics"),
    )


__all__ = ["embodiment_profile_from_summary"]

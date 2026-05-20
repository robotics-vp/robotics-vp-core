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
    diagnostics = dict(summary.get("diagnostics") or {})
    phase3_refs = {
        key: artifact_paths.get(key) or summary.get(key)
        for key in (
            "embodiment_actuation_state_path",
            "embodiment_actuation_receipts_path",
            "embodiment_actuation_consumers_path",
            "embodiment_phase34_training_rows_path",
            "embodiment_phase34_training_manifest_path",
            "embodiment_neural_architecture_manifest_path",
            "embodiment_morphology_profile_path",
            "embodiment_morphology_receipts_path",
        )
        if artifact_paths.get(key) or summary.get(key)
    }
    if phase3_refs:
        diagnostics["embodiment_actuation_artifact_refs"] = phase3_refs
    if isinstance(summary.get("embodiment_actuation"), dict):
        diagnostics["embodiment_actuation"] = summary["embodiment_actuation"]

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
        embodiment_actuation_state_json=phase3_refs.get("embodiment_actuation_state_path"),
        embodiment_actuation_receipts_json=phase3_refs.get("embodiment_actuation_receipts_path"),
        embodiment_actuation_consumers_json=phase3_refs.get("embodiment_actuation_consumers_path"),
        embodiment_phase34_training_rows_jsonl=phase3_refs.get("embodiment_phase34_training_rows_path"),
        embodiment_phase34_training_manifest_json=phase3_refs.get("embodiment_phase34_training_manifest_path"),
        embodiment_neural_architecture_manifest_json=phase3_refs.get(
            "embodiment_neural_architecture_manifest_path"
        ),
        embodiment_morphology_profile_json=phase3_refs.get("embodiment_morphology_profile_path"),
        embodiment_morphology_receipts_json=phase3_refs.get("embodiment_morphology_receipts_path"),
        cost_summary=cost_summary,
        value_summary=value_summary,
        diagnostics=diagnostics,
    )


__all__ = ["embodiment_profile_from_summary"]

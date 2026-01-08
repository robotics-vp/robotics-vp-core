from typing import List, Dict, Any
import math

from src.valuation.datapack_schema import (
    ObjectiveProfile,
    AttributionProfile,
    GuidanceProfile,
    EmbodimentProfileSummary,
    DataPackMeta,
)
from src.orchestrator.semantic_metrics import SemanticMetrics


def _is_nan(x):
    try:
        return math.isnan(x)
    except Exception:
        return False


def validate_objective_profile(op: ObjectiveProfile) -> List[str]:
    warnings = []
    if not op.objective_vector or len(op.objective_vector) < 3:
        warnings.append("ObjectiveProfile.objective_vector missing or too short")
    if op.wage_human is None:
        warnings.append("ObjectiveProfile.wage_human missing")
    if op.energy_price_kWh is None:
        warnings.append("ObjectiveProfile.energy_price_kWh missing")
    return warnings


def validate_attribution_profile(ap: AttributionProfile) -> List[str]:
    warnings = []
    fields = [
        ("delta_mpl_model", ap.delta_mpl_model),
        ("delta_mpl_data", ap.delta_mpl_data),
        ("delta_mpl_energy", ap.delta_mpl_energy),
        ("rebate_pct", ap.rebate_pct),
        ("attributable_spread_capture", ap.attributable_spread_capture),
        ("data_premium", ap.data_premium),
    ]
    for name, val in fields:
        if val is None or _is_nan(val):
            warnings.append(f"AttributionProfile.{name} missing or NaN")
    return warnings


def validate_guidance_profile(gp: GuidanceProfile) -> List[str]:
    warnings = []
    if gp.semantic_tags is None:
        warnings.append("GuidanceProfile.semantic_tags missing")
    if gp.quality_label is None:
        warnings.append("GuidanceProfile.quality_label missing")
    return warnings


def validate_semantic_metrics(sm: SemanticMetrics) -> List[str]:
    warnings = []
    if sm.task_cluster_purity < 0 or sm.task_cluster_purity > 1:
        warnings.append("SemanticMetrics.task_cluster_purity out of [0,1]")
    if sm.concept_drift_score < 0:
        warnings.append("SemanticMetrics.concept_drift_score negative")
    if sm.label_conflict_rate < 0:
        warnings.append("SemanticMetrics.label_conflict_rate negative")
    return warnings


def validate_embodiment_profile(ep: EmbodimentProfileSummary) -> List[str]:
    warnings = []
    if ep.w_embodiment < 0 or ep.w_embodiment > 1:
        warnings.append("EmbodimentProfileSummary.w_embodiment out of [0,1]")
    if ep.embodiment_quality_score < 0 or ep.embodiment_quality_score > 1:
        warnings.append("EmbodimentProfileSummary.embodiment_quality_score out of [0,1]")
    if ep.contact_coverage_pct < 0 or ep.contact_coverage_pct > 1:
        warnings.append("EmbodimentProfileSummary.contact_coverage_pct out of [0,1]")
    if ep.semantic_confidence_mean < 0 or ep.semantic_confidence_mean > 1:
        warnings.append("EmbodimentProfileSummary.semantic_confidence_mean out of [0,1]")
    if ep.drift_score < 0 or ep.drift_score > 1:
        warnings.append("EmbodimentProfileSummary.drift_score out of [0,1]")
    if ep.physically_impossible_contacts < 0:
        warnings.append("EmbodimentProfileSummary.physically_impossible_contacts negative")
    for field in (
        "embodiment_profile_npz",
        "affordance_graph_npz",
        "skill_segments_npz",
        "cost_breakdown_json",
        "value_attribution_json",
        "drift_report_json",
        "calibration_targets_json",
        "summary_jsonl",
    ):
        value = getattr(ep, field)
        if value is not None and not isinstance(value, str):
            warnings.append(f"EmbodimentProfileSummary.{field} must be a string path")
    return warnings


def validate_datapack_meta(dp: DataPackMeta) -> List[str]:
    warnings = []
    if dp.objective_profile:
        warnings.extend(validate_objective_profile(dp.objective_profile))
    if dp.guidance_profile:
        warnings.extend(validate_guidance_profile(dp.guidance_profile))
    if dp.embodiment_profile:
        warnings.extend(validate_embodiment_profile(dp.embodiment_profile))
    if dp.epiplexity_summary:
        warnings.extend(validate_epiplexity_summary(dp.epiplexity_summary))
    warnings.extend(validate_attribution_profile(dp.attribution))
    return warnings


def validate_epiplexity_summary(summary: Dict[str, Any]) -> List[str]:
    warnings = []
    if not isinstance(summary, dict):
        return ["epiplexity_summary must be a dict"]
    for repr_id, budgets in summary.items():
        if repr_id == "_default":
            continue
        if not isinstance(budgets, dict):
            continue
        for budget_id, stats in budgets.items():
            if not isinstance(stats, dict):
                continue
            mean = stats.get("mean", {})
            for key in ("S_T_proxy", "H_T_proxy", "epi_per_flop", "delta_epi_vs_baseline"):
                val = mean.get(key)
                if val is None:
                    continue
                if _is_nan(val):
                    warnings.append(f"epiplexity_summary.{repr_id}.{budget_id}.{key} is NaN")
    return warnings

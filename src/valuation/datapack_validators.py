from typing import List, Dict, Any
import math

from src.valuation.datapack_schema import ObjectiveProfile, AttributionProfile, GuidanceProfile, DataPackMeta
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


def validate_datapack_meta(dp: DataPackMeta) -> List[str]:
    warnings = []
    if dp.objective_profile:
        warnings.extend(validate_objective_profile(dp.objective_profile))
    if dp.guidance_profile:
        warnings.extend(validate_guidance_profile(dp.guidance_profile))
    warnings.extend(validate_attribution_profile(dp.attribution))
    return warnings

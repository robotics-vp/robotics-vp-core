import uuid
from typing import List, Tuple

import numpy as np

from src.valuation.datapack_repo import DataPackRepo
from src.valuation.datapack_schema import DataPackMeta
from src.valuation.guidance_profile import GuidanceProfile
from src.orchestrator.context import OrchestratorContext
from src.orchestrator.toolspecs import OrchestrationStep


def score_datapack_economic_value(dp: DataPackMeta) -> float:
    """
    Heuristic scalar score using objective weights to combine deltas.
    """
    op_vec = dp.objective_profile.objective_vector if dp.objective_profile else [1.0, 1.0, 1.0, 0.0, 0.0]
    w_mpl, w_error, w_energy, w_safety, _ = op_vec[:5]
    delta_mpl = dp.attribution.delta_mpl
    delta_error = dp.attribution.delta_error
    delta_ep = dp.attribution.delta_ep
    delta_j = dp.attribution.delta_J
    score = (
        w_mpl * delta_mpl
        - w_error * delta_error
        - w_energy * (-delta_ep)  # higher EP is good, so subtract negative delta_ep
        + 0.1 * delta_j
    )
    return float(score)


def classify_good_bad(dp: DataPackMeta, threshold: float = 0.0) -> Tuple[bool, str]:
    score = score_datapack_economic_value(dp)
    if score >= threshold * 2:
        return True, "high_value"
    if score >= threshold:
        return True, "medium"
    if score <= -abs(threshold):
        return False, "risky"
    return False, "low_value"


def build_guidance_profile_for_datapack(
    dp: DataPackMeta,
    ctx: OrchestratorContext,
    plan_id: str,
    step_index: int,
) -> GuidanceProfile:
    # Objective
    obj_vec = dp.objective_profile.objective_vector if dp.objective_profile else ctx.objective_vector
    # Drivers
    main_driver = "throughput_gain"
    if abs(dp.attribution.delta_error) > abs(dp.attribution.delta_mpl):
        main_driver = "error_reduction" if dp.attribution.delta_error < 0 else "error_increase"
    if abs(dp.attribution.delta_ep) > abs(dp.attribution.delta_mpl):
        main_driver = "energy_efficiency"

    semantic_tags = []
    semantic_tags.extend(dp.energy_driver_tags or [])
    semantic_tags.extend(dp.semantic_tags or [])
    if dp.vla_plan and "skill_sequence" in dp.vla_plan:
        semantic_tags.extend([str(s) for s in dp.vla_plan["skill_sequence"]])

    is_good, quality_label = classify_good_bad(dp, threshold=0.0)

    return GuidanceProfile(
        is_good=is_good,
        quality_label=quality_label,
        env_name=dp.env_type,
        engine_type=dp.condition.engine_type,
        task_type=dp.condition.task_name,
        customer_segment=getattr(dp.condition, "customer_segment", ctx.customer_segment),
        objective_vector=obj_vec,
        main_driver=main_driver,
        delta_mpl=dp.attribution.delta_mpl,
        delta_error=dp.attribution.delta_error,
        delta_energy_Wh=dp.energy.total_Wh,
        delta_J=dp.attribution.delta_J,
        semantic_tags=semantic_tags,
        orchestrator_plan_id=plan_id,
        orchestrator_step_index=step_index,
    )


def annotate_datapacks_with_guidance(
    repo: DataPackRepo,
    ctx: OrchestratorContext,
    plan_id: str,
    steps: List[OrchestrationStep],
    max_packs: int = 100,
) -> List[DataPackMeta]:
    dps = repo.query(
        task_name=ctx.env_name,
        engine_type=ctx.engine_type,
        condition_filters={"customer_segment": ctx.customer_segment} if ctx.customer_segment else None,
        limit=max_packs * 2,
    ) or []

    if not dps:
        return []

    scored = [(dp, score_datapack_economic_value(dp)) for dp in dps]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: max_packs // 2]
    bottom = scored[-max_packs // 2 :] if len(scored) > 1 else []

    updated = []
    for idx, (dp, _) in enumerate(top + bottom):
        gp = build_guidance_profile_for_datapack(dp, ctx, plan_id, step_index=idx if steps else 0)
        dp.guidance_profile = gp
        updated.append(dp)
    return updated

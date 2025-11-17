from typing import Any, Dict, Optional, List
from dataclasses import asdict
import uuid

from src.envs.dishwashing_env import EpisodeInfoSummary
from src.config.econ_params import EconParams
from src.valuation.energy_tags import infer_energy_driver_tags
from src.valuation.datapack_schema import (
    DATAPACK_SCHEMA_VERSION,
    DataPackMeta,
    EnergyProfile,
    ConditionProfile,
    AttributionProfile,
    SimaAnnotation,
    ObjectiveProfile,
)


def build_datapack_from_episode(
    episode_info: EpisodeInfoSummary,
    econ_params: EconParams,
    condition_profile: Dict[str, Any],
    agent_profile: Dict[str, Any],
    brick_id: Optional[str] = None,
    env_type: str = "dishwashing",
    extra_tags: Optional[list] = None,
    semantic_energy_drivers: Optional[list] = None,
    econ_semantic_tags: Optional[List[str]] = None,
    semantic_quality: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a standardized datapack record for valuation across Phase B/C tasks.

    Includes:
    - episode metrics (EpisodeInfoSummary)
    - econ params (for normalization/reference)
    - semantic/condition and agent profiles
    - brick/datapack identifiers

    Returns legacy dict format for backwards compatibility.
    Use build_datapack_meta_from_episode() for DataPackMeta object.
    """
    metrics = asdict(episode_info)
    tags = (condition_profile.get("tags", []) if condition_profile else []) + (extra_tags or [])
    datapack = {
        "schema_version": DATAPACK_SCHEMA_VERSION,
        "env_type": env_type,
        "brick_id": brick_id,
        "episode_metrics": metrics,
        "episode_id": metrics.get("episode_id"),
        "media_refs": metrics.get("media_refs", {}),
        "econ_params": {
            "price_per_unit": econ_params.price_per_unit,
            "damage_cost": econ_params.damage_cost,
            "energy_Wh_per_attempt": econ_params.energy_Wh_per_attempt,
            "time_step_s": econ_params.time_step_s,
            "max_steps": econ_params.max_steps,
            "preset": econ_params.preset,
        },
        "condition_profile": condition_profile,
        "agent_profile": agent_profile,
        "tags": tags,
        "econ_semantic_tags": econ_semantic_tags,
        "semantic_quality": max(0.0, min(1.0, semantic_quality)) if semantic_quality is not None else None,
        "attribution": {
            "delta_mpl": metrics.get("mpl_episode"),
            "delta_error": metrics.get("error_rate_episode"),
            "delta_ep": metrics.get("ep_episode"),
            "novelty": condition_profile.get("novelty", None) if condition_profile else None,
            "trust": metrics.get("wage_parity", None),  # placeholder if trust not logged
            "econ_weight": condition_profile.get("econ_weight", None) if condition_profile else None,
        },
        "energy": {
            "total_Wh": metrics.get("energy_Wh", 0.0),
            "Wh_per_unit": metrics.get("energy_Wh_per_unit", 0.0),
            "Wh_per_hour": metrics.get("energy_Wh_per_hour", 0.0),
            "limb_energy_Wh": metrics.get("limb_energy_Wh", {}),
            "skill_energy_Wh": metrics.get("skill_energy_Wh", {}),
            "energy_per_limb": metrics.get("energy_per_limb", {}),
            "energy_per_skill": metrics.get("energy_per_skill", {}),
            "energy_per_joint": metrics.get("energy_per_joint", {}),
            "energy_per_effector": metrics.get("energy_per_effector", {}),
            "coordination_metrics": metrics.get("coordination_metrics", {}),
        },
        "semantic_energy_drivers": semantic_energy_drivers or infer_energy_driver_tags(episode_info, econ_params),
    }
    return datapack


def build_datapack_meta_from_episode(
    episode_info: EpisodeInfoSummary,
    econ_params: EconParams,
    condition_profile: Dict[str, Any],
    agent_profile: Dict[str, Any],
    brick_id: Optional[str] = None,
    env_type: str = "dishwashing",
    extra_tags: Optional[list] = None,
    semantic_energy_drivers: Optional[list] = None,
    skill_trace: Optional[List[Dict[str, Any]]] = None,
    sima_annotation: Optional[SimaAnnotation] = None,
    vla_plan: Optional[Dict[str, Any]] = None,
    baseline_mpl: Optional[float] = None,
    baseline_error: Optional[float] = None,
    baseline_ep: Optional[float] = None,
    objective_profile: Optional[ObjectiveProfile] = None,
    econ_semantic_tags: Optional[List[str]] = None,
    semantic_quality: Optional[float] = None,
) -> DataPackMeta:
    """
    Build a DataPackMeta object from episode info (unified schema).

    This is the canonical way to create datapacks for use with DataPackRepo.
    Includes all 2.0-energy fields plus two-bucket taxonomy support.

    Args:
        episode_info: EpisodeInfoSummary from environment
        econ_params: Economic parameters
        condition_profile: Environment condition dict
        agent_profile: Agent/policy info dict
        brick_id: Optional datapack ID
        env_type: Environment type string
        extra_tags: Additional semantic tags
        semantic_energy_drivers: Energy driver tags (inferred if None)
        skill_trace: Optional HRL skill trace
        sima_annotation: Optional SIMA language annotation
        vla_plan: Optional VLA plan annotation
        baseline_mpl: Baseline MPL for ΔJ computation
        baseline_error: Baseline error rate for ΔJ computation
        baseline_ep: Baseline energy productivity for ΔJ computation
        objective_profile: Optional ObjectiveProfile for DL econ hyperparameters
        econ_semantic_tags: Optional econ/semantic advisory tags
        semantic_quality: Optional advisory quality score in [0, 1]

    Returns:
        DataPackMeta object ready for DataPackRepo
    """
    metrics = asdict(episode_info)
    tags = (condition_profile.get("tags", []) if condition_profile else []) + (extra_tags or [])

    # Compute deltas vs baseline
    ep_mpl = metrics.get("mpl_episode", 0.0)
    ep_error = metrics.get("error_rate_episode", 0.0)
    ep_ep = metrics.get("ep_episode", 1.0)

    if baseline_mpl is not None:
        delta_mpl = ep_mpl - baseline_mpl
    else:
        delta_mpl = ep_mpl

    if baseline_error is not None:
        delta_error = ep_error - baseline_error
    else:
        delta_error = ep_error

    if baseline_ep is not None:
        delta_ep = ep_ep - baseline_ep
    else:
        delta_ep = ep_ep

    # Compute ΔJ (simple weighted sum, can be customized)
    delta_j = delta_mpl - delta_error * 10 + delta_ep * 0.1

    # Determine bucket
    bucket = "positive" if delta_j >= 0 else "negative"

    # Build condition profile
    cond = ConditionProfile(
        task_name=env_type,
        engine_type=condition_profile.get("engine_type", "pybullet"),
        world_id=condition_profile.get("world_id", f"pyb_{env_type}_v1"),
        econ_preset=econ_params.preset,
        price_per_unit=econ_params.price_per_unit,
        vase_break_cost=econ_params.damage_cost,
        energy_price_kWh=getattr(econ_params, "energy_price_kWh", 0.12),
        tags=condition_profile,
    )

    # Build attribution profile
    attr = AttributionProfile(
        delta_mpl=delta_mpl,
        delta_error=delta_error,
        delta_ep=delta_ep,
        delta_J=delta_j,
        trust_score=metrics.get("wage_parity", 0.0) or 0.0,
        w_econ=condition_profile.get("econ_weight", 0.0) or 0.0,
    )

    # Build energy profile
    energy = EnergyProfile.from_episode_metrics(metrics)

    # Infer energy driver tags if not provided
    if semantic_energy_drivers is None:
        energy_tags = infer_energy_driver_tags(episode_info, econ_params)
    else:
        energy_tags = semantic_energy_drivers

    # Generate pack_id
    pack_id = brick_id or str(uuid.uuid4())

    return DataPackMeta(
        schema_version=DATAPACK_SCHEMA_VERSION,
        pack_id=pack_id,
        task_name=env_type,
        env_type=env_type,
        brick_id=brick_id,
        bucket=bucket,
        semantic_tags=tags,
        econ_semantic_tags=econ_semantic_tags,
        semantic_quality=max(0.0, min(1.0, semantic_quality)) if semantic_quality is not None else None,
        energy_driver_tags=energy_tags,
        condition=cond,
        attribution=attr,
        energy=energy,
        agent_profile=agent_profile,
        skill_trace=skill_trace or [],
        episode_metrics=metrics,
        sima_annotation=sima_annotation,
        vla_plan=vla_plan,
        objective_profile=objective_profile,
        counterfactual_plan=None,
        counterfactual_source=None,
    )


def wrap_legacy_datapack(legacy_dict: Dict[str, Any]) -> DataPackMeta:
    """
    Wrap a legacy 2.0-energy dict into a DataPackMeta object.

    For converting existing datapacks to the unified schema.
    """
    return DataPackMeta.from_legacy_energy_dict(legacy_dict)

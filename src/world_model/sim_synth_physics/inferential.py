"""Inferential learnability helpers for sim/synth/physics WM planning."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from src.economics.inferential_contract import (
    InferentialLearnabilityContract,
    build_inferential_learnability_contract,
)

from .common import clip01, mapping, safe_float


def _summary(payload: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    context = mapping(payload)
    summary = context.get("inferential_learnability_summary")
    if isinstance(summary, Mapping):
        return mapping(summary)
    return {}


def _context_signal_bias(semantic_context: Optional[Mapping[str, Any]], economic_context: Optional[Mapping[str, Any]]) -> tuple[float, float]:
    semantic_summary = _summary(semantic_context)
    economic_summary = _summary(economic_context)
    mean_signal = max(
        safe_float(semantic_summary.get("mean_signal_yield_score"), 0.0),
        safe_float(economic_summary.get("mean_signal_yield_score"), 0.0),
    )
    mean_weight = max(
        safe_float(semantic_summary.get("mean_inferential_replay_weight"), 0.0),
        safe_float(economic_summary.get("mean_inferential_replay_weight"), 0.0),
    )
    return clip01(mean_signal), clip01(mean_weight)


def benchmark_provenance_quality(benchmark_signals: Optional[Mapping[str, Any]]) -> float:
    payload = mapping(benchmark_signals)
    quality = 0.2
    quality += 0.35 * float(bool(payload.get("semantic_grounding_non_heuristic", False)))
    quality += 0.2 * float(bool(payload.get("scene_tracks_backend_real", False)))
    quality += 0.15 * float(bool(payload.get("vision_backbone_real", False)))
    quality += 0.1 * float(bool(payload.get("benchmark_eligible", False)))
    return clip01(quality)


def build_simulation_job_inferential_contract(
    *,
    job_id: str,
    coverage_gap_score: float,
    economic_priority: float,
    trust_priority: float,
    readiness: float,
    ranking_policy: str,
    wm_validation_pressure: float,
    benchmark_signals: Optional[Mapping[str, Any]] = None,
    semantic_context: Optional[Mapping[str, Any]] = None,
    economic_context: Optional[Mapping[str, Any]] = None,
) -> InferentialLearnabilityContract:
    context_signal_bias, context_weight_bias = _context_signal_bias(
        semantic_context,
        economic_context,
    )
    provenance_quality = benchmark_provenance_quality(benchmark_signals)
    frontier_gain = clip01(coverage_gap_score)
    epiplexity_delta = clip01(
        0.45 * frontier_gain
        + 0.25 * max(0.0, 1.0 - trust_priority)
        + 0.15 * wm_validation_pressure
        + 0.15 * context_signal_bias
    )
    epiplexity_confidence = clip01(
        0.25
        + 0.35 * provenance_quality
        + 0.2 * readiness
        + 0.2 * context_weight_bias
    )
    transfer_score = clip01(
        0.55 * economic_priority
        + 0.25 * readiness
        + 0.2 * context_signal_bias
    )
    data_quality = clip01(0.6 * readiness + 0.4 * frontier_gain)
    trust_score = clip01(
        0.2
        + 0.4 * provenance_quality
        + 0.2 * trust_priority
        + 0.2 * readiness
    )
    return build_inferential_learnability_contract(
        subject_id=job_id,
        subject_kind="sim_synth_job",
        datapack_id=job_id,
        frontier_gain=frontier_gain,
        epiplexity_delta=epiplexity_delta,
        epiplexity_confidence=epiplexity_confidence,
        transfer_score=transfer_score,
        data_quality=data_quality,
        provenance_quality=provenance_quality,
        trust_score=trust_score,
        overlay_joined=True,
        benchmark_eligible=bool(mapping(benchmark_signals).get("benchmark_eligible", False)),
        semantic_grounding_non_heuristic=bool(
            mapping(benchmark_signals).get("semantic_grounding_non_heuristic", False)
        ),
        promotion_trace_complete=ranking_policy != "heuristic_only",
        metadata={
            "ranking_policy": ranking_policy,
            "wm_validation_pressure": float(wm_validation_pressure),
            "context_signal_bias": float(context_signal_bias),
            "context_weight_bias": float(context_weight_bias),
        },
    )


def agenda_score_with_inferential_prior(
    *,
    base_ranking_score: float,
    contract: InferentialLearnabilityContract,
) -> float:
    signal_score = clip01(contract.signal_yield.get("score", 0.0))
    replay_weight = clip01(contract.inferential_replay_weight)
    return (
        safe_float(base_ranking_score, 0.0)
        + 0.2 * signal_score
        + 0.1 * replay_weight
    )


def build_branch_plan_inferential_contract(
    *,
    plan_id: str,
    job_id: str,
    expected_yield_score: float,
    job_contract: Optional[InferentialLearnabilityContract],
    benchmark_signals: Optional[Mapping[str, Any]] = None,
    semantic_context: Optional[Mapping[str, Any]] = None,
    economic_context: Optional[Mapping[str, Any]] = None,
) -> InferentialLearnabilityContract:
    context_signal_bias, context_weight_bias = _context_signal_bias(
        semantic_context,
        economic_context,
    )
    provenance_quality = benchmark_provenance_quality(benchmark_signals)
    upstream_signal = (
        clip01(job_contract.signal_yield.get("score", 0.0))
        if job_contract is not None
        else 0.0
    )
    upstream_weight = (
        clip01(job_contract.inferential_replay_weight)
        if job_contract is not None
        else 0.0
    )
    benchmark_payload = mapping(benchmark_signals)
    return build_inferential_learnability_contract(
        subject_id=plan_id,
        subject_kind="synthetic_branch_plan",
        datapack_id=job_id,
        frontier_gain=clip01(expected_yield_score),
        epiplexity_delta=clip01(
            0.45 * expected_yield_score
            + 0.25 * upstream_signal
            + 0.15 * context_signal_bias
            + 0.15 * upstream_weight
        ),
        epiplexity_confidence=clip01(
            0.3
            + 0.3 * provenance_quality
            + 0.2 * upstream_weight
            + 0.2 * context_weight_bias
        ),
        transfer_score=clip01(
            0.45 * expected_yield_score
            + 0.35 * upstream_signal
            + 0.2 * context_signal_bias
        ),
        data_quality=clip01(0.55 * expected_yield_score + 0.45 * upstream_signal),
        provenance_quality=provenance_quality,
        trust_score=clip01(
            0.25
            + 0.35 * provenance_quality
            + 0.2 * upstream_weight
            + 0.2 * expected_yield_score
        ),
        overlay_joined=True,
        benchmark_eligible=bool(benchmark_payload.get("benchmark_eligible", False)),
        semantic_grounding_non_heuristic=bool(
            benchmark_payload.get("semantic_grounding_non_heuristic", False)
        ),
        promotion_trace_complete=bool(
            job_contract is not None and job_contract.promotion_trace_complete
        ),
        metadata={
            "source_job_id": job_id,
            "upstream_job_learnability_class": (
                job_contract.learnability_class if job_contract is not None else "missing"
            ),
            "context_signal_bias": float(context_signal_bias),
            "context_weight_bias": float(context_weight_bias),
        },
    )


def adjusted_branch_yield_score(
    *,
    base_expected_yield_score: float,
    contract: InferentialLearnabilityContract,
) -> float:
    signal_score = clip01(contract.signal_yield.get("score", 0.0))
    replay_weight = clip01(contract.inferential_replay_weight)
    return clip01(
        0.7 * base_expected_yield_score
        + 0.2 * signal_score
        + 0.1 * replay_weight
    )


def diffusion_priority_with_inferential_prior(
    *,
    coverage_gap_score: float,
    economic_priority: float,
    trust_priority: float,
    branch_yield_score: float,
    branch_admissible: bool,
    contract: Optional[InferentialLearnabilityContract],
) -> float:
    signal_score = (
        clip01(contract.signal_yield.get("score", 0.0))
        if contract is not None
        else 0.0
    )
    replay_weight = (
        clip01(contract.inferential_replay_weight)
        if contract is not None
        else 0.0
    )
    admissible_bonus = 0.08 if branch_admissible else -0.05
    return clip01(
        0.3 * clip01(coverage_gap_score)
        + 0.2 * clip01(economic_priority)
        + 0.1 * clip01(trust_priority)
        + 0.2 * clip01(branch_yield_score)
        + 0.12 * signal_score
        + 0.08 * replay_weight
        + admissible_bonus
    )

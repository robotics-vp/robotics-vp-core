"""WM-owned gen2sim admission helpers and corpus assessment utilities."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.economics.inferential_contract import (
    coerce_inferential_learnability_contract,
    summarize_inferential_learnability_contracts,
)
from src.evidence.gen2sim_validity import assess_gen2sim_validity

from .common import clip01, mapping, safe_float, stable_id
from .receipts import Gen2SimAdmissionReceipt
from .state import Gen2SimAdmissionState, SyntheticBranchPlan


def compile_gen2sim_admission_state(
    branch_plans: Sequence[SyntheticBranchPlan],
    jobs: Sequence[Any],
    *,
    benchmark_signals: Mapping[str, Any],
    robot_asset_contract: Any | None = None,
) -> Gen2SimAdmissionState:
    benchmark_payload = mapping(benchmark_signals)
    benchmark_gate_ready = bool(
        benchmark_payload.get("ready", False)
        or benchmark_payload.get("benchmark_eligible", False)
    )
    semantic_grounding_non_heuristic = bool(
        benchmark_payload.get("semantic_grounding_non_heuristic", False)
    )
    admissible_rows: list[tuple[str, float]] = []
    blocked_rows: list[tuple[str, float]] = []
    contracts = []
    job_by_id = {str(getattr(job, "job_id", "")): job for job in jobs}
    for plan in branch_plans:
        job = job_by_id.get(str(plan.source_job_id))
        preconditions = dict(plan.admission_preconditions)
        plan_contract = coerce_inferential_learnability_contract(
            plan.inferential_learnability_contract
        )
        if plan_contract is not None:
            contracts.append(plan_contract)
        admissible = True
        if bool(preconditions.get("requires_benchmark_ready", False)) and not benchmark_gate_ready:
            admissible = False
        if (
            bool(preconditions.get("requires_non_heuristic_grounding", False))
            and not semantic_grounding_non_heuristic
        ):
            admissible = False
        if job is not None and safe_float(preconditions.get("min_readiness", 0.0), 0.0) > float(
            getattr(job, "readiness", 0.0)
        ):
            admissible = False
        if (
            plan_contract is not None
            and safe_float(preconditions.get("min_inferential_replay_weight", 0.0), 0.0)
            > float(plan_contract.inferential_replay_weight)
        ):
            admissible = False
        admission_score = clip01(
            0.55 * float(plan.expected_yield_score)
            + 0.2
            * clip01(
                plan_contract.signal_yield.get("score", 0.0)
                if plan_contract is not None
                else 0.0
            )
            + 0.15
            * clip01(
                plan_contract.inferential_replay_weight if plan_contract is not None else 0.0
            )
            + 0.1 * float(benchmark_gate_ready)
        )
        if admissible:
            admissible_rows.append((plan.plan_id, admission_score))
        else:
            blocked_rows.append((plan.plan_id, admission_score))
    admissible_rows.sort(key=lambda item: item[1], reverse=True)
    blocked_rows.sort(key=lambda item: item[1], reverse=True)
    admissible_branch_ids = [plan_id for plan_id, _score in admissible_rows]
    blocked_branch_ids = [plan_id for plan_id, _score in blocked_rows]
    rationale = (
        f"{len(admissible_branch_ids)} branch plans admissible, "
        f"{len(blocked_branch_ids)} blocked by benchmark, grounding, or inferential preconditions."
    )
    inferential_summary = summarize_inferential_learnability_contracts(contracts)
    asset_metadata = {}
    if robot_asset_contract is not None:
        asset_metadata = {
            "robot_asset_contract_id": str(getattr(robot_asset_contract, "contract_id", "") or ""),
            "asset_profile": str(getattr(robot_asset_contract, "asset_profile", "") or ""),
            "target_hardware_class": str(
                getattr(robot_asset_contract, "target_hardware_class", "") or ""
            ),
            "asset_readiness_score": safe_float(
                getattr(robot_asset_contract, "metadata", {}).get("asset_readiness_score", 0.0),
                0.0,
            ),
            "missing_assets": list(getattr(robot_asset_contract, "missing_assets", []) or []),
        }
    return Gen2SimAdmissionState(
        admission_id=stable_id(
            "gen2sim_admission",
            {
                "admissible": admissible_branch_ids,
                "blocked": blocked_branch_ids,
                "benchmark_gate_ready": benchmark_gate_ready,
            },
        ),
        benchmark_gate_ready=benchmark_gate_ready,
        admissible_branch_ids=admissible_branch_ids,
        blocked_branch_ids=blocked_branch_ids,
        selection_policy="receipt_gated_with_inferential_contracts",
        rationale=rationale,
        inferential_learnability_summary=inferential_summary,
        metadata={
            "benchmark_signals": benchmark_payload,
            "semantic_grounding_non_heuristic": semantic_grounding_non_heuristic,
            "admission_scores": {
                **{plan_id: score for plan_id, score in admissible_rows},
                **{plan_id: score for plan_id, score in blocked_rows},
            },
            "robot_asset_contract": asset_metadata,
        },
    )


def _helper_stage_counts(branch_plans: Sequence[SyntheticBranchPlan]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for plan in branch_plans:
        helper_status = mapping(getattr(plan, "metadata", {}).get("branch_helper_status"))
        counter[str(helper_status.get("promotion_stage", "") or "heuristic_fallback")] += 1
    return dict(counter)


def _render_provider_counts(branch_plans: Sequence[SyntheticBranchPlan]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for plan in branch_plans:
        provider = getattr(plan, "render_provider", None)
        provider_kind = "" if provider is None else str(getattr(provider, "provider_kind", "") or "")
        counter[provider_kind or "unknown"] += 1
    return dict(counter)


def build_gen2sim_admission_receipt(
    admission_state: Gen2SimAdmissionState,
    branch_plans: Sequence[SyntheticBranchPlan],
    jobs: Sequence[Any],
) -> Gen2SimAdmissionReceipt:
    """Build a typed receipt for the compiled gen2sim admission decision."""

    job_by_id = {str(getattr(job, "job_id", "") or ""): job for job in jobs}
    branch_by_id = {str(plan.plan_id): plan for plan in branch_plans}
    admissible_branch_ids = list(admission_state.admissible_branch_ids or [])
    blocked_branch_ids = list(admission_state.blocked_branch_ids or [])
    admissible_source_job_ids = sorted(
        {
            str(branch_by_id[branch_id].source_job_id)
            for branch_id in admissible_branch_ids
            if branch_id in branch_by_id
        }
    )
    blocked_source_job_ids = sorted(
        {
            str(branch_by_id[branch_id].source_job_id)
            for branch_id in blocked_branch_ids
            if branch_id in branch_by_id
        }
    )
    benchmark_signals = mapping(admission_state.metadata.get("benchmark_signals"))
    benchmark_provenance = {
        "semantic_grounding_non_heuristic": bool(
            benchmark_signals.get("semantic_grounding_non_heuristic", False)
        ),
        "scene_tracks_backend_real": bool(benchmark_signals.get("scene_tracks_backend_real", False)),
        "vision_backbone_real": bool(benchmark_signals.get("vision_backbone_real", False)),
        "teacher_runtime_real": bool(benchmark_signals.get("teacher_runtime_real", False)),
    }
    metadata = {
        "branch_count": len(branch_plans),
        "job_count": len([job for job in jobs if str(getattr(job, "job_id", "") or "")]),
        "admissible_branch_count": len(admissible_branch_ids),
        "blocked_branch_count": len(blocked_branch_ids),
        "admissible_source_job_ids": admissible_source_job_ids,
        "blocked_source_job_ids": blocked_source_job_ids,
        "helper_promotion_stage_counts": _helper_stage_counts(branch_plans),
        "render_provider_kind_counts": _render_provider_counts(branch_plans),
        "benchmark_provenance": benchmark_provenance,
        "synthetic_evidence_counts": {
            "synthetic_branch_plan_count": len(branch_plans),
            "admissible_synthetic_branch_count": len(admissible_branch_ids),
            "blocked_synthetic_branch_count": len(blocked_branch_ids),
        },
        "job_readiness": {
            job_id: safe_float(getattr(job_by_id[job_id], "readiness", 0.0), 0.0)
            for job_id in sorted(job_by_id)
        },
        "admission_scores": mapping(admission_state.metadata.get("admission_scores")),
        "robot_asset_contract": mapping(admission_state.metadata.get("robot_asset_contract")),
        "state_metadata": mapping(admission_state.metadata),
    }
    receipt_payload = {
        "admission_id": admission_state.admission_id,
        "admissible_branch_ids": admissible_branch_ids,
        "blocked_branch_ids": blocked_branch_ids,
        "selection_policy": admission_state.selection_policy,
    }
    return Gen2SimAdmissionReceipt(
        receipt_id=stable_id("gen2sim_admission_receipt", receipt_payload),
        admission_id=admission_state.admission_id,
        benchmark_gate_ready=bool(admission_state.benchmark_gate_ready),
        admissible_branch_ids=admissible_branch_ids,
        blocked_branch_ids=blocked_branch_ids,
        selection_policy=str(admission_state.selection_policy or "receipt_gated"),
        rationale=str(admission_state.rationale or ""),
        inferential_learnability_summary=mapping(
            admission_state.inferential_learnability_summary
        ),
        metadata=metadata,
    )


def assess_local_branch_corpus_gen2sim(
    branches: Sequence[Mapping[str, Any]],
    *,
    corpus_name: str,
    source_runtime_metadata: Optional[Mapping[str, Any]] = None,
    scene_tracks_backend: str = "unavailable",
    teacher_runtime_backend_selected: str = "unavailable",
    vision_backbone_selected: str = "unavailable",
    semantic_grounding_mode: str = "heuristic_fallback",
    semantic_memory_grounded: bool = False,
    gap_labels_path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validity_scores: list[float] = []
    admission_scores: list[float] = []
    stage_counts: dict[str, int] = {}
    benchmark_ready_count = 0
    execution_ready_count = 0

    source_metadata = dict(source_runtime_metadata or {})
    for idx, branch in enumerate(branches):
        branch_gap_labels = mapping(branch.get("gap_labels"))
        assessment = assess_gen2sim_validity(
            subject_id=f"{corpus_name}_branch_{idx:04d}",
            subject_kind="synthetic_branch",
            metadata={
                **source_metadata,
                "scene_tracks_backend": scene_tracks_backend,
                "teacher_runtime_backend_selected": teacher_runtime_backend_selected,
                "vision_backbone_selected": vision_backbone_selected,
                "semantic_grounding_mode": semantic_grounding_mode,
                "semantic_memory_grounded": semantic_memory_grounded,
                "source_runtime_metadata": source_metadata or None,
                "branch_gap_labels": gap_labels_path,
                "trust_score": branch.get("trust_score"),
                "std_ratio": branch.get("std_ratio"),
                "branch_value": branch.get("branch_value", branch.get("trust_score", 0.0)),
                "coverage_gap_contribution": branch_gap_labels.get(
                    "coverage_gap_contribution",
                    0.0,
                ),
                "economic_priority": branch_gap_labels.get("economic_priority", 0.0),
            },
            trust_score=branch.get("trust_score"),
            std_ratio=branch.get("std_ratio"),
            branch_value=branch.get("branch_value", branch.get("trust_score", 0.0)),
            gap_labels=branch_gap_labels,
        )
        validity_scores.append(float(assessment.validity_score))
        admission_scores.append(float(assessment.admission_score))
        benchmark_ready_count += int(assessment.benchmark_gate_ready)
        execution_ready_count += int(assessment.execution_ready)
        stage = str(assessment.promotion_stage or "heuristic_fallback")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        row = {"branch_idx": idx, **assessment.to_dict()}
        rows.append(row)

    summary = {
        "count": len(rows),
        "avg_validity_score": float(np.mean(validity_scores)) if validity_scores else 0.0,
        "avg_admission_score": float(np.mean(admission_scores)) if admission_scores else 0.0,
        "promotion_stage_counts": stage_counts,
        "benchmark_gate_ready_count": benchmark_ready_count,
        "execution_ready_count": execution_ready_count,
    }
    return rows, summary


__all__ = [
    "assess_local_branch_corpus_gen2sim",
    "build_gen2sim_admission_receipt",
    "compile_gen2sim_admission_state",
]

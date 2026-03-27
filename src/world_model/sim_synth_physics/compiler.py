"""Compiler for the canonical sim/synth/physics world state."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

from src.orchestrator.gap_agenda_ranking import rank_gaps_for_agenda

from .adapters import build_semantic_input_context
from .agenda import SimulationAgenda, SimulationJobSpec
from .common import clip01, mapping, safe_float, stable_id
from .promotion import HelperMode, infer_backend_payload, infer_branch_payload, resolve_helper
from .state import (
    DiffusionConditioningState,
    Gen2SimAdmissionState,
    PhysicsContextState,
    SimSynthPhysicsWorldState,
    SyntheticBranchPlan,
)

EXPECTED_RECEIPTS = [
    "simulation_outcome_receipt",
    "physics_calibration_receipt",
    "replay_artifact_refs",
    "training_feedback_refs",
]


def _task_family(src_node: Any, source_id: str) -> str:
    if src_node is not None and str(getattr(src_node, "node_type", "")) == "task":
        return str(getattr(src_node, "label", source_id))
    return "unknown"


def _env_backend(tgt_node: Any, default_backend: str) -> str:
    if tgt_node is not None and str(getattr(tgt_node, "node_type", "")) == "backend":
        return str(getattr(tgt_node, "label", default_backend))
    return str(default_backend)


def _risk_and_object_families(tgt_node: Any) -> tuple[str, str]:
    if tgt_node is None:
        return "", ""
    node_type = str(getattr(tgt_node, "node_type", ""))
    label = str(getattr(tgt_node, "label", ""))
    if node_type == "risk_family":
        return label, ""
    if node_type == "object_family":
        return "", label
    return "", ""


def _data_collection_intent(economic_priority: float, trust_priority: float) -> str:
    if economic_priority > 0.7:
        return "exploit"
    if trust_priority < 0.3:
        return "validate"
    return "explore"


def _job_rationale(ranked_gap: Any) -> str:
    gap = ranked_gap.gap
    wm_validation_pressure = safe_float(getattr(gap, "metadata", {}).get("wm_validation_pressure", 0.0), 0.0)
    rationale = f"Missing {gap.edge_type}: {gap.source_id} -> {gap.target_id}"
    if wm_validation_pressure > 0.0:
        rationale += f" | wm_validation_pressure={wm_validation_pressure:.2f}"
    return rationale


def _compile_jobs(
    coverage_graph: Any,
    *,
    economic_weight: float,
    trust_weight: float,
    readiness_weight: float,
    limit: int,
    default_backend: str,
    default_objective: str,
    gap_ranker: Any,
    gap_ranker_mode: Literal["disabled", "auto", "required"],
) -> list[SimulationJobSpec]:
    ranked_gaps = rank_gaps_for_agenda(
        coverage_graph=coverage_graph,
        economic_weight=economic_weight,
        trust_weight=trust_weight,
        readiness_weight=readiness_weight,
        limit=limit,
        gap_ranker=gap_ranker,
        gap_ranker_mode=gap_ranker_mode,
    )

    jobs: list[SimulationJobSpec] = []
    for rank_idx, ranked_gap in enumerate(ranked_gaps):
        gap = ranked_gap.gap
        if bool(getattr(gap, "metadata", {}).get("governance_blocked", False)):
            continue
        src_node = coverage_graph.node_by_id(gap.source_id)
        tgt_node = coverage_graph.node_by_id(gap.target_id)
        src_label = str(getattr(src_node, "label", gap.source_id))
        tgt_label = str(getattr(tgt_node, "label", gap.target_id))
        risk_family, object_family = _risk_and_object_families(tgt_node)
        economic_priority = clip01(getattr(gap, "economic_priority", 0.0))
        trust_priority = clip01(getattr(gap, "trust_priority", 0.0))
        readiness = clip01(getattr(gap, "promotion_readiness", 0.0))
        intent = _data_collection_intent(economic_priority, trust_priority)
        coverage_targets = {
            "source_id": gap.source_id,
            "target_id": gap.target_id,
            "edge_type": gap.edge_type,
            "wm_validation_pressure": safe_float(
                getattr(gap, "metadata", {}).get("wm_validation_pressure", 0.0),
                0.0,
            ),
        }
        job_payload = {
            "rank": rank_idx + 1,
            "source_id": gap.source_id,
            "target_id": gap.target_id,
            "edge_type": gap.edge_type,
            "objective_preset": default_objective,
            "env_backend": _env_backend(tgt_node, default_backend),
            "ranking_policy": ranked_gap.ranking_policy,
        }
        job = SimulationJobSpec(
            job_id=stable_id("sim_job", job_payload),
            rank=rank_idx + 1,
            task_family=_task_family(src_node, gap.source_id),
            env_backend=_env_backend(tgt_node, default_backend),
            skill_edge=f"{src_label} -> {tgt_label}",
            risk_family=risk_family,
            object_family=object_family,
            objective_preset=str(default_objective),
            data_collection_intent=intent,
            coverage_gap_score=safe_float(ranked_gap.ranking_score, 0.0),
            economic_priority=economic_priority,
            trust_priority=trust_priority,
            readiness=readiness,
            ranking_policy=str(ranked_gap.ranking_policy),
            rationale=_job_rationale(ranked_gap),
            coverage_targets=coverage_targets,
            expected_receipts=list(EXPECTED_RECEIPTS),
            metadata={
                "agenda_helper_status": mapping(ranked_gap.helper_status),
                "score_trace": {
                    "heuristic_score": safe_float(ranked_gap.heuristic_score, 0.0),
                    "heuristic_score_norm": clip01(ranked_gap.heuristic_score_norm),
                    "learned_score": safe_float(ranked_gap.learned_score, 0.0),
                    "learned_score_norm": clip01(ranked_gap.learned_score_norm),
                    "ranking_score": safe_float(ranked_gap.ranking_score, 0.0),
                },
            },
        )
        jobs.append(job)
    return jobs


def _agenda_ranking_policy(jobs: list[SimulationJobSpec]) -> str:
    policies = {job.ranking_policy for job in jobs}
    if not policies:
        return "heuristic_only"
    if len(policies) == 1:
        return next(iter(policies))
    return "mixed"


def _heuristic_fidelity_tier(jobs: list[SimulationJobSpec]) -> str:
    if any(job.risk_family for job in jobs):
        return "high_fidelity"
    if any(job.data_collection_intent == "validate" for job in jobs):
        return "high_fidelity"
    if any(job.data_collection_intent == "exploit" for job in jobs):
        return "branch_balanced"
    return "fast_scan"


def _heuristic_domain_randomization_regime(jobs: list[SimulationJobSpec]) -> str:
    if any(job.data_collection_intent == "validate" for job in jobs):
        return "calibration_focus"
    if any(job.data_collection_intent == "explore" for job in jobs):
        return "coverage_exploration"
    return "steady_state"


def _timestep_for_fidelity(fidelity_tier: str) -> float:
    if fidelity_tier == "high_fidelity":
        return 4.0
    if fidelity_tier == "branch_balanced":
        return 8.0
    return 16.0


def _compile_physics_context(
    jobs: list[SimulationJobSpec],
    *,
    default_backend: str,
    benchmark_signals: Mapping[str, Any],
    backend_selector: Any,
    backend_selector_mode: HelperMode,
) -> PhysicsContextState:
    heuristic_backend = jobs[0].env_backend if jobs else str(default_backend)
    heuristic_fidelity = _heuristic_fidelity_tier(jobs)
    heuristic_randomization = _heuristic_domain_randomization_regime(jobs)
    helper, helper_status = resolve_helper(
        backend_selector,
        mode=backend_selector_mode,
        name="sim_synth_physics backend selector",
    )
    helper_payload = infer_backend_payload(
        helper,
        context={
            "jobs": [job.to_dict() for job in jobs],
            "benchmark_signals": mapping(benchmark_signals),
            "heuristic_backend": heuristic_backend,
            "heuristic_fidelity_tier": heuristic_fidelity,
            "heuristic_domain_randomization_regime": heuristic_randomization,
        },
    )

    selection_policy = "heuristic_only"
    backend = heuristic_backend
    fidelity_tier = heuristic_fidelity
    domain_randomization_regime = heuristic_randomization
    if helper_payload:
        selection_policy = "heuristic_plus_learned_backend_selector"
        if str(helper_status.get("promotion_stage")) == "promoted":
            backend = str(
                helper_payload.get("preferred_backend")
                or helper_payload.get("backend")
                or heuristic_backend
            )
            fidelity_tier = str(helper_payload.get("fidelity_tier") or heuristic_fidelity)
            domain_randomization_regime = str(
                helper_payload.get("domain_randomization_regime") or heuristic_randomization
            )

    calibration_profile = "benchmark_ready" if bool(benchmark_signals.get("ready", False)) else "shadow_replay"
    context_payload = {
        "backend": backend,
        "fidelity_tier": fidelity_tier,
        "domain_randomization_regime": domain_randomization_regime,
        "selection_policy": selection_policy,
        "job_ids": [job.job_id for job in jobs],
    }
    return PhysicsContextState(
        context_id=stable_id("physics_context", context_payload),
        backend=backend,
        fidelity_tier=fidelity_tier,
        timestep_ms=_timestep_for_fidelity(fidelity_tier),
        domain_randomization_regime=domain_randomization_regime,
        calibration_profile=calibration_profile,
        selection_policy=selection_policy,
        metadata={
            "heuristic_backend": heuristic_backend,
            "heuristic_fidelity_tier": heuristic_fidelity,
            "heuristic_domain_randomization_regime": heuristic_randomization,
            "backend_helper_status": helper_status,
            "backend_helper_trace": helper_payload,
            "benchmark_signals": mapping(benchmark_signals),
        },
    )


def _heuristic_generation_mode(job: SimulationJobSpec, physics_context: PhysicsContextState) -> str:
    if job.data_collection_intent == "validate":
        return "physics_probe"
    if physics_context.fidelity_tier == "high_fidelity":
        return "geometry_guarded_rollout"
    if job.data_collection_intent == "exploit":
        return "targeted_synth_rollout"
    return "coverage_branch"


def _heuristic_yield_score(job: SimulationJobSpec) -> float:
    return clip01(
        (0.4 * clip01(job.coverage_gap_score))
        + (0.35 * job.economic_priority)
        + (0.15 * (1.0 - job.trust_priority))
        + (0.10 * job.readiness)
    )


def _compile_branch_plans(
    jobs: list[SimulationJobSpec],
    *,
    physics_context: PhysicsContextState,
    benchmark_signals: Mapping[str, Any],
    branch_planner: Any,
    branch_planner_mode: HelperMode,
) -> list[SyntheticBranchPlan]:
    helper, helper_status = resolve_helper(
        branch_planner,
        mode=branch_planner_mode,
        name="sim_synth_physics branch planner",
    )
    plans: list[SyntheticBranchPlan] = []
    for job in jobs:
        heuristic_generation_mode = _heuristic_generation_mode(job, physics_context)
        heuristic_yield_score = _heuristic_yield_score(job)
        helper_payload = infer_branch_payload(
            helper,
            job=job.to_dict(),
            context={
                "job": job.to_dict(),
                "physics_context": physics_context.to_dict(),
                "benchmark_signals": mapping(benchmark_signals),
            },
        )
        selection_policy = "heuristic_only"
        generation_mode = heuristic_generation_mode
        expected_yield_score = heuristic_yield_score
        if helper_payload:
            selection_policy = "heuristic_plus_learned_branch_planner"
            if str(helper_status.get("promotion_stage")) == "promoted":
                generation_mode = str(helper_payload.get("generation_mode") or heuristic_generation_mode)
                expected_yield_score = clip01(
                    helper_payload.get("expected_yield_score", heuristic_yield_score)
                )
        admission_preconditions = {
            "requires_non_heuristic_grounding": bool(
                job.data_collection_intent == "validate" and bool(job.risk_family)
            ),
            "requires_benchmark_ready": bool(job.readiness >= 0.8 and job.economic_priority >= 0.8),
            "min_readiness": 0.0,
        }
        plan_payload = {
            "job_id": job.job_id,
            "branch_family": f"{job.task_family}:{job.data_collection_intent}",
            "generation_mode": generation_mode,
            "render_backend": physics_context.backend,
        }
        plans.append(
            SyntheticBranchPlan(
                plan_id=stable_id("branch_plan", plan_payload),
                source_job_id=job.job_id,
                branch_family=f"{job.task_family}:{job.data_collection_intent}",
                generation_mode=generation_mode,
                render_backend=physics_context.backend,
                gap_target_refs=[mapping(job.coverage_targets)],
                admission_preconditions=admission_preconditions,
                expected_yield_score=expected_yield_score,
                selection_policy=selection_policy,
                metadata={
                    "agenda_rank": job.rank,
                    "source_ranking_policy": job.ranking_policy,
                    "heuristic_generation_mode": heuristic_generation_mode,
                    "heuristic_expected_yield_score": heuristic_yield_score,
                    "branch_helper_status": helper_status,
                    "branch_helper_trace": helper_payload,
                },
            )
        )
    return plans


def _compile_diffusion_conditioning(
    jobs: list[SimulationJobSpec],
    branch_plans: list[SyntheticBranchPlan],
    *,
    physics_context: PhysicsContextState,
) -> Optional[DiffusionConditioningState]:
    if not jobs:
        return None
    top_job = jobs[0]
    semantic_tags = sorted(
        {
            tag
            for tag in [
                top_job.task_family,
                top_job.risk_family,
                top_job.object_family,
                top_job.data_collection_intent,
            ]
            if tag not in ("", "unknown")
        }
    )
    governed_modes = [plan.generation_mode for plan in branch_plans[: min(len(branch_plans), 3)]]
    prompt_hints = {
        "coverage_targets": [job.coverage_targets for job in jobs[: min(len(jobs), 3)]],
        "data_collection_intents": [job.data_collection_intent for job in jobs[: min(len(jobs), 3)]],
        "objective_preset": top_job.objective_preset,
    }
    routing_context = {
        "routing_source": "sim_synth_physics_world_state",
        "agenda_ranking_policy": top_job.ranking_policy,
        "physics_selection_policy": physics_context.selection_policy,
        "branch_selection_policies": [plan.selection_policy for plan in branch_plans[: min(len(branch_plans), 3)]],
    }
    conditioning_payload = {
        "env_backend": physics_context.backend,
        "objective_preset": top_job.objective_preset,
        "branch_job_ids": [plan.source_job_id for plan in branch_plans],
        "governed_modes": governed_modes,
    }
    return DiffusionConditioningState(
        conditioning_id=stable_id("diffusion_conditioning", conditioning_payload),
        objective_preset=top_job.objective_preset,
        env_backend=physics_context.backend,
        semantic_tags=semantic_tags,
        branch_job_ids=[plan.source_job_id for plan in branch_plans],
        governed_modes=governed_modes,
        render_budget=min(len(branch_plans), 3),
        prompt_hints=prompt_hints,
        routing_context=routing_context,
        metadata={
            "source_job_ids": [job.job_id for job in jobs],
        },
    )


def _compile_gen2sim_admission(
    branch_plans: list[SyntheticBranchPlan],
    jobs: list[SimulationJobSpec],
    *,
    benchmark_signals: Mapping[str, Any],
) -> Gen2SimAdmissionState:
    benchmark_gate_ready = bool(
        benchmark_signals.get("ready", False)
        or benchmark_signals.get("benchmark_eligible", False)
    )
    semantic_grounding_non_heuristic = bool(
        benchmark_signals.get("semantic_grounding_non_heuristic", False)
    )
    admissible_branch_ids: list[str] = []
    blocked_branch_ids: list[str] = []
    job_by_id = {job.job_id: job for job in jobs}
    for plan in branch_plans:
        job = job_by_id.get(plan.source_job_id)
        preconditions = dict(plan.admission_preconditions)
        admissible = True
        if bool(preconditions.get("requires_benchmark_ready", False)) and not benchmark_gate_ready:
            admissible = False
        if (
            bool(preconditions.get("requires_non_heuristic_grounding", False))
            and not semantic_grounding_non_heuristic
        ):
            admissible = False
        if job is not None and safe_float(preconditions.get("min_readiness", 0.0), 0.0) > job.readiness:
            admissible = False
        if admissible:
            admissible_branch_ids.append(plan.plan_id)
        else:
            blocked_branch_ids.append(plan.plan_id)
    rationale = (
        f"{len(admissible_branch_ids)} branch plans admissible, "
        f"{len(blocked_branch_ids)} blocked by benchmark or grounding preconditions."
    )
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
        selection_policy="receipt_gated_with_helper_traces",
        rationale=rationale,
        metadata={
            "benchmark_signals": mapping(benchmark_signals),
            "semantic_grounding_non_heuristic": semantic_grounding_non_heuristic,
        },
    )


def compile_sim_synth_physics_world_state(
    coverage_graph: Any,
    *,
    semantic_context: Optional[Mapping[str, Any]] = None,
    economic_context: Optional[Mapping[str, Any]] = None,
    embodiment_context: Optional[Mapping[str, Any]] = None,
    benchmark_signals: Optional[Mapping[str, Any]] = None,
    economic_weight: float = 1.0,
    trust_weight: float = 1.0,
    readiness_weight: float = 1.0,
    limit: int = 10,
    default_backend: str = "pybullet",
    default_objective: str = "balanced",
    gap_ranker: Any = None,
    gap_ranker_mode: Literal["disabled", "auto", "required"] = "auto",
    backend_selector: Any = None,
    backend_selector_mode: HelperMode = "auto",
    branch_planner: Any = None,
    branch_planner_mode: HelperMode = "auto",
) -> SimSynthPhysicsWorldState:
    """Compile the canonical sim/synth/physics WM state for one planning window."""

    benchmark_payload = mapping(benchmark_signals)
    jobs = _compile_jobs(
        coverage_graph,
        economic_weight=economic_weight,
        trust_weight=trust_weight,
        readiness_weight=readiness_weight,
        limit=limit,
        default_backend=default_backend,
        default_objective=default_objective,
        gap_ranker=gap_ranker,
        gap_ranker_mode=gap_ranker_mode,
    )
    coverage_window_ref = stable_id(
        "coverage_window",
        {"job_ids": [job.job_id for job in jobs], "ranking_policy": _agenda_ranking_policy(jobs)},
    )
    agenda = SimulationAgenda(
        agenda_id=stable_id(
            "simulation_agenda",
            {"job_ids": [job.job_id for job in jobs], "coverage_window_ref": coverage_window_ref},
        ),
        coverage_window_ref=coverage_window_ref,
        jobs=jobs,
        ranking_policy=_agenda_ranking_policy(jobs),
        metadata={
            "job_count": len(jobs),
            "top_ranked_job_id": jobs[0].job_id if jobs else None,
        },
    )
    physics_context = _compile_physics_context(
        jobs,
        default_backend=default_backend,
        benchmark_signals=benchmark_payload,
        backend_selector=backend_selector,
        backend_selector_mode=backend_selector_mode,
    )
    branch_plans = _compile_branch_plans(
        jobs,
        physics_context=physics_context,
        benchmark_signals=benchmark_payload,
        branch_planner=branch_planner,
        branch_planner_mode=branch_planner_mode,
    )
    diffusion_conditioning = _compile_diffusion_conditioning(
        jobs,
        branch_plans,
        physics_context=physics_context,
    )
    gen2sim_admission = _compile_gen2sim_admission(
        branch_plans,
        jobs,
        benchmark_signals=benchmark_payload,
    )
    input_context = {
        "semantic": build_semantic_input_context(
            coverage_graph=coverage_graph,
            semantic_context=semantic_context,
        ),
        "economic": mapping(economic_context),
        "embodiment": mapping(embodiment_context),
        "benchmark": benchmark_payload,
    }
    artifact_refs = {
        "coverage_window_ref": coverage_window_ref,
        "branch_plan_ids": [plan.plan_id for plan in branch_plans],
    }
    state_payload = {
        "agenda_id": agenda.agenda_id,
        "physics_context_id": physics_context.context_id,
        "branch_plan_ids": [plan.plan_id for plan in branch_plans],
        "admission_id": gen2sim_admission.admission_id,
    }
    return SimSynthPhysicsWorldState(
        state_id=stable_id("sim_synth_physics", state_payload),
        simulation_agenda=agenda,
        physics_context=physics_context,
        synthetic_branch_plans=branch_plans,
        gen2sim_admission=gen2sim_admission,
        diffusion_conditioning=diffusion_conditioning,
        input_context=input_context,
        artifact_refs=artifact_refs,
        metadata={
            "helper_modes": {
                "gap_ranker_mode": gap_ranker_mode,
                "backend_selector_mode": backend_selector_mode,
                "branch_planner_mode": branch_planner_mode,
            },
            "world_model_scope": "sim_synth_physics",
            "job_count": len(jobs),
            "blocked_branch_count": len(gen2sim_admission.blocked_branch_ids),
        },
    )

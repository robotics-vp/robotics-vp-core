"""Compiler for the canonical sim/synth/physics world state."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, Mapping, Optional

from src.economics.inferential_contract import summarize_inferential_learnability_contracts

from .adapters import (
    build_economic_input_context,
    build_embodiment_input_context,
    build_semantic_input_context,
)
from .agenda import SimulationAgenda, SimulationJobSpec
from .asset_contracts import compile_robot_asset_contract
from .backend_adapters import describe_backend_adapter
from .backend_bindings import compile_backend_execution_binding
from .backend_router import build_physics_execution_contract
from .backend_selector_runtime import resolve_backend_selector_helper
from .common import clip01, mapping, safe_float, stable_id
from .inferential import (
    agenda_score_with_inferential_prior,
    build_simulation_job_inferential_contract,
)
from .gen2sim_admission import compile_gen2sim_admission_state
from .promotion import HelperMode, infer_backend_payload
from .randomization import compile_physics_adaptation_policy
from .runtime_bridge import compile_backend_runtime_bridge
from .state import (
    DiffusionConditioningState,
    Gen2SimAdmissionState,
    PhysicsAdaptationPolicyState,
    PhysicsContextState,
    SimSynthPhysicsWorldState,
    SyntheticBranchPlan,
)
from .synthetic_branches import compile_synthetic_branch_plans

EXPECTED_RECEIPTS = [
    "simulation_outcome_receipt",
    "physics_calibration_receipt",
    "replay_artifact_refs",
    "training_feedback_refs",
]

COMPILER_OWNED_RECEIPTS = [
    "physics_execution_contract_v1",
    "physics_adaptation_receipt_v1",
    "gen2sim_admission_receipt_v1",
    "backend_execution_binding_receipt_v1",
    "robot_asset_contract_receipt_v1",
    "backend_runtime_bridge_receipt_v1",
]

RUNTIME_OWNED_RECEIPTS = [
    "backend_runtime_work_order_receipt_v1",
    "backend_runtime_execution_receipt_v1",
    "backend_runtime_adapter_receipt_v1",
    "backend_runtime_launch_receipt_v1",
    "backend_runtime_outcome_receipt_v1",
    "backend_shadow_execution_receipt_v1",
    "physics_calibration_receipt_v1",
    "sim_synth_training_feedback_v1",
]

PER_BRANCH_RECEIPTS = [
    "render_provider_receipt_v1",
    "simulation_outcome_receipt_v1",
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
    semantic_context: Optional[Mapping[str, Any]],
    economic_context: Optional[Mapping[str, Any]],
    benchmark_signals: Mapping[str, Any],
    economic_weight: float,
    trust_weight: float,
    readiness_weight: float,
    limit: int,
    default_backend: str,
    default_objective: str,
    gap_ranker: Any,
    gap_ranker_mode: Literal["disabled", "auto", "required"],
) -> list[SimulationJobSpec]:
    from src.orchestrator.gap_agenda_ranking import rank_gaps_for_agenda

    ranked_gaps = rank_gaps_for_agenda(
        coverage_graph=coverage_graph,
        economic_weight=economic_weight,
        trust_weight=trust_weight,
        readiness_weight=readiness_weight,
        limit=limit,
        gap_ranker=gap_ranker,
        gap_ranker_mode=gap_ranker_mode,
    )

    provisional_jobs: list[tuple[SimulationJobSpec, float, int]] = []
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
            "wm_validation_pressure": safe_float(getattr(gap, "metadata", {}).get("wm_validation_pressure", 0.0), 0.0),
        }
        inferential_contract = build_simulation_job_inferential_contract(
            job_id=stable_id(
                "sim_job",
                {
                    "rank": rank_idx + 1,
                    "source_id": gap.source_id,
                    "target_id": gap.target_id,
                    "edge_type": gap.edge_type,
                    "objective_preset": default_objective,
                    "env_backend": _env_backend(tgt_node, default_backend),
                    "ranking_policy": ranked_gap.ranking_policy,
                },
            ),
            coverage_gap_score=safe_float(ranked_gap.ranking_score, 0.0),
            economic_priority=economic_priority,
            trust_priority=trust_priority,
            readiness=readiness,
            ranking_policy=str(ranked_gap.ranking_policy),
            wm_validation_pressure=safe_float(coverage_targets.get("wm_validation_pressure", 0.0), 0.0),
            benchmark_signals=benchmark_signals,
            semantic_context=semantic_context,
            economic_context=economic_context,
        )
        inferential_signal_score = clip01(inferential_contract.signal_yield.get("score", 0.0))
        inferential_replay_weight = clip01(inferential_contract.inferential_replay_weight)
        combined_agenda_score = agenda_score_with_inferential_prior(
            base_ranking_score=safe_float(ranked_gap.ranking_score, 0.0),
            contract=inferential_contract,
        )
        job = SimulationJobSpec(
            job_id=inferential_contract.subject_id,
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
            inferential_learnability_contract=inferential_contract.to_dict(),
            metadata={
                "agenda_helper_status": mapping(ranked_gap.helper_status),
                "score_trace": {
                    "heuristic_score": safe_float(ranked_gap.heuristic_score, 0.0),
                    "heuristic_score_norm": clip01(ranked_gap.heuristic_score_norm),
                    "learned_score": safe_float(ranked_gap.learned_score, 0.0),
                    "learned_score_norm": clip01(ranked_gap.learned_score_norm),
                    "ranking_score": safe_float(ranked_gap.ranking_score, 0.0),
                    "inferential_signal_yield_score": inferential_signal_score,
                    "inferential_replay_weight": inferential_replay_weight,
                    "combined_agenda_score": combined_agenda_score,
                },
                "agenda_inferential_policy": "ranking_plus_inferential_contract",
            },
        )
        provisional_jobs.append((job, combined_agenda_score, rank_idx))
    provisional_jobs.sort(
        key=lambda item: (
            item[1],
            item[0].economic_priority,
            item[0].readiness,
            -item[2],
        ),
        reverse=True,
    )
    jobs: list[SimulationJobSpec] = []
    for resolved_rank, (job, combined_score, _original_idx) in enumerate(provisional_jobs, start=1):
        metadata = dict(job.metadata)
        score_trace = dict(metadata.get("score_trace", {}) or {})
        score_trace["combined_agenda_rank"] = resolved_rank
        score_trace["combined_agenda_score"] = combined_score
        metadata["score_trace"] = score_trace
        jobs.append(replace(job, rank=resolved_rank, metadata=metadata))
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
    helper, helper_status = resolve_backend_selector_helper(
        backend_selector,
        mode=backend_selector_mode,
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
    adapter_descriptor = describe_backend_adapter(backend)
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
            "backend_adapter": adapter_descriptor.to_dict(),
            "benchmark_signals": mapping(benchmark_signals),
        },
    )


def _compile_diffusion_conditioning(
    jobs: list[SimulationJobSpec],
    branch_plans: list[SyntheticBranchPlan],
    *,
    physics_context: PhysicsContextState,
    gen2sim_admission: Gen2SimAdmissionState,
) -> Optional[DiffusionConditioningState]:
    if not jobs:
        return None
    admissible_branch_ids = list(gen2sim_admission.admissible_branch_ids or [])
    blocked_branch_ids = list(gen2sim_admission.blocked_branch_ids or [])
    branch_order = {
        plan_id: index for index, plan_id in enumerate(admissible_branch_ids + blocked_branch_ids)
    }
    ordered_branch_plans = sorted(
        branch_plans,
        key=lambda plan: (
            branch_order.get(plan.plan_id, len(branch_order)),
            -safe_float(plan.expected_yield_score, 0.0),
        ),
    )
    top_branch = ordered_branch_plans[0] if ordered_branch_plans else branch_plans[0]
    top_job = next(
        (job for job in jobs if job.job_id == top_branch.source_job_id),
        jobs[0],
    )
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
    governed_modes = [
        plan.generation_mode for plan in ordered_branch_plans[: min(len(ordered_branch_plans), 3)]
    ]
    prompt_hints = {
        "coverage_targets": [job.coverage_targets for job in jobs[: min(len(jobs), 3)]],
        "data_collection_intents": [job.data_collection_intent for job in jobs[: min(len(jobs), 3)]],
        "objective_preset": top_job.objective_preset,
        "admissible_branch_ids": admissible_branch_ids[:3],
    }
    routing_context = {
        "routing_source": "sim_synth_physics_world_state",
        "agenda_ranking_policy": top_job.ranking_policy,
        "physics_selection_policy": physics_context.selection_policy,
        "branch_selection_policies": [
            plan.selection_policy
            for plan in ordered_branch_plans[: min(len(ordered_branch_plans), 3)]
        ],
        "gen2sim_selection_policy": gen2sim_admission.selection_policy,
        "gen2sim_admission_id": gen2sim_admission.admission_id,
    }
    conditioning_payload = {
        "env_backend": physics_context.backend,
        "objective_preset": top_job.objective_preset,
        "branch_job_ids": [plan.source_job_id for plan in ordered_branch_plans],
        "governed_modes": governed_modes,
    }
    return DiffusionConditioningState(
        conditioning_id=stable_id("diffusion_conditioning", conditioning_payload),
        objective_preset=top_job.objective_preset,
        env_backend=physics_context.backend,
        semantic_tags=semantic_tags,
        branch_job_ids=[plan.source_job_id for plan in ordered_branch_plans],
        admissible_branch_ids=admissible_branch_ids,
        blocked_branch_ids=blocked_branch_ids,
        governed_modes=governed_modes,
        render_budget=min(len(admissible_branch_ids), 3),
        prompt_hints=prompt_hints,
        routing_context=routing_context,
        inferential_learnability_summary=dict(
            gen2sim_admission.inferential_learnability_summary or {}
        ),
        metadata={
            "source_job_ids": [job.job_id for job in jobs],
            "ordered_branch_plan_ids": [plan.plan_id for plan in ordered_branch_plans],
        },
    )


def _compile_runtime_depth_projection(
    *,
    physics_execution_contract: Any,
    backend_execution_binding: Any,
    backend_runtime_bridge: Any,
) -> dict[str, Any]:
    binding_metadata = (
        {} if backend_execution_binding is None else mapping(backend_execution_binding.metadata)
    )
    bridge_metadata = (
        {} if backend_runtime_bridge is None else mapping(backend_runtime_bridge.metadata)
    )
    runtime_target_contract = mapping(binding_metadata.get("runtime_target_contract"))
    runtime_layout_contract = mapping(binding_metadata.get("runtime_layout_contract"))
    policy_contract = mapping(binding_metadata.get("policy_contract"))
    deployment_contract = mapping(binding_metadata.get("deployment_contract"))
    upstream_runtime_pack = mapping(binding_metadata.get("upstream_runtime_pack"))
    ready_modes = [
        str(mode)
        for mode in list(deployment_contract.get("ready_modes") or [])
        if str(mode)
    ]
    ready_profiles = [
        str(profile)
        for profile in list(
            bridge_metadata.get("runtime_layout_ready_profiles")
            or runtime_layout_contract.get("ready_profiles")
            or []
        )
        if str(profile)
    ]
    usable_profiles = [
        str(profile)
        for profile in list(
            bridge_metadata.get("runtime_layout_usable_profiles")
            or runtime_layout_contract.get("usable_profiles")
            or []
        )
        if str(profile)
    ]
    ready_target_ids = [
        str(target_id)
        for target_id in list(runtime_target_contract.get("ready_target_ids") or [])
        if str(target_id)
    ]
    missing_target_ids = [
        str(target_id)
        for target_id in list(runtime_target_contract.get("missing_required_target_ids") or [])
        if str(target_id)
    ]
    return {
        "route_status": str(getattr(physics_execution_contract, "route_status", "") or ""),
        "requested_backend": str(
            getattr(physics_execution_contract, "requested_backend", "") or ""
        ),
        "resolved_backend": str(
            getattr(physics_execution_contract, "resolved_backend", "") or ""
        ),
        "binding_status": (
            "" if backend_execution_binding is None else str(backend_execution_binding.binding_status)
        ),
        "bridge_status": (
            "" if backend_runtime_bridge is None else str(backend_runtime_bridge.bridge_status)
        ),
        "transport_profile": (
            ""
            if backend_runtime_bridge is None
            else str(backend_runtime_bridge.transport_profile)
        ),
        "runtime_targets_ready": bool(runtime_target_contract.get("runtime_targets_ready", False)),
        "ready_runtime_target_ids": ready_target_ids,
        "missing_runtime_target_ids": missing_target_ids,
        "runtime_layout_ready_profiles": ready_profiles,
        "runtime_layout_usable_profiles": usable_profiles,
        "policy_ready": bool(
            bridge_metadata.get("policy_ready", policy_contract.get("policy_ready", False))
        ),
        "deployment_ready_modes": ready_modes,
        "upstream_runtime_pack_status": str(upstream_runtime_pack.get("pack_status", "") or ""),
        "upstream_runtime_ready_surfaces": [
            str(surface)
            for surface in list(upstream_runtime_pack.get("ready_surfaces") or [])
            if str(surface)
        ],
        "upstream_runtime_missing_components": [
            str(component)
            for component in list(upstream_runtime_pack.get("missing_components") or [])
            if str(component)
        ],
        "ladder_surface_versions": {
            "runtime_target_contract": str(runtime_target_contract.get("version", "") or ""),
            "runtime_layout_contract": str(runtime_layout_contract.get("version", "") or ""),
            "policy_contract": str(policy_contract.get("version", "") or ""),
            "deployment_contract": str(deployment_contract.get("version", "") or ""),
            "upstream_runtime_pack": str(upstream_runtime_pack.get("version", "") or ""),
            "backend_runtime_bridge": (
                ""
                if backend_runtime_bridge is None
                else str(backend_runtime_bridge.version)
            ),
        },
    }


def _compile_receipt_inventory(
    *,
    state_id: str,
    physics_execution_contract: Any,
    backend_execution_binding: Any,
    backend_runtime_bridge: Any,
    branch_plans: list[SyntheticBranchPlan],
    gen2sim_admission: Gen2SimAdmissionState,
    diffusion_conditioning: Optional[DiffusionConditioningState],
) -> dict[str, Any]:
    runtime_depth = _compile_runtime_depth_projection(
        physics_execution_contract=physics_execution_contract,
        backend_execution_binding=backend_execution_binding,
        backend_runtime_bridge=backend_runtime_bridge,
    )
    inventory_payload = {
        "state_id": state_id,
        "physics_execution_contract_id": str(
            getattr(physics_execution_contract, "contract_id", "") or ""
        ),
        "binding_status": str(runtime_depth.get("binding_status", "") or ""),
        "bridge_status": str(runtime_depth.get("bridge_status", "") or ""),
        "resolved_backend": str(runtime_depth.get("resolved_backend", "") or ""),
    }
    return {
        "version": "sim_synth_compiled_receipt_inventory_v1",
        "inventory_id": stable_id("sim_synth_compiled_receipt_inventory", inventory_payload),
        "compiler_owned_receipts": list(COMPILER_OWNED_RECEIPTS),
        "runtime_owned_receipts": list(RUNTIME_OWNED_RECEIPTS),
        "per_branch_receipts": list(PER_BRANCH_RECEIPTS),
        "state_artifacts": {
            "physics_execution_contract_id": str(
                getattr(physics_execution_contract, "contract_id", "") or ""
            ),
            "backend_execution_binding_id": (
                "" if backend_execution_binding is None else str(backend_execution_binding.binding_id)
            ),
            "backend_runtime_bridge_id": (
                "" if backend_runtime_bridge is None else str(backend_runtime_bridge.bridge_id)
            ),
            "admission_id": str(getattr(gen2sim_admission, "admission_id", "") or ""),
            "diffusion_conditioning_id": (
                ""
                if diffusion_conditioning is None
                else str(diffusion_conditioning.conditioning_id)
            ),
        },
        "branch_plan_count": len(branch_plans),
        "admissible_branch_count": len(gen2sim_admission.admissible_branch_ids),
        "blocked_branch_count": len(gen2sim_admission.blocked_branch_ids),
        "runtime_depth_projection": runtime_depth,
    }


def compile_sim_synth_physics_world_state(
    coverage_graph: Any,
    *,
    semantic_context: Optional[Mapping[str, Any]] = None,
    perception_grounding_state: Any = None,
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
    fallback_backend: str | None = None,
) -> SimSynthPhysicsWorldState:
    """Compile the canonical sim/synth/physics WM state for one planning window."""

    benchmark_payload = mapping(benchmark_signals)
    jobs = _compile_jobs(
        coverage_graph,
        semantic_context=semantic_context,
        economic_context=economic_context,
        benchmark_signals=benchmark_payload,
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
            "inferential_learnability_summary": summarize_inferential_learnability_contracts(
                [job.inferential_learnability_contract for job in jobs]
            ),
        },
    )
    physics_context = _compile_physics_context(
        jobs,
        default_backend=default_backend,
        benchmark_signals=benchmark_payload,
        backend_selector=backend_selector,
        backend_selector_mode=backend_selector_mode,
    )
    physics_adaptation_policy: PhysicsAdaptationPolicyState = compile_physics_adaptation_policy(
        physics_context,
        adapter=describe_backend_adapter(physics_context.backend),
        benchmark_signals=benchmark_payload,
        embodiment_context=embodiment_context,
    )
    backend_execution_binding = compile_backend_execution_binding(
        physics_context,
        adaptation_policy=physics_adaptation_policy,
        embodiment_context=embodiment_context,
    )
    robot_asset_contract = compile_robot_asset_contract(
        backend_execution_binding,
        adaptation_policy=physics_adaptation_policy,
        embodiment_context=embodiment_context,
    )
    backend_runtime_bridge = compile_backend_runtime_bridge(
        physics_context,
        backend_execution_binding,
        robot_asset_contract=robot_asset_contract,
        embodiment_context=embodiment_context,
    )
    branch_plans = compile_synthetic_branch_plans(
        jobs,
        physics_context=physics_context,
        physics_adaptation_policy=physics_adaptation_policy,
        benchmark_signals=benchmark_payload,
        semantic_context=semantic_context,
        economic_context=economic_context,
        branch_planner=branch_planner,
        branch_planner_mode=branch_planner_mode,
    )
    gen2sim_admission = compile_gen2sim_admission_state(
        branch_plans,
        jobs,
        benchmark_signals=benchmark_payload,
        robot_asset_contract=robot_asset_contract,
    )
    diffusion_conditioning = _compile_diffusion_conditioning(
        jobs,
        branch_plans,
        physics_context=physics_context,
        gen2sim_admission=gen2sim_admission,
    )
    input_context = {
        "semantic": build_semantic_input_context(
            coverage_graph=coverage_graph,
            semantic_context=semantic_context,
            perception_grounding_state=perception_grounding_state,
        ),
        "economic": build_economic_input_context(economic_context),
        "embodiment": build_embodiment_input_context(embodiment_context),
        "benchmark": benchmark_payload,
    }
    artifact_refs = {
        "coverage_window_ref": coverage_window_ref,
        "physics_adaptation_policy_id": physics_adaptation_policy.policy_id,
        "backend_execution_binding_id": backend_execution_binding.binding_id,
        "robot_asset_contract_id": robot_asset_contract.contract_id,
        "branch_plan_ids": [plan.plan_id for plan in branch_plans],
        "diffusion_conditioning_id": (
            diffusion_conditioning.conditioning_id if diffusion_conditioning is not None else None
        ),
    }
    state_payload = {
        "agenda_id": agenda.agenda_id,
        "physics_context_id": physics_context.context_id,
        "physics_adaptation_policy_id": physics_adaptation_policy.policy_id,
        "robot_asset_contract_id": robot_asset_contract.contract_id,
        "branch_plan_ids": [plan.plan_id for plan in branch_plans],
        "admission_id": gen2sim_admission.admission_id,
    }
    provisional_world_state = SimSynthPhysicsWorldState(
        state_id=stable_id("sim_synth_physics", state_payload),
        simulation_agenda=agenda,
        physics_context=physics_context,
        physics_adaptation_policy=physics_adaptation_policy,
        backend_execution_binding=backend_execution_binding,
        robot_asset_contract=robot_asset_contract,
        backend_runtime_bridge=backend_runtime_bridge,
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
            "job_inferential_summary": summarize_inferential_learnability_contracts(
                [job.inferential_learnability_contract for job in jobs]
            ),
            "branch_inferential_summary": summarize_inferential_learnability_contracts(
                [plan.inferential_learnability_contract for plan in branch_plans]
            ),
        },
    )
    physics_execution_contract = build_physics_execution_contract(
        provisional_world_state,
        fallback_backend=str(fallback_backend or default_backend),
    )
    compiled_receipt_inventory = _compile_receipt_inventory(
        state_id=provisional_world_state.state_id,
        physics_execution_contract=physics_execution_contract,
        backend_execution_binding=backend_execution_binding,
        backend_runtime_bridge=backend_runtime_bridge,
        branch_plans=branch_plans,
        gen2sim_admission=gen2sim_admission,
        diffusion_conditioning=diffusion_conditioning,
    )
    final_artifact_refs = dict(provisional_world_state.artifact_refs)
    final_artifact_refs["physics_execution_contract_id"] = physics_execution_contract.contract_id
    final_artifact_refs["compiled_receipt_inventory_id"] = str(
        compiled_receipt_inventory.get("inventory_id", "") or ""
    )
    final_metadata = dict(provisional_world_state.metadata)
    final_metadata["compiled_receipt_inventory"] = compiled_receipt_inventory
    final_metadata["runtime_depth_projection"] = mapping(
        compiled_receipt_inventory.get("runtime_depth_projection")
    )
    final_metadata["phase1_compiler_closure"] = {
        "physics_execution_contract_compiled": True,
        "runtime_depth_projected_in_compiler": True,
        "fallback_backend": str(fallback_backend or default_backend),
    }
    return replace(
        provisional_world_state,
        physics_execution_contract=physics_execution_contract,
        artifact_refs=final_artifact_refs,
        metadata=final_metadata,
    )

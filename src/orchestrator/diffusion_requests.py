import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from src.constraints.constraint_set import ConstraintSet
from src.valuation.datapack_schema import DataPackMeta
from src.valuation.guidance_profile import GuidanceProfile
from src.world_model.sim_synth_physics import (
    SimSynthPhysicsRuntime,
    SimSynthPhysicsRuntimeConfig,
    SimSynthPhysicsWorldState,
    compile_gap_driven_diffusion_plans,
)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _tag_mode_priority(mode: str, semantic_tags: List[str]) -> float:
    tag_set = {str(tag) for tag in semantic_tags}
    lower_mode = str(mode).lower()
    if any(tag in tag_set for tag in ("fragile", "safety", "avoid_collision")) and "fragile" in lower_mode:
        return 0.2
    if any(tag in tag_set for tag in ("energy", "energy_efficient")) and "energy" in lower_mode:
        return 0.2
    if any(tag in tag_set for tag in ("error_recovery", "recover")) and "recovery" in lower_mode:
        return 0.2
    if any(tag in tag_set for tag in ("high_speed", "objective:throughput")) and "throughput" in lower_mode:
        return 0.15
    if any(tag in tag_set for tag in ("semantic_gap", "semantic_conflict")) and "disambiguation" in lower_mode:
        return 0.15
    return 0.0


@dataclass
class DiffusionPromptSpec:
    request_id: str
    env_name: str
    engine_type: str
    task_type: str
    objective_vector: List[float]
    customer_segment: str

    skill_ids: List[int]
    semantic_tags: List[str]
    camera_pose_hint: Dict[str, float]
    difficulty_hint: str

    rationale: str
    target_economic_effect: Dict[str, float]

    source_datapack_ids: List[str]
    vla_hint: Optional[Dict[str, Any]] = None
    constraint_set_ref: Optional[Dict[str, Any]] = None

    # ── Coverage-gap fields (populated by gap-driven builder) ──────────
    topology_slice: Optional[Dict[str, Any]] = None
    meta_node_targets: Optional[List[str]] = None
    missing_skill_edges: Optional[List[Dict[str, str]]] = None
    missing_env_primitives: Optional[List[str]] = None
    risk_family_targets: Optional[List[str]] = None
    affordance_family_targets: Optional[List[str]] = None
    coverage_gap_score: float = 0.0
    economic_priority_score: float = 0.0
    trust_priority_score: float = 0.0
    routing_source: str = "semantic_prompt"
    routing_context: Optional[Dict[str, Any]] = None
    governed_hypotheses: Optional[List[Dict[str, Any]]] = None
    benchmark_signals: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_constraint_set_from_datapack(dp: DataPackMeta) -> ConstraintSet:
    metrics = dp.episode_metrics or {}
    sem = dp.vla_action_summary or {}
    map_first_summary = {}
    if isinstance(metrics.get("map_first_summary"), dict):
        map_first_summary = metrics.get("map_first_summary", {})
    if "map_first_quality_score" in metrics:
        map_first_summary.setdefault("map_first_quality_score", metrics.get("map_first_quality_score"))
    fusion_metrics = {
        "semantic_fusion_confidence_mean": metrics.get("semantic_fusion_confidence_mean", 0.0),
        "semantic_disagreement_vla_vs_map": metrics.get("semantic_disagreement_vla_vs_map", 0.0),
    }
    semantic_evidence = {
        "semantic_tags": (dp.semantic_tags or []) + (sem.get("semantic_tags", []) if isinstance(sem, dict) else []),
        "fragile": "fragile" in (dp.semantic_tags or []),
        "safety_critical": "safety" in (dp.semantic_tags or []),
        "vla_confidence": sem.get("confidence") if isinstance(sem, dict) else None,
        "source": "datapack_episode_metrics",
    }
    return ConstraintSet.from_artifacts(
        semantic_evidence=semantic_evidence,
        map_first_summary=map_first_summary,
        fusion_metrics=fusion_metrics,
    )


def _objective_preset_from_vector(objective_vector: List[float]) -> str:
    if len(objective_vector) >= 4:
        if _safe_float(objective_vector[0], 0.0) > 1.5:
            return "throughput"
        if _safe_float(objective_vector[3], 0.0) > 1.5:
            return "safety"
        if _safe_float(objective_vector[2], 0.0) > 1.5:
            return "energy_saver"
    return "balanced"


def _routing_context_from_guidance(
    dp: DataPackMeta,
    guidance: GuidanceProfile,
    constraint_set: ConstraintSet,
    semantic_tags: List[str],
) -> Dict[str, Any]:
    metrics = dp.episode_metrics or {}
    execution_preconditions = (
        dict(metrics.get("execution_preconditions", {}) or {})
        if isinstance(metrics, dict)
        else {}
    )
    benchmark_gate = (
        dict(metrics.get("benchmark_gate", {}) or {})
        if isinstance(metrics, dict)
        else {}
    )
    objective_preset = _objective_preset_from_vector(guidance.objective_vector)
    scene_tracks_backend = str(metrics.get("scene_tracks_backend", "") or "")
    semantic_grounding_mode = "non_heuristic" if scene_tracks_backend == "real" else "heuristic_fallback"
    semantic_disagreement = _safe_float(metrics.get("semantic_disagreement_vla_vs_map", 0.0), 0.0)
    confidence_mean = _safe_float(metrics.get("semantic_fusion_confidence_mean", 0.0), 0.0)
    coverage = max(
        _safe_float(execution_preconditions.get("readiness_score", 0.0), 0.0),
        _safe_float(metrics.get("map_first_quality_score", 0.0), 0.0),
    )
    return {
        "routing_source": "guidance_contract",
        "objective_preset": objective_preset,
        "benchmark_gate_ready": bool(benchmark_gate.get("ready", False)),
        "semantic_grounding_mode": semantic_grounding_mode,
        "scene_tracks_backend": scene_tracks_backend,
        "vision_backbone_selected": str(metrics.get("vision_backbone_selected", "") or ""),
        "teacher_runtime_backend_selected": str(
            metrics.get("teacher_runtime_backend_selected")
            or metrics.get("openvla_backend_selected")
            or ""
        ),
        "evidence_coverage": _clip01(coverage),
        "semantic_disagreement": _clip01(semantic_disagreement),
        "semantic_confidence": _clip01(confidence_mean),
        "coverage_gap_score": _clip01(1.0 - coverage),
        "economic_priority_score": _clip01(max(0.0, _safe_float(guidance.delta_mpl, 0.0)) / 5.0),
        "trust_priority_score": _clip01(1.0 - semantic_disagreement),
        "constraint_pressure": _clip01(
            float(len((constraint_set.to_structured_fields().get("hard_bounds") or {}))) / 6.0
        ),
        "benchmark_signals": benchmark_gate.get("signal_values", {}),
    }


def _guidance_hypotheses(
    guidance: GuidanceProfile,
    semantic_tags: List[str],
    routing_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    hypotheses: List[Dict[str, Any]] = []
    objective_preset = str(routing_context.get("objective_preset", "balanced"))
    benchmark_ready = bool(routing_context.get("benchmark_gate_ready", False))
    semantic_confidence = _safe_float(routing_context.get("semantic_confidence", 0.0), 0.0)
    semantic_disagreement = _safe_float(routing_context.get("semantic_disagreement", 0.0), 0.0)
    coverage_gap = _safe_float(routing_context.get("coverage_gap_score", 0.0), 0.0)
    driver = str(guidance.main_driver or "").lower()
    base_mode = "geometry_guarded_continuation"
    if "safety" in driver or {"fragile", "safety", "avoid_collision"} & set(semantic_tags):
        base_mode = "fragile_object_preservation"
    elif "energy" in driver or objective_preset == "energy_saver":
        base_mode = "energy_saver_retiming"
    elif "throughput" in driver or objective_preset == "throughput":
        base_mode = "throughput_push"
    elif "recovery" in driver or "error" in driver:
        base_mode = "recovery_branch"
    elif semantic_disagreement >= 0.25:
        base_mode = "semantic_disambiguation"

    base_priority = _clip01(
        0.35
        + 0.25 * _safe_float(routing_context.get("economic_priority_score", 0.0), 0.0)
        + 0.15 * _safe_float(routing_context.get("trust_priority_score", 0.0), 0.0)
        + 0.15 * coverage_gap
        + _tag_mode_priority(base_mode, semantic_tags)
    )
    base_plausibility = _clip01(
        0.35 + 0.35 * semantic_confidence + 0.15 * float(guidance.is_good) + 0.15 * float(benchmark_ready)
    )
    hypotheses.append(
        {
            "hypothesis_id": f"guidance_hyp_{uuid.uuid4().hex[:12]}",
            "mode": base_mode,
            "semantic_tags": list(semantic_tags),
            "scores": {
                "render_priority": base_priority,
                "plausibility": base_plausibility,
                "novelty": _clip01(
                    0.25
                    + 0.4 * abs(_safe_float(guidance.delta_mpl, 0.0)) / 5.0
                    + 0.2 * abs(_safe_float(guidance.delta_J, 0.0)) / 2.0
                ),
                "economic_priority": _clip01(_safe_float(routing_context.get("economic_priority_score", 0.0), 0.0)),
                "trust_priority": _clip01(_safe_float(routing_context.get("trust_priority_score", 0.0), 0.0)),
            },
            "rationale": (
                "Guidance-backed diffusion request compiled from datapack metrics, objective driver, "
                f"and semantic constraints for {base_mode}."
            ),
            "render_intent": {
                "should_render": True,
                "geometry_first": True,
                "routing_source": "guidance_contract",
            },
            "action_conditioning": {
                "speed_scale": 0.35 if base_mode == "fragile_object_preservation" else 0.55,
                "clearance_bias": 1.0 if base_mode == "fragile_object_preservation" else 0.65,
            },
            "metadata": {
                "difficulty_hint": guidance.quality_label,
                "benchmark_gate_ready": benchmark_ready,
            },
        }
    )
    if semantic_disagreement >= 0.2:
        hypotheses.append(
            {
                "hypothesis_id": f"guidance_hyp_{uuid.uuid4().hex[:12]}",
                "mode": "semantic_disambiguation",
                "semantic_tags": list(semantic_tags),
                "scores": {
                    "render_priority": _clip01(0.3 + 0.4 * semantic_disagreement + 0.2 * coverage_gap),
                    "plausibility": _clip01(0.4 + 0.3 * semantic_confidence),
                    "novelty": _clip01(0.25 + 0.35 * semantic_disagreement),
                    "economic_priority": _clip01(_safe_float(routing_context.get("economic_priority_score", 0.0), 0.0)),
                    "trust_priority": _clip01(0.55 + 0.25 * semantic_disagreement),
                },
                "rationale": "Semantic disagreement or low-confidence VLA hints require re-observation before broader variation.",
                "render_intent": {
                    "should_render": True,
                    "geometry_first": True,
                    "routing_source": "guidance_contract",
                },
                "action_conditioning": {"camera_reframe": 1.0, "speed_scale": 0.2},
                "metadata": {"difficulty_hint": guidance.quality_label},
            }
        )
    return hypotheses


def build_diffusion_prompt_from_guidance(
    dp: DataPackMeta,
    guidance: GuidanceProfile,
) -> DiffusionPromptSpec:
    skill_ids = []
    for s in dp.skill_trace or []:
        if "skill_id" in s:
            skill_ids.append(s["skill_id"])

    # Merge semantic tags from datapack, guidance, and VLA annotations
    constraint_set = _build_constraint_set_from_datapack(dp)
    semantic_tags = list(set(
        (dp.energy_driver_tags or []) +
        (guidance.semantic_tags or []) +
        (dp.semantic_tags or []) +
        constraint_set.to_prompt_tags()
    ))
    routing_context = _routing_context_from_guidance(dp, guidance, constraint_set, semantic_tags)
    governed_hypotheses = _guidance_hypotheses(guidance, semantic_tags, routing_context)

    difficulty_hint = "typical"
    if not guidance.is_good:
        difficulty_hint = "hard_neg"

    # Build enhanced rationale incorporating VLA insights
    vla_context = ""
    if dp.vla_action_summary and dp.vla_action_summary.get("has_vla", False):
        vla_tags = dp.vla_action_summary.get("semantic_tags", [])
        if "vla:grasp_confident" in vla_tags:
            vla_context = " VLA indicates confident grasp."
        elif "vla:scene_confusing" in vla_tags:
            vla_context = " VLA found scene confusing."
        elif "vla:grasp_uncertain" in vla_tags:
            vla_context = " VLA shows grasp uncertainty."
        if "vla:coordinated_motion" in vla_tags:
            vla_context += " Coordinated position+rotation motion."

    rationale = (
        f"Request data in env {guidance.env_name} ({guidance.engine_type}) to improve {guidance.main_driver} "
        f"under objective {guidance.objective_vector} with tags {semantic_tags}.{vla_context}"
    )

    target_effect = {
        "delta_mpl": guidance.delta_mpl,
        "delta_error": guidance.delta_error,
        "delta_energy_Wh": guidance.delta_energy_Wh,
        "delta_J": guidance.delta_J,
    }

    # Build VLA hint with enhanced semantic context
    vla_hint = None
    if dp.vla_action_summary and dp.vla_action_summary.get("action_7dof"):
        raw = dp.vla_action_summary["action_7dof"]
        if len(raw) >= 7:
            # Use human-readable vla_hint_text if available
            hint_text = dp.vla_action_summary.get("vla_hint_text", "")
            if hint_text:
                desc = hint_text
            else:
                desc = f"approx dx={raw[0]:.2f}, dy={raw[1]:.2f}, dz={raw[2]:.2f}, gripper={raw[6]:.2f}"

            # Determine confidence level from VLA semantic tags
            vla_sem_tags = dp.vla_action_summary.get("semantic_tags", [])
            if "vla:grasp_confident" in vla_sem_tags:
                confidence = "high"
            elif "vla:scene_confusing" in vla_sem_tags:
                confidence = "low"
            else:
                confidence = "medium"

            vla_hint = {
                "instruction": guidance.main_driver,
                "action_desc": desc,
                "semantic_tags": vla_sem_tags,
                "confidence": confidence,
            }

    return DiffusionPromptSpec(
        request_id=str(uuid.uuid4()),
        env_name=guidance.env_name,
        engine_type=guidance.engine_type,
        task_type=guidance.task_type,
        objective_vector=guidance.objective_vector,
        customer_segment=guidance.customer_segment,
        skill_ids=skill_ids,
        semantic_tags=semantic_tags,
        camera_pose_hint={},
        difficulty_hint=difficulty_hint,
        rationale=rationale,
        target_economic_effect=target_effect,
        source_datapack_ids=[dp.pack_id],
        vla_hint=vla_hint,
        constraint_set_ref=constraint_set.to_structured_fields(),
        routing_source=str(routing_context.get("routing_source", "guidance_contract")),
        routing_context=routing_context,
        governed_hypotheses=governed_hypotheses,
        benchmark_signals=dict(routing_context.get("benchmark_signals", {}) or {}),
    )


def build_diffusion_requests_from_guidance(pairs):
    """
    Convenience to build a list of DiffusionPromptSpec from (datapack, guidance) tuples.
    """
    prompts = []
    for dp, gp in pairs:
        prompts.append(build_diffusion_prompt_from_guidance(dp, gp))
    return prompts

# ==============================================================================
# Integration with VideoDiffusionStub (Stage 1/4)
# ==============================================================================

def prompt_to_diffusion_stub_input(prompt: DiffusionPromptSpec) -> Dict[str, Any]:
    """
    Convert DiffusionPromptSpec to inputs for VideoDiffusionStub.

    This bridges the orchestrator's prompt generation with the diffusion stub's API.
    """
    # Derive objective preset from objective vector
    obj_vec = prompt.objective_vector
    if len(obj_vec) >= 4:
        if obj_vec[0] > 1.5:  # High MPL weight
            objective_preset = "throughput"
        elif obj_vec[3] > 1.5:  # High safety weight
            objective_preset = "safety"
        elif obj_vec[2] > 1.5:  # High energy weight
            objective_preset = "energy_saver"
        else:
            objective_preset = "balanced"
    else:
        objective_preset = "balanced"

    # Derive energy profile from customer segment
    if prompt.customer_segment == "energy_saver":
        energy_profile = "SAVER"
    elif prompt.customer_segment == "premium_safety":
        energy_profile = "SAFE"
    elif prompt.customer_segment == "throughput_focused":
        energy_profile = "BOOST"
    else:
        energy_profile = "BASE"

    # Build econ context
    econ_context = {
        "wage_human": 18.0,  # Default
        "energy_price_kWh": 0.12,  # Default
        "customer_segment": prompt.customer_segment,
        "target_delta_mpl": prompt.target_economic_effect.get("delta_mpl", 0.0),
        "target_delta_error": prompt.target_economic_effect.get("delta_error", 0.0),
    }

    return {
        "episode_id": prompt.request_id,
        "media_refs": prompt.source_datapack_ids,
        "semantic_tags": prompt.semantic_tags,
        "objective_preset": objective_preset,
        "energy_profile": energy_profile,
        "econ_context": econ_context,
        "constraint_set": prompt.constraint_set_ref or {},
        "routing_context": {
            **dict(prompt.routing_context or {}),
            "routing_source": prompt.routing_source,
            "coverage_gap_score": float(prompt.coverage_gap_score),
            "economic_priority_score": float(prompt.economic_priority_score),
            "trust_priority_score": float(prompt.trust_priority_score),
            "meta_node_targets": list(prompt.meta_node_targets or []),
            "missing_skill_edges": list(prompt.missing_skill_edges or []),
            "missing_env_primitives": list(prompt.missing_env_primitives or []),
            "risk_family_targets": list(prompt.risk_family_targets or []),
            "affordance_family_targets": list(prompt.affordance_family_targets or []),
            "benchmark_signals": dict(prompt.benchmark_signals or {}),
        },
        "governed_hypotheses": list(prompt.governed_hypotheses or []),
    }


def generate_proposals_from_prompts(
    prompts: List[DiffusionPromptSpec],
    diffusion_stub=None,
) -> List[Dict[str, Any]]:
    """
    Generate diffusion proposals from orchestrator prompts using VideoDiffusionStub.

    Args:
        prompts: List of DiffusionPromptSpec from orchestrator
        diffusion_stub: Optional VideoDiffusionStub instance (creates if None)

    Returns:
        List of proposal dicts in datapack-like format
    """
    if diffusion_stub is None:
        from src.diffusion.real_video_diffusion_stub import VideoDiffusionStub
        diffusion_stub = VideoDiffusionStub()

    all_proposals = []

    for prompt in prompts:
        stub_input = prompt_to_diffusion_stub_input(prompt)

        proposals = diffusion_stub.propose_augmented_clips(
            episode_id=stub_input["episode_id"],
            media_refs=stub_input["media_refs"],
            semantic_tags=stub_input["semantic_tags"],
            objective_preset=stub_input["objective_preset"],
            energy_profile=stub_input["energy_profile"],
            econ_context=stub_input["econ_context"],
            constraint_set=stub_input.get("constraint_set"),
            routing_context=stub_input.get("routing_context"),
            hypotheses=stub_input.get("governed_hypotheses"),
            num_proposals=2,
        )

        # Convert proposals to datapack-like JSON format
        for proposal in proposals:
            from src.diffusion.real_video_diffusion_stub import proposal_to_dict
            proposal_dict = proposal_to_dict(proposal)

            # Add orchestrator context
            proposal_dict["orchestrator_request_id"] = prompt.request_id
            proposal_dict["orchestrator_rationale"] = prompt.rationale
            proposal_dict["target_economic_effect"] = prompt.target_economic_effect
            proposal_dict["constraint_set_ref"] = prompt.constraint_set_ref

            all_proposals.append(proposal_dict)

    return all_proposals


# ==============================================================================
# Coverage-gap-driven prompt generation (Phase C)
# ==============================================================================

def build_diffusion_prompt_from_coverage_gaps(
    coverage_graph: Any,
    *,
    env_name: str = "unknown",
    engine_type: str = "pybullet",
    task_type: str = "unknown",
    objective_vector: Optional[List[float]] = None,
    customer_segment: str = "balanced",
    economic_weight: float = 1.0,
    trust_weight: float = 1.0,
    readiness_weight: float = 1.0,
    limit: int = 5,
    gap_ranker: Any = None,
    gap_ranker_mode: str = "auto",
    backend_selector: Any = None,
    backend_selector_mode: str = "auto",
    branch_planner: Any = None,
    branch_planner_mode: str = "auto",
) -> List[DiffusionPromptSpec]:
    """Build diffusion prompts from ranked coverage gaps.

    This compatibility wrapper now compiles the canonical sim/synth/physics
    world state first and then adapts the WM-owned diffusion contracts into
    ``DiffusionPromptSpec`` objects for downstream consumers.

    Parameters
    ----------
    coverage_graph : SemanticCoverageGraph
        From ``src.world_model.semantic_coverage_graph``.
    env_name, engine_type, task_type, objective_vector, customer_segment
        Defaults for the generated prompts.
    economic_weight, trust_weight, readiness_weight
        Weights for gap ranking.
    limit : int
        Maximum number of prompts to generate.

    Returns
    -------
    list of DiffusionPromptSpec
    """
    runtime = SimSynthPhysicsRuntime(
        SimSynthPhysicsRuntimeConfig(
            economic_weight=economic_weight,
            trust_weight=trust_weight,
            readiness_weight=readiness_weight,
            agenda_limit=limit,
            default_backend=engine_type,
            default_objective=_objective_preset_from_vector(
                objective_vector or [1.0, 1.0, 1.0, 1.0, 0.0]
            ),
            gap_ranker_mode=str(gap_ranker_mode),
            backend_selector_mode=str(backend_selector_mode),
            branch_planner_mode=str(branch_planner_mode),
        )
    )
    world_state, _diffusion_plans = runtime.compile_world_state_and_diffusion_plans(
        coverage_graph,
        gap_ranker=gap_ranker,
        backend_selector=backend_selector,
        branch_planner=branch_planner,
        limit=limit,
    )
    return build_diffusion_prompts_from_world_state(
        world_state,
        coverage_graph=coverage_graph,
        env_name=env_name,
        engine_type=engine_type,
        task_type=task_type,
        objective_vector=objective_vector,
        customer_segment=customer_segment,
        limit=limit,
    )


def build_diffusion_prompts_from_world_state(
    world_state: SimSynthPhysicsWorldState,
    *,
    coverage_graph: Any = None,
    env_name: str = "unknown",
    engine_type: Optional[str] = None,
    task_type: Optional[str] = None,
    objective_vector: Optional[List[float]] = None,
    customer_segment: str = "balanced",
    limit: Optional[int] = None,
) -> List[DiffusionPromptSpec]:
    """Adapt WM-owned diffusion plans into orchestrator prompt specs."""

    plans = compile_gap_driven_diffusion_plans(
        world_state,
        coverage_graph=coverage_graph,
        limit=limit,
    )
    prompts: List[DiffusionPromptSpec] = []
    for plan in plans:
        prompts.append(
            DiffusionPromptSpec(
                request_id=str(plan.request_id),
                env_name=env_name,
                engine_type=engine_type or str(plan.env_backend),
                task_type=task_type or str(plan.task_family),
                objective_vector=objective_vector or list(plan.objective_vector),
                customer_segment=customer_segment,
                skill_ids=[],
                semantic_tags=list(plan.semantic_tags),
                camera_pose_hint={},
                difficulty_hint="gap_driven",
                rationale=str(plan.rationale),
                target_economic_effect={},
                source_datapack_ids=[],
                missing_skill_edges=list(plan.missing_skill_edges),
                missing_env_primitives=list(plan.missing_env_primitives),
                risk_family_targets=list(plan.risk_family_targets),
                affordance_family_targets=list(plan.affordance_family_targets),
                meta_node_targets=list(plan.meta_node_targets),
                coverage_gap_score=float(plan.coverage_gap_score),
                economic_priority_score=float(plan.economic_priority_score),
                trust_priority_score=float(plan.trust_priority_score),
                routing_source="sim_synth_physics_world_state",
                routing_context=dict(plan.routing_context),
                governed_hypotheses=list(plan.governed_hypotheses),
                benchmark_signals=dict(plan.benchmark_signals),
            )
        )
    return prompts

import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from src.constraints.constraint_set import ConstraintSet
from src.valuation.datapack_schema import DataPackMeta
from src.valuation.guidance_profile import GuidanceProfile


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
) -> List[DiffusionPromptSpec]:
    """Build diffusion prompts from ranked coverage gaps.

    Instead of generating prompts from datapack-level guidance and flat
    tag mixtures, this function compiles prompts from the semantic
    coverage graph's ranked missing edges.  Each prompt targets a specific
    missing skill–env-primitive combination.

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
    ranked_gaps = coverage_graph.rank_gaps(
        economic_weight=economic_weight,
        trust_weight=trust_weight,
        readiness_weight=readiness_weight,
        limit=limit,
    )

    prompts: List[DiffusionPromptSpec] = []
    for gap in ranked_gaps:
        if bool(getattr(gap, "metadata", {}).get("governance_blocked", False)):
            continue
        # Collect missing-edge information
        missing_skill_edges: List[Dict[str, str]] = []
        missing_env_prims: List[str] = []
        risk_targets: List[str] = []
        affordance_targets: List[str] = []

        src_node = coverage_graph.node_by_id(gap.source_id)
        tgt_node = coverage_graph.node_by_id(gap.target_id)

        if tgt_node:
            if tgt_node.node_type == "env_primitive":
                missing_env_prims.append(tgt_node.label)
            elif tgt_node.node_type == "risk_family":
                risk_targets.append(tgt_node.label)
            elif tgt_node.node_type == "affordance_family":
                affordance_targets.append(tgt_node.label)

        if src_node and tgt_node:
            missing_skill_edges.append({
                "from": src_node.label,
                "to": tgt_node.label,
                "edge_type": gap.edge_type,
            })

        rationale = (
            f"Gap-driven: missing {gap.edge_type} edge from "
            f"{gap.source_id} → {gap.target_id} "
            f"(economic_priority={gap.economic_priority:.2f}, "
            f"trust_priority={gap.trust_priority:.2f}, "
            f"wm_validation={float(getattr(gap, 'metadata', {}).get('wm_validation_pressure', 0.0)):.2f})"
        )

        prompts.append(DiffusionPromptSpec(
            request_id=str(uuid.uuid4()),
            env_name=env_name,
            engine_type=engine_type,
            task_type=task_type,
            objective_vector=objective_vector or [1.0, 1.0, 1.0, 1.0, 0.0],
            customer_segment=customer_segment,
            skill_ids=[],
            semantic_tags=[gap.source_id, gap.target_id],
            camera_pose_hint={},
            difficulty_hint="gap_driven",
            rationale=rationale,
            target_economic_effect={},
            source_datapack_ids=[],
            missing_skill_edges=missing_skill_edges,
            missing_env_primitives=missing_env_prims,
            risk_family_targets=risk_targets,
            affordance_family_targets=affordance_targets,
            coverage_gap_score=gap.gap_score(
                economic_weight=economic_weight,
                trust_weight=trust_weight,
                readiness_weight=readiness_weight,
            ),
            economic_priority_score=gap.economic_priority,
            trust_priority_score=gap.trust_priority,
        ))

    return prompts

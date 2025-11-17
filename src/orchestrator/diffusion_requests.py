import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_diffusion_prompt_from_guidance(
    dp: DataPackMeta,
    guidance: GuidanceProfile,
) -> DiffusionPromptSpec:
    skill_ids = []
    for s in dp.skill_trace or []:
        if "skill_id" in s:
            skill_ids.append(s["skill_id"])

    # Merge semantic tags from datapack, guidance, and VLA annotations
    semantic_tags = list(set(
        (dp.energy_driver_tags or []) +
        (guidance.semantic_tags or []) +
        (dp.semantic_tags or [])
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

            all_proposals.append(proposal_dict)

    return all_proposals

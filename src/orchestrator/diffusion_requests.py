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

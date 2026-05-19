"""WM-owned branch/render provider contracts for sim/synth/physics."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from src.envs.lsd3d_env.ggds import create_default_optimizer
from src.vision.nag.integration_lsd_backend import (
    NAGEditPolicyConfig,
    NAGFromLSDConfig,
)
from src.config.lsd_vector_scene_config import LSDVectorSceneConfig

from .common import stable_id
from .state import (
    BranchRenderProviderState,
    PhysicsAdaptationPolicyState,
    PhysicsContextState,
    SceneHierarchyState,
)


def _lsd_available() -> bool:
    return True


def _nag_renderer_available() -> bool:
    try:
        from src.vision.nag.gaussian_renderer import TORCH_AVAILABLE

        return bool(TORCH_AVAILABLE)
    except Exception:
        return False


def _ggds_concrete_available() -> bool:
    try:
        optimizer = create_default_optimizer()
        return bool(getattr(optimizer, "_is_initialized", False))
    except Exception:
        return False


def compile_branch_render_provider_state(
    *,
    branch_plan_id: str,
    generation_mode: str,
    branch_family: str,
    physics_context: PhysicsContextState,
    physics_adaptation_policy: PhysicsAdaptationPolicyState,
    benchmark_signals: Mapping[str, Any],
    scene_hierarchy: Optional[SceneHierarchyState] = None,
) -> BranchRenderProviderState:
    benchmark_ready = bool(
        benchmark_signals.get("ready", False) or benchmark_signals.get("benchmark_eligible", False)
    )
    semantic_grounding_ready = bool(
        benchmark_signals.get("semantic_grounding_non_heuristic", False)
    )
    lsd_available = _lsd_available()
    nag_renderer_available = _nag_renderer_available()
    ggds_concrete_available = _ggds_concrete_available()
    generation_mode = str(generation_mode or "")
    branch_family = str(branch_family or "")

    provider_kind = "lsd_scene_graph"
    render_mode = "lsd_vector_scene"
    counterfactual_mode = "none"
    ggds_mode = "disabled"
    provider_status = "ready" if lsd_available else "blocked"
    materialization_status = "ready" if lsd_available else "blocked"
    materialization_entrypoint = "src.motor_backend.factory:make_motor_backend"
    provider_config: dict[str, Any] = {
        "lsd_vector_scene": LSDVectorSceneConfig().to_dict(),
    }
    fallback_provider = ""
    fallback_reason = ""

    if generation_mode in {"physics_probe", "geometry_guarded_rollout"} or branch_family.endswith(":validate"):
        provider_kind = "nag_lsd_counterfactual"
        counterfactual_mode = "nag_counterfactual"
        render_mode = "gaussian_renderer" if nag_renderer_available else "pre_rendered_frames"
        provider_status = "ready" if nag_renderer_available else "partial"
        materialization_status = "ready" if nag_renderer_available else "partial"
        materialization_entrypoint = (
            "src.vision.nag.integration_lsd_backend:generate_nag_counterfactuals_for_lsd_episode"
        )
        provider_config = {
            "nag_from_lsd": NAGFromLSDConfig(
                use_stub_renderer=not nag_renderer_available,
                enable_scene_ir_filter=semantic_grounding_ready,
                enable_motion_plausibility_filter=benchmark_ready,
            ).__dict__,
            "nag_edit_policy": NAGEditPolicyConfig().__dict__,
        }
        if not nag_renderer_available:
            fallback_provider = "lsd_pre_rendered_frames"
            fallback_reason = (
                "NAG counterfactual path is compiled, but concrete Gaussian rendering requires torch-backed renderer availability"
            )
    elif generation_mode in {"targeted_synth_rollout", "coverage_branch"}:
        provider_kind = "lsd_ggds_scene"
        ggds_mode = "concrete_ggds" if ggds_concrete_available else "stub_only"
        render_mode = "ggds_texturing" if ggds_concrete_available else "lsd_scene_only"
        provider_status = "ready" if ggds_concrete_available else "partial"
        materialization_status = "ready" if ggds_concrete_available else "partial"
        materialization_entrypoint = "src.envs.lsd3d_env.ggds:GGDSOptimizer.optimize_scene"
        provider_config = {
            "lsd_vector_scene": LSDVectorSceneConfig(enable_nag_overlay=True).to_dict(),
            "ggds": create_default_optimizer().config.__dict__,
        }
        if not ggds_concrete_available:
            fallback_provider = "lsd_scene_graph"
            fallback_reason = (
                "GGDS texturing remains stub-only until a concrete LDM and renderer are wired into the optimizer"
            )

    if not lsd_available:
        provider_status = "blocked"
        fallback_provider = ""
        fallback_reason = "LSD scene provider is unavailable"

    payload = {
        "branch_plan_id": branch_plan_id,
        "provider_kind": provider_kind,
        "render_mode": render_mode,
        "counterfactual_mode": counterfactual_mode,
        "ggds_mode": ggds_mode,
    }
    scene_hierarchy_ref = (
        {}
        if scene_hierarchy is None
        else {
            "hierarchy_id": scene_hierarchy.hierarchy_id,
            "scene_id": scene_hierarchy.scene_id,
            "scene_kind": scene_hierarchy.scene_kind,
            "hierarchy_levels": list(scene_hierarchy.hierarchy_levels),
            "node_counts_by_level": dict(scene_hierarchy.node_counts_by_level),
            "materialization_status": scene_hierarchy.materialization_status,
        }
    )
    provider_config = dict(provider_config)
    if scene_hierarchy_ref:
        provider_config["scene_hierarchy"] = scene_hierarchy_ref
    return BranchRenderProviderState(
        provider_id=stable_id("branch_render_provider", payload),
        provider_kind=provider_kind,
        provider_status=provider_status,
        render_mode=render_mode,
        counterfactual_mode=counterfactual_mode,
        ggds_mode=ggds_mode,
        materialization_status=materialization_status,
        materialization_entrypoint=materialization_entrypoint,
        provider_config=provider_config,
        fallback_provider=fallback_provider,
        fallback_reason=fallback_reason,
        metadata={
            "branch_plan_id": branch_plan_id,
            "physics_backend": physics_context.backend,
            "physics_fidelity_tier": physics_context.fidelity_tier,
            "domain_randomization_profile": physics_adaptation_policy.domain_randomization_profile,
            "system_identification_profile": physics_adaptation_policy.system_identification_profile,
            "target_hardware_class": physics_adaptation_policy.target_hardware_class,
            "benchmark_ready": benchmark_ready,
            "semantic_grounding_ready": semantic_grounding_ready,
            "lsd_available": lsd_available,
            "nag_renderer_available": nag_renderer_available,
            "ggds_concrete_available": ggds_concrete_available,
            "provider_selection_policy": "wm_branch_render_provider_v1",
            "requires_companion_gpu": bool(
                provider_kind in {"nag_lsd_counterfactual", "lsd_ggds_scene"}
            ),
            "concrete_render_required": bool(
                benchmark_ready or generation_mode in {"physics_probe", "geometry_guarded_rollout"}
            ),
            "materialization_status": materialization_status,
            "materialization_entrypoint": materialization_entrypoint,
            "scene_hierarchy_ref": scene_hierarchy_ref,
            "gap_kind": (
                ""
                if provider_status == "ready"
                else "missing_concrete_render_provider"
            ),
            "target_runtime_stack": list(
                physics_adaptation_policy.metadata.get("target_runtime_stack", []) or []
            ),
        },
    )


__all__ = ["compile_branch_render_provider_state"]

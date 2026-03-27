"""WM-owned diffusion prompt plans derived from sim/synth/physics state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.economics.inferential_contract import coerce_inferential_learnability_contract

from .common import clip01, mapping, stable_id, strings
from .inferential import diffusion_priority_with_inferential_prior
from .state import SimSynthPhysicsWorldState, SyntheticBranchPlan


def _node(coverage_graph: Any, node_id: str) -> Any | None:
    if coverage_graph is None:
        return None
    try:
        return coverage_graph.node_by_id(node_id)
    except Exception:
        return None


def _objective_vector_from_preset(objective_preset: str) -> list[float]:
    if objective_preset == "throughput":
        return [2.0, 1.0, 1.0, 1.0, 0.0]
    if objective_preset == "energy_saver":
        return [1.0, 1.0, 2.0, 1.0, 0.0]
    if objective_preset == "safety":
        return [1.0, 1.0, 1.0, 2.0, 0.0]
    return [1.0, 1.0, 1.0, 1.0, 0.0]


def _primary_mode(job: Any, branch_plan: SyntheticBranchPlan) -> str:
    if str(job.risk_family):
        return "fragile_object_preservation"
    if str(job.objective_preset) == "energy_saver":
        return "energy_saver_retiming"
    if str(job.objective_preset) == "throughput":
        return "throughput_push"
    if str(branch_plan.generation_mode) == "physics_probe":
        return "semantic_disambiguation"
    if str(job.data_collection_intent) == "validate":
        return "semantic_disambiguation"
    return "geometry_guarded_continuation"


def _tag_mode_priority(mode: str, semantic_tags: list[str]) -> float:
    tag_set = {str(tag) for tag in semantic_tags}
    lower_mode = str(mode).lower()
    if any(tag in tag_set for tag in ("fragile", "safety", "avoid_collision", "collision")) and "fragile" in lower_mode:
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


def _action_conditioning(primary_mode: str, risk_targets: list[str], data_collection_intent: str) -> Dict[str, float]:
    if primary_mode == "fragile_object_preservation":
        return {"speed_scale": 0.25, "clearance_bias": 1.0}
    if primary_mode == "semantic_disambiguation":
        return {"camera_reframe": 1.0, "speed_scale": 0.2}
    if data_collection_intent == "exploit":
        return {"speed_scale": 0.65, "clearance_bias": 0.65 if risk_targets else 0.55}
    return {"speed_scale": 0.5, "clearance_bias": 1.0 if risk_targets else 0.75}


@dataclass(frozen=True)
class GapDrivenDiffusionPlan:
    """Typed diffusion prompt plan compiled from sim/synth/physics WM state."""

    request_id: str
    source_job_id: str
    source_branch_plan_id: str
    env_backend: str
    task_family: str
    objective_preset: str
    objective_vector: list[float]
    semantic_tags: list[str] = field(default_factory=list)
    rationale: str = ""
    missing_skill_edges: list[Dict[str, str]] = field(default_factory=list)
    missing_env_primitives: list[str] = field(default_factory=list)
    risk_family_targets: list[str] = field(default_factory=list)
    affordance_family_targets: list[str] = field(default_factory=list)
    meta_node_targets: list[str] = field(default_factory=list)
    coverage_gap_score: float = 0.0
    economic_priority_score: float = 0.0
    trust_priority_score: float = 0.0
    routing_context: Dict[str, Any] = field(default_factory=dict)
    governed_hypotheses: list[Dict[str, Any]] = field(default_factory=list)
    benchmark_signals: Dict[str, Any] = field(default_factory=dict)
    inferential_learnability_contract: Dict[str, Any] = field(default_factory=dict)
    diffusion_priority_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "gap_driven_diffusion_plan_v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_job_id": self.source_job_id,
            "source_branch_plan_id": self.source_branch_plan_id,
            "env_backend": self.env_backend,
            "task_family": self.task_family,
            "objective_preset": self.objective_preset,
            "objective_vector": list(self.objective_vector),
            "semantic_tags": strings(self.semantic_tags),
            "rationale": self.rationale,
            "missing_skill_edges": [mapping(item) for item in self.missing_skill_edges],
            "missing_env_primitives": strings(self.missing_env_primitives),
            "risk_family_targets": strings(self.risk_family_targets),
            "affordance_family_targets": strings(self.affordance_family_targets),
            "meta_node_targets": strings(self.meta_node_targets),
            "coverage_gap_score": clip01(self.coverage_gap_score),
            "economic_priority_score": clip01(self.economic_priority_score),
            "trust_priority_score": clip01(self.trust_priority_score),
            "routing_context": mapping(self.routing_context),
            "governed_hypotheses": [mapping(item) for item in self.governed_hypotheses],
            "benchmark_signals": mapping(self.benchmark_signals),
            "inferential_learnability_contract": mapping(self.inferential_learnability_contract),
            "diffusion_priority_score": clip01(self.diffusion_priority_score),
            "metadata": mapping(self.metadata),
            "version": self.version,
        }


def compile_gap_driven_diffusion_plans(
    world_state: SimSynthPhysicsWorldState,
    *,
    coverage_graph: Any = None,
    limit: Optional[int] = None,
) -> list[GapDrivenDiffusionPlan]:
    """Compile WM-owned diffusion prompt plans from canonical world state."""

    conditioning = world_state.diffusion_conditioning
    if conditioning is None:
        return []
    jobs = {job.job_id: job for job in world_state.simulation_agenda.jobs}
    admissible_branch_ids = set(
        list(getattr(world_state.gen2sim_admission, "admissible_branch_ids", []) or [])
    )
    benchmark_signals = dict(
        getattr(world_state.gen2sim_admission, "metadata", {}).get("benchmark_signals", {})
        or world_state.physics_context.metadata.get("benchmark_signals", {})
        or {}
    )
    benchmark_gate_ready = bool(
        getattr(world_state.gen2sim_admission, "benchmark_gate_ready", False)
    )
    semantic_grounding_mode = (
        "non_heuristic"
        if bool(
            benchmark_signals.get("semantic_grounding_non_heuristic", False)
            or getattr(world_state.gen2sim_admission, "metadata", {}).get("semantic_grounding_non_heuristic", False)
        )
        else "coverage_gap_pending"
    )

    diffusion_rows: list[tuple[float, GapDrivenDiffusionPlan]] = []
    ordered_branch_ids = list(conditioning.admissible_branch_ids or []) + list(
        conditioning.blocked_branch_ids or []
    )
    branch_order = {plan_id: index for index, plan_id in enumerate(ordered_branch_ids)}
    branch_plans = sorted(
        world_state.synthetic_branch_plans,
        key=lambda plan: (
            branch_order.get(plan.plan_id, len(branch_order)),
            -float(plan.expected_yield_score),
        ),
    )
    for branch_plan in branch_plans[:limit]:
        job = jobs.get(branch_plan.source_job_id)
        if job is None:
            continue
        coverage_targets = dict(job.coverage_targets or {})
        source_id = str(coverage_targets.get("source_id", ""))
        target_id = str(coverage_targets.get("target_id", ""))
        edge_type = str(coverage_targets.get("edge_type", ""))
        src_node = _node(coverage_graph, source_id)
        tgt_node = _node(coverage_graph, target_id)
        src_label = str(getattr(src_node, "label", source_id))
        tgt_label = str(getattr(tgt_node, "label", target_id))

        missing_skill_edges: list[Dict[str, str]] = []
        if src_label or tgt_label:
            missing_skill_edges.append(
                {
                    "from": src_label,
                    "to": tgt_label,
                    "edge_type": edge_type,
                }
            )

        missing_env_primitives: list[str] = []
        risk_targets: list[str] = []
        affordance_targets: list[str] = []
        target_node_type = str(getattr(tgt_node, "node_type", ""))
        if target_node_type == "env_primitive":
            missing_env_primitives.append(tgt_label)
        elif target_node_type == "risk_family":
            risk_targets.append(tgt_label)
        elif target_node_type == "affordance_family":
            affordance_targets.append(tgt_label)

        semantic_tags = sorted(
            {
                *list(conditioning.semantic_tags or []),
                source_id,
                target_id,
                str(job.task_family),
                str(job.risk_family),
                str(job.object_family),
            }
            - {"", "unknown"}
        )
        primary_mode = _primary_mode(job, branch_plan)
        action_conditioning = _action_conditioning(
            primary_mode,
            risk_targets,
            str(job.data_collection_intent),
        )
        branch_admissible = branch_plan.plan_id in admissible_branch_ids
        render_provider = branch_plan.render_provider
        branch_contract = coerce_inferential_learnability_contract(
            branch_plan.inferential_learnability_contract
        )
        diffusion_priority = diffusion_priority_with_inferential_prior(
            coverage_gap_score=float(job.coverage_gap_score),
            economic_priority=float(job.economic_priority),
            trust_priority=float(job.trust_priority),
            branch_yield_score=float(branch_plan.expected_yield_score),
            branch_admissible=branch_admissible,
            contract=branch_contract,
        )
        rationale = (
            f"{job.rationale} | wm_generation_mode={branch_plan.generation_mode}"
            f" | branch_plan_id={branch_plan.plan_id}"
        )
        routing_context = {
            **mapping(conditioning.routing_context),
            "routing_source": "sim_synth_physics_world_state",
            "coverage_origin": "coverage_gap_graph",
            "coverage_gap_score": float(job.coverage_gap_score),
            "economic_priority_score": float(job.economic_priority),
            "trust_priority_score": float(job.trust_priority),
            "benchmark_gate_ready": benchmark_gate_ready,
            "semantic_grounding_mode": semantic_grounding_mode,
            "agenda_ranking_policy": str(job.ranking_policy),
            "agenda_helper_status": dict(job.metadata.get("agenda_helper_status", {}) or {}),
            "branch_selection_policy": str(branch_plan.selection_policy),
            "branch_helper_status": dict(branch_plan.metadata.get("branch_helper_status", {}) or {}),
            "render_provider_kind": (
                "" if render_provider is None else str(render_provider.provider_kind)
            ),
            "render_provider_status": (
                "" if render_provider is None else str(render_provider.provider_status)
            ),
            "physics_selection_policy": str(world_state.physics_context.selection_policy),
            "physics_backend": str(world_state.physics_context.backend),
            "physics_helper_status": dict(
                world_state.physics_context.metadata.get("backend_helper_status", {}) or {}
            ),
            "missing_skill_edges": list(missing_skill_edges),
            "missing_env_primitives": list(missing_env_primitives),
            "risk_family_targets": list(risk_targets),
            "affordance_family_targets": list(affordance_targets),
            "meta_node_targets": [src_label] if src_label else [],
            "branch_plan_id": branch_plan.plan_id,
            "branch_admissible": branch_admissible,
            "diffusion_priority_score": diffusion_priority,
            "benchmark_signals": mapping(benchmark_signals),
            "inferential_learnability_contract": (
                branch_contract.to_dict() if branch_contract is not None else {}
            ),
        }
        governed_hypotheses = [
            {
                "hypothesis_id": stable_id(
                    "gap_hyp",
                    {
                        "job_id": job.job_id,
                        "plan_id": branch_plan.plan_id,
                        "mode": primary_mode,
                    },
                ),
                "mode": primary_mode,
                "semantic_tags": list(semantic_tags),
                "scores": {
                    "render_priority": clip01(
                        0.35
                        + (0.22 * float(job.economic_priority))
                        + (0.18 * float(job.trust_priority))
                        + (0.15 * float(job.coverage_gap_score))
                        + (0.10 * float(branch_plan.expected_yield_score))
                        + (0.1 * diffusion_priority)
                        + _tag_mode_priority(primary_mode, semantic_tags)
                    ),
                    "plausibility": clip01(
                        0.45
                        + (0.2 * float(job.trust_priority))
                        + (0.15 * float(job.readiness))
                        + (0.1 * float(branch_admissible))
                    ),
                    "novelty": clip01(
                        0.25
                        + (0.3 * float(job.coverage_gap_score))
                        + (0.15 * float(branch_plan.expected_yield_score))
                    ),
                    "economic_priority": clip01(float(job.economic_priority)),
                    "trust_priority": clip01(float(job.trust_priority)),
                },
                "rationale": rationale,
                "render_intent": {
                    "should_render": True,
                    "geometry_first": True,
                    "routing_source": "sim_synth_physics_world_state",
                    "branch_admissible": branch_admissible,
                },
                "action_conditioning": action_conditioning,
                "metadata": {
                    "gap_edge_type": edge_type,
                    "promotion_readiness": float(job.readiness),
                    "agenda_score_trace": dict(job.metadata.get("score_trace", {}) or {}),
                    "branch_plan_id": branch_plan.plan_id,
                    "branch_selection_policy": branch_plan.selection_policy,
                    "wm_generation_mode": branch_plan.generation_mode,
                    "render_provider": (
                        {} if render_provider is None else render_provider.to_dict()
                    ),
                    "branch_helper_trace": dict(branch_plan.metadata.get("branch_helper_trace", {}) or {}),
                    "diffusion_priority_score": diffusion_priority,
                },
            }
        ]

        diffusion_rows.append(
            (
                diffusion_priority,
                GapDrivenDiffusionPlan(
                request_id=stable_id(
                    "diffusion_prompt",
                    {
                        "job_id": job.job_id,
                        "plan_id": branch_plan.plan_id,
                        "mode": primary_mode,
                    },
                ),
                source_job_id=job.job_id,
                source_branch_plan_id=branch_plan.plan_id,
                env_backend=world_state.physics_context.backend,
                task_family=str(job.task_family),
                objective_preset=str(conditioning.objective_preset),
                objective_vector=_objective_vector_from_preset(str(conditioning.objective_preset)),
                semantic_tags=semantic_tags,
                rationale=rationale,
                missing_skill_edges=missing_skill_edges,
                missing_env_primitives=missing_env_primitives,
                risk_family_targets=risk_targets,
                affordance_family_targets=affordance_targets,
                meta_node_targets=[src_label] if src_label else [],
                coverage_gap_score=float(job.coverage_gap_score),
                economic_priority_score=float(job.economic_priority),
                trust_priority_score=float(job.trust_priority),
                routing_context=routing_context,
                governed_hypotheses=governed_hypotheses,
                benchmark_signals=benchmark_signals,
                inferential_learnability_contract=(
                    branch_contract.to_dict() if branch_contract is not None else {}
                ),
                diffusion_priority_score=diffusion_priority,
                metadata={
                    "world_state_id": world_state.state_id,
                    "conditioning_id": conditioning.conditioning_id,
                    "branch_admissible": branch_admissible,
                },
                ),
            )
        )
    diffusion_rows.sort(
        key=lambda item: (
            item[0],
            float(item[1].economic_priority_score),
            float(item[1].coverage_gap_score),
        ),
        reverse=True,
    )
    return [plan for _score, plan in diffusion_rows]

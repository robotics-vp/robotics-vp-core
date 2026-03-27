from src.orchestrator.semantic_simulation import compile_simulation_agenda
from src.world_model.semantic_coverage_graph import CoverageEdge, CoverageNode, SemanticCoverageGraph
from src.world_model.sim_synth_physics import compile_sim_synth_physics_world_state


def _make_test_graph() -> SemanticCoverageGraph:
    return SemanticCoverageGraph(
        nodes=[
            CoverageNode("task:drawer_vase", "task", "drawer_vase"),
            CoverageNode("hrl:grasp_handle", "skill", "Grasp Handle"),
            CoverageNode("prim:locate_handle", "env_primitive", "Locate Handle"),
            CoverageNode("risk:collision", "risk_family", "collision"),
        ],
        edges=[
            CoverageEdge(
                "hrl:grasp_handle",
                "prim:locate_handle",
                "requires",
                evidence_count=0,
                economic_priority=0.8,
                trust_priority=0.5,
                promotion_readiness=0.2,
            ),
            CoverageEdge(
                "hrl:grasp_handle",
                "risk:collision",
                "requires",
                evidence_count=0,
                economic_priority=0.3,
                trust_priority=0.2,
                promotion_readiness=0.9,
            ),
        ],
    )


class PromotedBackendSelector:
    benchmark_gate = {"ready": True}

    def select_backend(self, *, context):
        assert context["heuristic_backend"] == "pybullet"
        return {
            "preferred_backend": "isaac",
            "fidelity_tier": "high_fidelity",
            "domain_randomization_regime": "benchmark_focus",
        }


class ShadowBranchPlanner:
    benchmark_gate = {"ready": False}

    def plan_branch(self, *, job, context):
        assert context["physics_context"]["backend"] == "pybullet"
        return {
            "generation_mode": "neural_branch_candidate",
            "expected_yield_score": 0.95,
        }


def test_world_state_compiles_canonical_agenda_and_branch_plans() -> None:
    world_state = compile_sim_synth_physics_world_state(_make_test_graph(), limit=2)

    assert world_state.simulation_agenda.jobs[0].skill_edge == "Grasp Handle -> Locate Handle"
    assert world_state.physics_context.backend == "pybullet"
    assert len(world_state.synthetic_branch_plans) == 2
    assert world_state.diffusion_conditioning is not None
    assert world_state.diffusion_conditioning.branch_job_ids == [
        plan.source_job_id for plan in world_state.synthetic_branch_plans
    ]
    assert world_state.gen2sim_admission is not None


def test_world_state_uses_promoted_backend_selector_from_day_one() -> None:
    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        backend_selector=PromotedBackendSelector(),
        backend_selector_mode="auto",
    )

    assert world_state.physics_context.backend == "isaac"
    assert world_state.physics_context.selection_policy == "heuristic_plus_learned_backend_selector"
    assert world_state.physics_context.metadata["backend_helper_status"]["promotion_stage"] == "promoted"


def test_shadow_branch_planner_records_neural_trace_without_overriding() -> None:
    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        branch_planner=ShadowBranchPlanner(),
        branch_planner_mode="auto",
    )

    first_plan = world_state.synthetic_branch_plans[0]
    assert first_plan.generation_mode != "neural_branch_candidate"
    assert first_plan.selection_policy == "heuristic_plus_learned_branch_planner"
    assert first_plan.metadata["branch_helper_status"]["promotion_stage"] == "shadow_candidate"
    assert first_plan.metadata["branch_helper_trace"]["generation_mode"] == "neural_branch_candidate"


def test_legacy_agenda_wrapper_surfaces_wm_owned_fields() -> None:
    agenda = compile_simulation_agenda(_make_test_graph(), limit=1)

    assert agenda[0]["job_id"].startswith("sim_job_")
    assert agenda[0]["coverage_targets"]["source_id"] == "hrl:grasp_handle"
    assert "simulation_outcome_receipt" in agenda[0]["expected_receipts"]

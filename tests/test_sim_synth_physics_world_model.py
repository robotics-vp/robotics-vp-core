import json
from pathlib import Path

import pytest

from src.orchestrator.diffusion_requests import build_diffusion_prompts_from_world_state
from src.orchestrator.semantic_simulation import compile_simulation_agenda
from src.world_model.semantic_coverage_graph import CoverageEdge, CoverageNode, SemanticCoverageGraph
from src.world_model.sim_synth_physics import (
    LearnedBackendSelector,
    LearnedBranchPlanner,
    compile_gap_driven_diffusion_plans,
    compile_sim_synth_physics_world_state,
)
from src.world_model.sim_synth_physics.backend_selector import (
    BACKEND_LABELS,
    FIDELITY_LABELS,
    RANDOMIZATION_LABELS,
)


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
    assert (
        world_state.simulation_agenda.jobs[0].inferential_learnability_contract["subject_kind"]
        == "sim_synth_job"
    )
    assert (
        world_state.synthetic_branch_plans[0].inferential_learnability_contract["subject_kind"]
        == "synthetic_branch_plan"
    )


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


def test_diffusion_conditioning_uses_admitted_branches_for_budget() -> None:
    world_state = compile_sim_synth_physics_world_state(_make_test_graph(), limit=2)

    assert world_state.diffusion_conditioning is not None
    assert len(world_state.diffusion_conditioning.admissible_branch_ids) == 1
    assert len(world_state.diffusion_conditioning.blocked_branch_ids) == 1
    assert world_state.diffusion_conditioning.render_budget == 1
    assert (
        world_state.diffusion_conditioning.admissible_branch_ids
        == world_state.gen2sim_admission.admissible_branch_ids
    )


def test_diffusion_plans_rank_admitted_branches_ahead_of_blocked_ones() -> None:
    world_state = compile_sim_synth_physics_world_state(_make_test_graph(), limit=2)

    plans = compile_gap_driven_diffusion_plans(world_state, coverage_graph=_make_test_graph())

    assert len(plans) == 2
    assert plans[0].routing_context["branch_admissible"] is True
    assert plans[0].diffusion_priority_score >= plans[1].diffusion_priority_score
    assert plans[0].inferential_learnability_contract["subject_kind"] == "synthetic_branch_plan"


def test_legacy_agenda_wrapper_surfaces_wm_owned_fields() -> None:
    agenda = compile_simulation_agenda(_make_test_graph(), limit=1)

    assert agenda[0]["job_id"].startswith("sim_job_")
    assert agenda[0]["coverage_targets"]["source_id"] == "hrl:grasp_handle"
    assert "simulation_outcome_receipt" in agenda[0]["expected_receipts"]
    assert "inferential_learnability_contract" in agenda[0]


def test_diffusion_prompts_compile_from_world_state_contract() -> None:
    world_state = compile_sim_synth_physics_world_state(_make_test_graph(), limit=1)
    prompts = build_diffusion_prompts_from_world_state(
        world_state,
        coverage_graph=_make_test_graph(),
        limit=1,
    )

    assert prompts[0].routing_source == "sim_synth_physics_world_state"
    assert prompts[0].engine_type == world_state.physics_context.backend
    assert prompts[0].routing_context["physics_selection_policy"] == world_state.physics_context.selection_policy
    assert prompts[0].routing_context["branch_selection_policy"] == world_state.synthetic_branch_plans[0].selection_policy
    assert prompts[0].governed_hypotheses[0]["metadata"]["branch_plan_id"] == world_state.synthetic_branch_plans[0].plan_id


def _write_backend_selector_package(tmp_path: Path) -> str:
    torch = pytest.importorskip("torch")

    checkpoint_path = tmp_path / "backend_selector.pt"
    model = LearnedBackendSelector(hidden_dim=16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.backend_head.bias[BACKEND_LABELS.index("isaac")] = 8.0
        model.fidelity_head.bias[FIDELITY_LABELS.index("high_fidelity")] = 8.0
        model.randomization_head.bias[RANDOMIZATION_LABELS.index("benchmark_focus")] = 8.0
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": model.net[0].in_features,
            "hidden_dim": model.net[0].out_features,
        },
        checkpoint_path,
    )
    package_path = tmp_path / "backend_selector_package.json"
    package_path.write_text(
        json.dumps(
            {
                "package_id": "backend_selector_test_pkg",
                "checkpoint_path": checkpoint_path.name,
                "benchmark_gate": {"ready": True},
                "execution_preconditions": {
                    "unsatisfied_preconditions": [],
                    "benchmark_gate_ready": True,
                },
                "promotion_stage": "promoted",
                "inference_contract": {"helper_blend_policy": "bounded_backend_selector_helper_v1"},
                "metadata": {"target_hardware_class": "unitree_g1_r1_class"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(package_path)


def _write_branch_planner_package(tmp_path: Path) -> str:
    torch = pytest.importorskip("torch")

    checkpoint_path = tmp_path / "branch_planner.pt"
    model = LearnedBranchPlanner(hidden_dim=16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.mode_head.bias[4] = 8.0
        model.yield_head[0].bias.fill_(3.0)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": model.net[0].in_features,
            "hidden_dim": model.net[0].out_features,
        },
        checkpoint_path,
    )
    package_path = tmp_path / "branch_planner_package.json"
    package_path.write_text(
        json.dumps(
            {
                "package_id": "branch_planner_test_pkg",
                "checkpoint_path": checkpoint_path.name,
                "benchmark_gate": {"ready": True},
                "execution_preconditions": {
                    "unsatisfied_preconditions": [],
                    "benchmark_gate_ready": True,
                },
                "promotion_stage": "promoted",
                "inference_contract": {"helper_blend_policy": "bounded_branch_planner_helper_v1"},
                "metadata": {"target_hardware_class": "unitree_g1_r1_class"},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(package_path)


def test_world_state_loads_backend_selector_runtime_package(tmp_path: Path) -> None:
    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        backend_selector=_write_backend_selector_package(tmp_path),
        backend_selector_mode="auto",
    )

    helper_status = world_state.physics_context.metadata["backend_helper_status"]
    helper_trace = world_state.physics_context.metadata["backend_helper_trace"]

    assert world_state.physics_context.backend == "isaac"
    assert world_state.physics_context.fidelity_tier == "high_fidelity"
    assert world_state.physics_context.domain_randomization_regime == "benchmark_focus"
    assert world_state.physics_context.selection_policy == "heuristic_plus_learned_backend_selector"
    assert helper_status["status"] == "loaded"
    assert helper_status["promotion_stage"] == "promoted"
    assert helper_status["package_id"] == "backend_selector_test_pkg"
    assert helper_trace["preferred_backend"] == "isaac"


def test_world_state_loads_branch_planner_runtime_package(tmp_path: Path) -> None:
    world_state = compile_sim_synth_physics_world_state(
        _make_test_graph(),
        branch_planner=_write_branch_planner_package(tmp_path),
        branch_planner_mode="auto",
    )

    first_plan = world_state.synthetic_branch_plans[0]
    helper_status = first_plan.metadata["branch_helper_status"]
    helper_trace = first_plan.metadata["branch_helper_trace"]

    assert first_plan.generation_mode == "neural_branch_candidate"
    assert first_plan.selection_policy == "heuristic_plus_learned_branch_planner"
    assert helper_trace["expected_yield_score"] > 0.9
    assert helper_trace["expected_yield_score"] > first_plan.metadata["heuristic_expected_yield_score"]
    assert first_plan.expected_yield_score > 0.7
    assert helper_status["status"] == "loaded"
    assert helper_status["promotion_stage"] == "promoted"
    assert helper_status["package_id"] == "branch_planner_test_pkg"
    assert helper_trace["generation_mode"] == "neural_branch_candidate"

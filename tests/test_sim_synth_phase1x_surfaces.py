from src.world_model.semantic_coverage_graph import CoverageEdge, CoverageNode, SemanticCoverageGraph
from src.world_model.sim_synth_physics import compile_sim_synth_physics_world_state


def _graph() -> SemanticCoverageGraph:
    return SemanticCoverageGraph(
        nodes=[
            CoverageNode("task:drawer_vase", "task", "drawer_vase"),
            CoverageNode("skill:grasp", "skill", "grasp"),
            CoverageNode("risk:collision", "risk_family", "collision"),
        ],
        edges=[
            CoverageEdge(
                "skill:grasp",
                "risk:collision",
                "requires",
                evidence_count=0,
                economic_priority=0.7,
                trust_priority=0.3,
                promotion_readiness=0.2,
            )
        ],
    )


def test_phase1x_surfaces_compile_from_local_context() -> None:
    state = compile_sim_synth_physics_world_state(
        _graph(),
        semantic_context={
            "scene_id": "drawer_vase_scene_a",
            "scene_kind": "workcell",
            "scene_hierarchy_levels": ["scene", "surface", "object"],
            "region_ids": ["countertop"],
            "object_ids": ["drawer", "vase"],
            "sensor_profile": "rgbd_front",
        },
        benchmark_signals={
            "ready": True,
            "vector_env_count": 4,
            "measurement_window_steps": 32,
            "differentiable_provider_available": True,
            "differentiable_provider_family": "jaxsim_like",
            "surrogate_provider_available": True,
            "surrogate_provider_family": "windinet_like",
            "surrogate_calibration_status": "shadow_calibrated",
        },
    )

    assert state.task_measurements is not None
    assert state.task_measurements.benchmark_gate_ready is True
    assert state.task_measurements.vector_env_count == 4
    assert state.task_measurements.measurement_window_steps == 32
    assert state.simulator_backend_contract is not None
    assert state.simulator_backend_contract.supported_task_families == ["unknown"]
    assert state.task_definition_contract is not None
    assert state.task_definition_contract.required_measurements == state.task_measurements.measurement_names
    assert state.scene_hierarchy is not None
    assert state.scene_hierarchy.scene_id == "drawer_vase_scene_a"
    assert state.scene_hierarchy.hierarchy_levels == ["scene", "surface", "object"]
    assert state.scene_hierarchy.node_counts_by_level["object"] == 2
    assert state.synthetic_branch_plans[0].metadata["scene_hierarchy_ref"]["scene_id"] == "drawer_vase_scene_a"
    assert (
        state.synthetic_branch_plans[0].render_provider.provider_config["scene_hierarchy"]["scene_id"]
        == "drawer_vase_scene_a"
    )
    assert state.differentiable_physics_provider is not None
    assert state.differentiable_physics_provider.available is True
    assert state.differentiable_physics_provider.provider_family == "jaxsim_like"
    assert state.surrogate_physics_provider is not None
    assert state.surrogate_physics_provider.available is True
    assert state.surrogate_physics_provider.provider_family == "windinet_like"

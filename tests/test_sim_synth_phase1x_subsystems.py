from src.world_model.semantic_coverage_graph import CoverageEdge, CoverageNode, SemanticCoverageGraph
from src.world_model.sim_synth_physics import (
    PHASE1X_SUBSYSTEM_SPECS,
    build_backend_selector_rows_from_receipts,
    build_branch_planner_rows_from_receipts,
    build_phase1x_subsystem_index,
    compile_sim_synth_physics_world_state,
)


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


def test_phase1x_subsystem_index_maps_all_ten_subsystems() -> None:
    index = build_phase1x_subsystem_index()

    assert len(PHASE1X_SUBSYSTEM_SPECS) == 10
    assert index["schema_version"] == "phase1x_subsystem_index_v1"
    assert index["subsystem_count"] == 10
    assert [item["ordinal"] for item in index["subsystems"]] == list(range(1, 11))
    assert index["provider_ownership_rule"] == (
        "providers_may_span_subsystems_but_never_own_wm_truth"
    )
    assert index["honest_remaining_blocker_class"] == (
        "external_gpu_runtime_asset_benchmark_or_provider_evidence"
    )
    assert index["coverage_summary"]["subsystems_with_typed_state_surfaces"] == 10
    assert index["coverage_summary"]["subsystems_with_receipt_surfaces"] == 10
    assert index["coverage_summary"]["subsystems_with_learned_or_reserved_seams"] >= 8

    subsystem_ids = {item["subsystem_id"] for item in index["subsystems"]}
    assert "phase1x_subsystem_01_backend_runtime_provider_surface" in subsystem_ids
    assert "phase1x_subsystem_10_training_worthiness_synthetic_yield_evaluator" in subsystem_ids
    assert all(
        item["provider_truth_owner"] == "sim_synth_physics_wm"
        for item in index["subsystems"]
    )


def test_compiled_world_state_embeds_phase1x_subsystem_index() -> None:
    state = compile_sim_synth_physics_world_state(
        _graph(),
        semantic_context={
            "scene_id": "drawer_vase_scene_a",
            "scene_kind": "workcell",
            "scene_hierarchy_levels": ["scene", "surface", "object"],
            "object_ids": ["drawer", "vase"],
        },
        benchmark_signals={"ready": False},
    )

    index = state.metadata["phase1x_subsystem_index"]
    by_id = {item["subsystem_id"]: item for item in index["subsystems"]}

    assert index["schema_version"] == "phase1x_subsystem_index_v1"
    assert index["structural_status"] == "mapped_static_with_runtime_refs"
    assert index["subsystem_count"] == 10
    assert index["coverage_summary"]["subsystems_with_present_artifact_refs"] >= 8
    assert index["coverage_summary"]["subsystems_with_compiled_receipt_families"] == 10

    backend_surface = by_id["phase1x_subsystem_01_backend_runtime_provider_surface"]
    assert backend_surface["artifact_refs_present"]["backend_execution_binding_id"] == (
        state.backend_execution_binding.binding_id
    )
    assert "backend_execution_binding_receipt_v1" in backend_surface["receipt_surfaces_present"]
    assert "LearnedBackendSelector" in backend_surface["learned_seams"]

    training_surface = by_id[
        "phase1x_subsystem_10_training_worthiness_synthetic_yield_evaluator"
    ]
    assert training_surface["artifact_refs_present"]["admission_id"] == (
        state.gen2sim_admission.admission_id
    )
    assert "gen2sim_admission_receipt_v1" in training_surface["receipt_surfaces_present"]
    assert "phase1x_training_gate_v1" in training_surface["promotion_gates"]


def test_phase1x_subsystem_index_survives_into_training_rows() -> None:
    state = compile_sim_synth_physics_world_state(
        _graph(),
        benchmark_signals={"ready": False},
    )
    bundle = {"bundle_id": "subsystem_bundle", "world_state": state.to_dict()}

    backend_rows = build_backend_selector_rows_from_receipts([bundle])
    branch_rows = build_branch_planner_rows_from_receipts([bundle])

    for row in (backend_rows[0], branch_rows[0]):
        metadata = row["metadata"]
        assert metadata["phase1x_subsystem_index_id"] == (
            state.metadata["phase1x_subsystem_index"]["index_id"]
        )
        assert metadata["phase1x_subsystem_count"] == 10
        assert metadata["phase1x_subsystem_provider_ownership_rule"] == (
            "providers_may_span_subsystems_but_never_own_wm_truth"
        )
        assert metadata["phase1x_subsystem_blocker_class"] == (
            "external_gpu_runtime_asset_benchmark_or_provider_evidence"
        )
        assert (
            "phase1x_subsystem_10_training_worthiness_synthetic_yield_evaluator"
            in metadata["phase1x_subsystem_ids"]
        )
        assert (
            metadata["phase1x_subsystem_coverage_summary"][
                "subsystems_with_compiled_receipt_families"
            ]
            == 10
        )

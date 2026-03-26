from src.observation.adapter import ObservationAdapter
from src.observation.condition_vector_builder import ConditionVectorBuilder
from src.semantic.models import EconSlice, MetaTransformerSlice, SemanticSnapshot


def _semantic_snapshot(metadata: dict) -> SemanticSnapshot:
    return SemanticSnapshot(
        task_id="task_semantic",
        ontology_proposals=[],
        task_refinements=[],
        semantic_tags=["drawer"],
        econ_slice=EconSlice(
            task_id="task_semantic",
            avg_mpl_units_per_hour=4.0,
            avg_wage_parity=1.0,
            avg_energy_cost=0.1,
            avg_error_rate=0.2,
        ),
        meta_slice=MetaTransformerSlice(task_id="task_semantic"),
        metadata=metadata,
    )


def test_observation_adapter_threads_semantic_truth_into_condition_vector() -> None:
    adapter = ObservationAdapter(
        policy_registry={},
        condition_builder=ConditionVectorBuilder(),
        use_condition_vector=True,
    )
    snapshot = _semantic_snapshot(
        {
            "scene_tracks_backend": "passthrough",
            "teacher_runtime_backend_selected": "unavailable",
            "vision_backbone_selected": "unavailable",
            "semantic_grounding_mode": "heuristic_fallback",
            "benchmark_signals": {
                "scene_tracks_backend": "passthrough",
                "semantic_grounding_non_heuristic": False,
            },
            "execution_preconditions": {
                "ready": False,
                "readiness_score": 0.2,
                "blocking_preconditions": ["signal_bool::semantic_grounding_non_heuristic"],
            },
            "semantic_fusion": {"status": "blocked", "ready_fraction": 0.0},
        }
    )

    _observation, condition = adapter.build_observation_and_condition(
        vision_frame=None,
        vision_latent=None,
        reward_scalar=0.0,
        reward_components={},
        econ_vector=None,
        semantic_snapshot=snapshot,
        recap_scores=None,
        descriptor={"task_id": "task_semantic", "env_id": "sim_env", "backend_id": "sim", "metadata": {}},
        episode_metadata={"episode_id": "ep_semantic"},
        condition_kwargs={"enable_condition": True},
    )

    assert condition is not None
    assert condition.ood_risk_level > 0.0
    assert condition.recovery_priority > 0.0
    assert condition.metadata["execution_preconditions"]["ready"] is False
    assert condition.metadata["benchmark_signals"]["semantic_grounding_non_heuristic"] is False
    assert condition.metadata["semantic_fusion"]["status"] == "blocked"

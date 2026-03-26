"""Tests for semantic policy helpers."""
from src.motor_backend.datapacks import DatapackConfig
from src.orchestrator.semantic_policy import (
    DatapackSelectionDecision,
    MissingScenarioSpec,
    apply_arh_penalty,
    detect_semantic_gaps,
    rank_datapacks_for_intent,
    select_datapacks_for_intent,
    summarize_datapack_selection,
)
from src.scenarios.metadata import ScenarioMetadata


def test_apply_arh_penalty_adjusts_mpl():
    metrics = {"mpl_units_per_hour": 100.0, "anti_reward_hacking_suspicious": 1.0}
    adjusted = apply_arh_penalty(metrics, penalty_factor=0.2)
    assert adjusted["mpl_units_per_hour_adjusted"] == 80.0
    assert adjusted["anti_reward_hacking_penalty"] == 0.2


def test_detect_semantic_gaps():
    scenario = ScenarioMetadata(
        scenario_id="holosoma:task:obj:run",
        task_id="task",
        motor_backend="holosoma",
        objective_name="obj",
        objective_weights={"mpl_weight": 1.0},
        datapack_ids=["dp1"],
        datapack_tags=["humanoid"],
        task_tags=[],
        robot_families=["G1"],
    )
    missing = detect_semantic_gaps(["humanoid", "warehouse"], "G1", [scenario])
    assert missing == [MissingScenarioSpec(tags=["warehouse"], robot_family="G1")]


def test_select_datapacks_prefers_non_arh():
    candidates = [
        DatapackConfig(id="dp1", description="", tags=["humanoid", "warehouse"]),
        DatapackConfig(id="dp2", description="", tags=["humanoid", "warehouse"]),
    ]
    scenarios = [
        {
            "datapack_ids": ["dp1"],
            "datapack_tags": ["humanoid", "warehouse"],
            "robot_families": ["G1"],
            "train_metrics_anti_reward_hacking_suspicious": 1.0,
        }
    ]

    selected = select_datapacks_for_intent(
        tags=["humanoid", "warehouse"],
        robot_family="G1",
        objective_hint=None,
        candidates=candidates,
        scenarios=scenarios,
    )
    assert selected
    assert selected[0].id == "dp2"


def test_rank_datapacks_prefers_grounded_high_quality_history() -> None:
    candidates = [
        DatapackConfig(id="dp1", description="", tags=["humanoid", "warehouse"], objective_hint="baseline"),
        DatapackConfig(id="dp2", description="", tags=["humanoid", "warehouse"], objective_hint="baseline"),
    ]
    scenarios = [
        {
            "datapack_ids": ["dp1"],
            "datapack_tags": ["humanoid", "warehouse"],
            "robot_families": ["G1"],
            "train_metrics": {"mpl_units_per_hour": 55.0, "anti_reward_hacking_suspicious": 0.0},
            "eval_metrics": {"mpl_units_per_hour": 57.0},
        },
        {
            "datapack_ids": ["dp2"],
            "datapack_tags": ["humanoid", "warehouse"],
            "robot_families": ["G1"],
            "train_metrics": {"mpl_units_per_hour": 80.0, "anti_reward_hacking_suspicious": 0.0},
            "eval_metrics": {"mpl_units_per_hour": 82.0},
        },
    ]
    ranked = rank_datapacks_for_intent(
        tags=["humanoid", "warehouse"],
        robot_family="G1",
        objective_hint="baseline",
        candidates=candidates,
        scenarios=scenarios,
        candidate_metadata_by_id={
            "dp1": {
                "quality_score": 0.4,
                "metadata": {
                    "scene_tracks_backend": "passthrough",
                    "vision_backbone_selected": "real",
                    "semantic_grounding_mode": "heuristic_fallback",
                    "semantic_memory_grounded": True,
                },
            },
            "dp2": {
                "quality_score": 0.9,
                "metadata": {
                    "scene_tracks_backend": "real",
                    "vision_backbone_selected": "real",
                    "semantic_grounding_mode": "non_heuristic",
                    "semantic_memory_grounded": True,
                },
            },
        },
    )

    assert ranked
    assert ranked[0].datapack.id == "dp2"
    assert ranked[0].benchmark_support["semantic_grounding_non_heuristic"] is True
    assert ranked[0].historical_support["scenario_count"] == 1
    assert ranked[0].selection_policy == "heuristic_only"
    assert ranked[0].heuristic_score >= ranked[0].score - 1e-9
    assert ranked[0].selection_features["semantic_grounding_non_heuristic"] == 1.0


def test_rank_datapacks_accepts_learned_helper_adjustment() -> None:
    candidates = [
        DatapackConfig(id="dp1", description="", tags=["humanoid", "warehouse"]),
        DatapackConfig(id="dp2", description="", tags=["humanoid", "warehouse"]),
    ]
    ranked = rank_datapacks_for_intent(
        tags=["humanoid", "warehouse"],
        robot_family="G1",
        objective_hint=None,
        candidates=candidates,
        scenarios=[],
        candidate_metadata_by_id={
            "dp1": {"quality_score": 0.2, "novelty_score": 0.1},
            "dp2": {"quality_score": 0.2, "novelty_score": 0.9},
        },
        selection_scorer_package={
            "package_id": "selection_helper_v1",
            "feature_weights": {"novelty_score": 2.5},
            "bias": 0.0,
            "max_adjustment": 0.5,
        },
    )

    assert ranked
    assert ranked[0].selection_policy == "heuristic_plus_learned_helper"
    assert ranked[0].scorer_package_id == "selection_helper_v1"
    assert ranked[0].learned_score > 0.0
    assert ranked[0].scorer_trace["top_contributors"][0]["feature"] == "novelty_score"
    assert ranked[0].selection_features["novelty_score"] >= ranked[-1].selection_features["novelty_score"]


def test_summarize_datapack_selection_keeps_top_candidate_reasons() -> None:
    ranked = [
        DatapackSelectionDecision(
            datapack=DatapackConfig(id="dp_summary", description="", tags=["warehouse"]),
            score=3.2,
            source="ontology",
            heuristic_score=3.2,
            learned_score=0.0,
            selection_policy="heuristic_only",
            matched_tags=["warehouse"],
            missing_tags=[],
            gap_fill_tags=["warehouse"],
            exact_tag_match=True,
            objective_match=True,
            historical_support={"scenario_count": 2, "support_score": 0.8},
            benchmark_support={"benchmark_eligible": True},
            reasons=["exact_tag_match", "benchmark_eligible"],
        )
    ]

    summary = summarize_datapack_selection(
        ranked,
        selected=ranked,
        tags=["warehouse"],
        robot_family="G1",
        objective_hint="baseline",
        selection_helper_status={"mode": "disabled", "status": "disabled"},
    )

    assert summary["selected_ids"] == ["dp_summary"]
    assert summary["selection_policy"] == "heuristic_only"
    assert summary["selection_helper_status"]["status"] == "disabled"
    assert summary["top_candidates"][0]["reasons"] == ["exact_tag_match", "benchmark_eligible"]

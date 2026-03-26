import json

from src.orchestrator.datapack_selection_training import (
    build_datapack_selection_training_dataset,
    train_datapack_selection_scorer_package,
)


def _run_row(run_id: str, *, outcome_reward: float, promotion_stage: str) -> dict:
    return {
        "scenario_id": run_id,
        "eval_metrics": {
            "mpl_units_per_hour": 82.0,
            "wage_parity": 0.9,
            "reward_scalar_sum": outcome_reward,
            "error_rate": 0.02,
            "anti_reward_hacking_suspicious": 0.0,
        },
        "selection_summary": {
            "required_tags": ["warehouse", "grasp"],
            "robot_family": "G1",
            "objective_hint": "baseline",
            "candidate_count": 2,
            "selection_policy": "heuristic_plus_learned_helper",
            "selected_ids": ["dp_good"],
            "selection_helper_status": {
                "status": "available",
                "promotion_stage": promotion_stage,
            },
            "selection_context": {
                "required_tag_count_norm": 0.25,
                "gap_pressure": 0.5,
                "candidate_pool_size_norm": 0.2,
                "benchmark_ready_ratio": 0.5,
                "execution_ready_ratio": 0.5,
                "history_density": 0.1,
                "cold_start_pressure": 0.9,
                "objective_present": 1.0,
                "robot_specificity": 1.0,
            },
            "top_candidates": [
                {
                    "datapack_id": "dp_good",
                    "selection_features": {
                        "tag_coverage": 1.0,
                        "exact_tag_match": 1.0,
                        "gap_fill_score": 0.5,
                        "objective_match": 1.0,
                        "history_support_score": 0.4,
                        "quality_score": 0.9,
                        "novelty_score": 0.4,
                        "semantic_grounding_non_heuristic": 1.0,
                        "benchmark_eligible": 1.0,
                        "execution_ready": 1.0,
                        "cold_start_bonus": 0.0,
                        "max_arh_penalty": 0.0,
                        "mean_adjusted_mpl_norm": 0.8,
                        "mean_reward_norm": 0.7,
                        "scenario_count_norm": 0.2,
                        "eval_count_norm": 0.2,
                    },
                },
                {
                    "datapack_id": "dp_alt",
                    "selection_features": {
                        "tag_coverage": 1.0,
                        "exact_tag_match": 1.0,
                        "gap_fill_score": 0.0,
                        "objective_match": 1.0,
                        "history_support_score": 0.2,
                        "quality_score": 0.3,
                        "novelty_score": 0.9,
                        "semantic_grounding_non_heuristic": 0.0,
                        "benchmark_eligible": 0.0,
                        "execution_ready": 0.0,
                        "cold_start_bonus": 1.0,
                        "max_arh_penalty": 0.0,
                        "mean_adjusted_mpl_norm": 0.4,
                        "mean_reward_norm": 0.5,
                        "scenario_count_norm": 0.0,
                        "eval_count_norm": 0.0,
                    },
                },
            ],
        },
    }


def test_datapack_selection_training_dataset_tracks_context_and_policies() -> None:
    dataset = build_datapack_selection_training_dataset(
        [
            _run_row("run_a", outcome_reward=7.5, promotion_stage="shadow_candidate"),
            _run_row("run_b", outcome_reward=6.0, promotion_stage="promoted"),
        ]
    )

    assert dataset.examples
    assert dataset.summary["selection_context_contract"]["schema_version"] == "datapack_selection_context_v1"
    assert dataset.summary["selection_policy_counts"]["heuristic_plus_learned_helper"] == 2
    assert dataset.summary["promotion_stage_counts"]["shadow_candidate"] == 1
    assert dataset.summary["promotion_stage_counts"]["promoted"] == 1
    assert dataset.examples[0].selection_context["gap_pressure"] == 0.5


def test_datapack_selection_training_learns_context_conditioning() -> None:
    dataset = build_datapack_selection_training_dataset(
        [
            _run_row("run_a", outcome_reward=7.5, promotion_stage="shadow_candidate"),
            _run_row("run_b", outcome_reward=6.0, promotion_stage="promoted"),
        ]
    )

    scorer_package = train_datapack_selection_scorer_package(dataset)
    payload = scorer_package.to_dict()

    assert payload["context_weights"]
    assert payload["max_adjustment"] >= payload["min_adjustment"] >= 0.0
    assert payload["metadata"]["conditioning_contract"] == "datapack_selection_context_v1"
    assert "future_conditioning_path" in payload["metadata"]
    json.dumps(payload)

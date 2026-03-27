from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.orchestrator.queue_dispatch_policy import (
    build_queue_dispatch_feature_map,
    extract_queue_dispatch_target,
)
from src.orchestrator.queue_selection import QueueDispatchConfig, apply_live_queue_selection
from src.orchestrator.queue_dispatch_policy_training import (
    build_queue_dispatch_training_dataset,
    train_queue_dispatch_policy_model,
)


def _entry(episode_id: str, *, priority: float, success: bool, value: float) -> dict:
    return {
        "episode_id": episode_id,
        "priority_score": priority,
        "replay_action": "upweight" if success else "downweight",
        "tags": ["frontier_candidate"] if success else ["downweight_candidate"],
        "metadata": {
            "promotion_stage": "advisory",
            "influence_source": "heuristic",
            "deploy_recommendation": "allow_shadow" if success else "require_review",
            "pricing_recommendation": "publish" if success else "review",
            "datapack_recommendation": "keep" if success else "downweight",
            "semantic_runtime_score": {
                "meta_route_success_probability": 0.85 if success else 0.2,
                "orchestration_route_success_probability": 0.81 if success else 0.25,
                "authority_success_probability": 0.8 if success else 0.3,
                "estimated_regret": 0.1 if success else 0.8,
            },
            "evidence": {
                "receipt_feedback": {
                    "deployment_outcome": {
                        "task_success": success,
                        "objective_satisfied": success,
                        "realized_value": value,
                        "pricing_accepted": success,
                    }
                }
            },
        },
    }


def _write_package(tmp_path: Path) -> Path:
    payload = {"queue_name": "shadow_advisory_queue", "entries": [_entry("ep_high", priority=0.2, success=True, value=1.5), _entry("ep_low", priority=0.9, success=False, value=-1.0)]}
    dataset = build_queue_dispatch_training_dataset([payload, payload])
    checkpoint_path = tmp_path / "queue_dispatch_policy.pt"
    _, training_result = train_queue_dispatch_policy_model(
        dataset,
        epochs=2,
        hidden_dim=16,
        save_path=str(checkpoint_path),
    )
    assert training_result["checkpoint_path"] is not None
    package_path = tmp_path / "queue_dispatch_policy_package.json"
    package_path.write_text(
        json.dumps(
            {
                "package_id": "queue_dispatch_policy_test",
                "checkpoint_path": checkpoint_path.name,
                "model_config": {"input_dim": len(dataset.summary["feature_names"]), "hidden_dim": 16},
                "benchmark_gate": {"ready": False},
                "execution_preconditions": {"benchmark_gate_ready": False},
                "promotion_stage": "shadow_candidate",
                "inference_contract": {"helper_blend_policy": {"shadow_candidate_helper_weight": 0.12, "promoted_helper_weight": 0.35}},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return package_path


def test_queue_dispatch_feature_map_and_target_reflect_receipts() -> None:
    feature_map = build_queue_dispatch_feature_map(_entry("ep_ok", priority=0.6, success=True, value=1.0))
    target = extract_queue_dispatch_target(_entry("ep_ok", priority=0.6, success=True, value=1.0))

    assert feature_map["receipt_task_success"] == 1.0
    assert feature_map["semantic_route_success_prob"] > 0.5
    assert target["target_source"] == "receipt_feedback"
    assert target["dispatch_score"] > 0.5


def test_apply_live_queue_selection_uses_queue_policy_helper(tmp_path: Path) -> None:
    episodes = [
        {"descriptor": {"pack_id": "ep_low", "sampling_weight": 1.0}},
        {"descriptor": {"pack_id": "ep_high", "sampling_weight": 1.0}},
    ]
    payload = {
        "queue_name": "shadow_advisory_queue",
        "entries": [
            _entry("ep_low", priority=0.9, success=False, value=-1.0),
            _entry("ep_high", priority=0.2, success=True, value=1.5),
        ],
    }
    package_path = _write_package(tmp_path)

    dispatch = apply_live_queue_selection(
        episodes,
        live_queue_selection=payload,
        config=QueueDispatchConfig(
            mode="bounded_reweight",
            policy_helper_mode="auto",
            policy_package_path=str(package_path),
        ),
    )

    entry_map = {row["episode_id"]: row for row in dispatch["entries"]}
    assert entry_map["ep_high"]["dispatch_policy_source"] == "heuristic_plus_learned_helper"
    assert entry_map["ep_high"]["evidence"]["queue_policy_trace"]["helper_weight"] == pytest.approx(0.12)
    assert entry_map["ep_high"]["adjusted_weight"] >= entry_map["ep_low"]["adjusted_weight"]

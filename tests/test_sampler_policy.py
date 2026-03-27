from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.rl.episode_sampling import DataPackRLSampler
from src.rl.sampler_policy import SAMPLER_POLICY_STRATEGIES
from src.rl.sampler_policy_training import (
    build_sampler_policy_training_dataset,
    train_sampler_policy_models,
)


def _descriptor(
    pack_id: str,
    *,
    tier: int,
    trust: float,
    frontier_gain: float,
    expected_gain: float,
    priority_level: str,
) -> dict:
    return {
        "pack_id": pack_id,
        "env_name": "sampler_env",
        "task_type": "sampler_task",
        "engine_type": "synthetic",
        "backend": "synthetic",
        "objective_vector": [1.0, 1.0, 1.0, 1.0, 0.0],
        "tier": tier,
        "trust_score": trust,
        "sampling_weight": 1.0 + frontier_gain,
        "delta_mpl": frontier_gain,
        "delta_J": frontier_gain * 0.8,
        "episode_length": 24,
        "w_embodiment": 1.0,
        "embodiment_drift_score": 0.1,
        "w_epi": frontier_gain * 0.5,
        "inferential_replay_weight": 0.6 + 0.2 * frontier_gain,
        "unified_quality_weight": 1.0,
        "unified_quality_eligible": True,
        "enrichment": {
            "novelty_tags": [
                {
                    "novelty_score": min(1.0, frontier_gain / 2.0),
                    "expected_mpl_gain": expected_gain,
                }
            ],
            "supervision_hints": {
                "priority_level": priority_level,
                "suggested_weight_multiplier": 1.0,
            },
            "coherence_score": 0.8,
        },
    }


def _receipt() -> dict:
    sampler = DataPackRLSampler(
        existing_descriptors=[
            _descriptor(
                "ep_high",
                tier=2,
                trust=0.92,
                frontier_gain=1.8,
                expected_gain=7.5,
                priority_level="critical",
            ),
            _descriptor(
                "ep_mid",
                tier=1,
                trust=0.65,
                frontier_gain=0.7,
                expected_gain=2.5,
                priority_level="high",
            ),
            _descriptor(
                "ep_low",
                tier=0,
                trust=0.35,
                frontier_gain=0.1,
                expected_gain=0.2,
                priority_level="low",
            ),
        ],
        default_strategy="balanced",
    )
    sampler.sample_batch(batch_size=3, seed=3, strategy="balanced")
    receipt = dict(sampler.last_sampler_policy_artifact or {})
    receipt["strategy_targets"] = {
        strategy: (1.0 if strategy == "frontier_prioritized" else 0.0)
        for strategy in SAMPLER_POLICY_STRATEGIES
    }
    receipt["sampling_plan_targets"] = {
        "frontier_threshold_quantile": 0.45,
        "frontier_focus_ratio": 0.85,
        "econ_threshold_quantile": 0.55,
        "econ_focus_ratio": 0.60,
    }
    for entry in receipt["episode_entries"]:
        episode_id = entry["episode_id"]
        entry["strategy_weight_targets"]["frontier_prioritized"] = 1.0 if episode_id == "ep_high" else 0.15
        entry["strategy_weight_targets"]["balanced"] = 0.4 if episode_id == "ep_high" else 0.3
    return receipt


def _write_package(tmp_path: Path) -> Path:
    receipt = _receipt()
    dataset = build_sampler_policy_training_dataset([receipt] * 24)
    checkpoint_path = tmp_path / "sampler_policy.pt"
    _, _, training_result = train_sampler_policy_models(
        dataset,
        epochs=80,
        hidden_dim=16,
        save_path=str(checkpoint_path),
    )
    assert training_result["checkpoint_path"] is not None
    package_path = tmp_path / "sampler_policy_package.json"
    package_path.write_text(
        json.dumps(
            {
                "package_id": "sampler_policy_test",
                "checkpoint_path": checkpoint_path.name,
                "model_config": {
                    "pool_input_dim": len(receipt["pool_feature_map"]),
                    "episode_input_dim": len(receipt["episode_entries"][0]["feature_map"]) + len(SAMPLER_POLICY_STRATEGIES),
                    "hidden_dim": 16,
                },
                "benchmark_gate": {"ready": True},
                "execution_preconditions": {"benchmark_gate_ready": True},
                "promotion_stage": "promoted",
                "inference_contract": {
                    "helper_blend_policy": {
                        "shadow_candidate_strategy_weight": 0.12,
                        "promoted_strategy_weight": 0.80,
                        "shadow_candidate_episode_weight": 0.12,
                        "promoted_episode_weight": 0.80,
                        "shadow_candidate_plan_weight": 0.12,
                        "promoted_plan_weight": 0.80,
                    }
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return package_path


def test_sampler_policy_helper_updates_strategy_and_weight_trace(tmp_path: Path) -> None:
    package_path = _write_package(tmp_path)
    sampler = DataPackRLSampler(
        existing_descriptors=[
            _descriptor(
                "ep_high",
                tier=2,
                trust=0.92,
                frontier_gain=1.8,
                expected_gain=7.5,
                priority_level="critical",
            ),
            _descriptor(
                "ep_mid",
                tier=1,
                trust=0.65,
                frontier_gain=0.7,
                expected_gain=2.5,
                priority_level="high",
            ),
            _descriptor(
                "ep_low",
                tier=0,
                trust=0.35,
                frontier_gain=0.1,
                expected_gain=0.2,
                priority_level="low",
            ),
        ],
        default_strategy="balanced",
        sampler_policy_helper_mode="auto",
        sampler_policy_package_path=str(package_path),
    )

    batch = sampler.sample_batch(batch_size=3, seed=7)

    trace = batch[0]["sampling_metadata"]["sampler_strategy_trace"]
    assert trace["strategy_source"] == "heuristic_plus_learned_helper"
    assert trace["final_strategy"] == "frontier_prioritized"
    assert batch[0]["sampling_metadata"]["sampler_policy"]["weight_source"] == "heuristic_plus_learned_helper"
    assert sampler.last_sampler_policy_artifact is not None
    assert sampler.last_sampler_policy_artifact["strategy_targets"]["frontier_prioritized"] > 0.0

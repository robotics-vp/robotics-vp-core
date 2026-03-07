from src.orchestrator.queue_selection import (
    QueueDispatchConfig,
    apply_live_queue_selection,
)
from src.rl.episode_sampling import DataPackRLSampler


def _descriptor(pack_id: str, weight: float) -> dict:
    return {
        "pack_id": pack_id,
        "env_name": "shadow_env",
        "task_type": "shadow_task",
        "engine_type": "synthetic",
        "backend": "synthetic",
        "objective_vector": [1.0, 1.0, 1.0, 1.0, 0.0],
        "tier": 1,
        "trust_score": 0.8,
        "sampling_weight": weight,
        "episode_length": 10,
    }


def test_apply_live_queue_selection_bounded_reweight_and_drop():
    episodes = [
        {"descriptor": _descriptor("ep_low", 1.0)},
        {"descriptor": _descriptor("ep_high", 1.0)},
    ]
    payload = {
        "queue_name": "shadow_advisory_queue",
        "entries": [
            {
                "episode_id": "ep_high",
                "priority_score": 0.95,
                "replay_action": "upweight",
                "tags": ["high_value_uncertain", "frontier_candidate"],
                "metadata": {"deploy_recommendation": "allow_shadow"},
            },
            {
                "episode_id": "ep_low",
                "priority_score": 0.9,
                "replay_action": "downweight",
                "tags": ["low_provenance_review", "downweight_candidate"],
                "metadata": {
                    "deploy_recommendation": "deny_shadow",
                    "promotion_stage": "budget_gate",
                },
            },
        ],
    }
    dispatch = apply_live_queue_selection(
        episodes,
        live_queue_selection=payload,
        config=QueueDispatchConfig(
            mode="promoted_gate_eligible",
            max_upweight=2.0,
            max_downweight=0.5,
            allow_slice_removal_on_integrity_failure=True,
        ),
    )
    assert dispatch["summary"]["num_reweighted"] >= 1
    assert dispatch["summary"]["num_dropped"] == 1
    assert dispatch["ordered_episode_ids"][0] == "ep_high"


def test_sampler_dispatch_queue_consumes_live_queue_selection():
    descriptors = [_descriptor("ep_a", 1.0), _descriptor("ep_b", 1.0), _descriptor("ep_c", 1.0)]
    payload = {
        "queue_name": "shadow_advisory_queue",
        "entries": [
            {"episode_id": "ep_c", "priority_score": 0.9, "replay_action": "upweight", "tags": ["frontier_candidate"]},
            {"episode_id": "ep_a", "priority_score": 0.2, "replay_action": "holdout", "tags": ["holdout_candidate"]},
        ],
    }
    sampler = DataPackRLSampler(
        existing_descriptors=descriptors,
        live_queue_selection=payload,
        queue_dispatch_mode="bounded_reweight",
    )
    dispatch = sampler.dispatch_queue(batch_size=3, seed=0, strategy="balanced")
    assert dispatch["mode"] == "bounded_reweight"
    assert dispatch["ordered_episode_ids"][0] == "ep_c"
    assert sampler.last_queue_dispatch_artifact is not None

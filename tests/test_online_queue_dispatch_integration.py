import numpy as np
import torch

from src.rl.episode_sampling import DataPackRLSampler
from src.rl.sac import SACAgent


class _DummyEncoder(torch.nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(obs_dim, obs_dim)
        self.use_consistency = False
        self.use_contrastive = False

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(obs)


def _descriptor(pack_id: str, weight: float = 1.0) -> dict:
    return {
        "pack_id": pack_id,
        "env_name": "online_shadow_env",
        "task_type": "online_shadow_task",
        "engine_type": "synthetic",
        "backend": "synthetic",
        "objective_vector": [1.0, 1.0, 1.0, 1.0, 0.0],
        "tier": 1,
        "trust_score": 0.8,
        "sampling_weight": weight,
        "episode_length": 10,
    }


def test_online_sampler_and_replay_buffer_apply_bounded_queue_influence(tmp_path):
    np.random.seed(0)
    torch.manual_seed(0)

    payload = {
        "queue_name": "online_sac_queue",
        "entries": [
            {
                "episode_id": "ep_high",
                "priority_score": 0.95,
                "replay_action": "upweight",
                "tags": ["frontier_candidate", "high_value_uncertain"],
                "metadata": {
                    "promotion_stage": "advisory",
                    "influence_source": "heuristic",
                    "evidence": {"reason": "recent_success"},
                },
            },
            {
                "episode_id": "ep_low",
                "priority_score": 0.15,
                "replay_action": "downweight",
                "tags": ["downweight_candidate"],
                "metadata": {
                    "promotion_stage": "compare_only",
                    "influence_source": "heuristic",
                    "evidence": {"reason": "recent_failure"},
                },
            },
        ],
    }

    sampler = DataPackRLSampler(
        existing_descriptors=[
            _descriptor("ep_low", 1.0),
            _descriptor("ep_high", 1.0),
            _descriptor("ep_mid", 1.0),
        ],
        live_queue_selection=payload,
        queue_dispatch_mode="bounded_reweight",
    )
    batch = sampler.sample_batch(batch_size=3, seed=0, strategy="balanced")
    assert batch[0]["pack_id"] == "ep_high"
    assert batch[0]["sampling_metadata"]["queue_dispatch"]["decision"]["reweight_factor"] > 1.0

    dispatch = sampler.dispatch_queue(batch_size=3, seed=0, strategy="balanced")
    assert dispatch["original_queue_order"][0] != dispatch["adjusted_queue_order"][0]
    assert dispatch["entries"][0]["promotion_stage"] == "advisory"
    assert dispatch["entries"][0]["influence_source"] == "heuristic"

    agent = SACAgent(
        encoder=_DummyEncoder(obs_dim=4),
        latent_dim=4,
        action_dim=2,
        batch_size=1,
        device="cpu",
        sampling_artifact_dir=str(tmp_path),
    )
    for idx in range(8):
        obs = np.array([idx, idx + 1, idx + 2, idx + 3], dtype=np.float32)
        agent.store_transition(
            obs,
            np.array([0.1, 0.2], dtype=np.float32),
            0.5,
            obs + 0.1,
            False,
            novelty=1.0,
            episode_id="ep_high",
        )
    for idx in range(8, 16):
        obs = np.array([idx, idx + 1, idx + 2, idx + 3], dtype=np.float32)
        agent.store_transition(
            obs,
            np.array([0.1, 0.2], dtype=np.float32),
            0.1,
            obs + 0.1,
            False,
            novelty=1.0,
            episode_id="ep_low",
        )
    agent.apply_queue_dispatch(dispatch)

    counts = {"ep_high": 0, "ep_low": 0}
    for _ in range(128):
        *_, sampled_metadata = agent.replay_buffer.sample(1, return_metadata=True)
        counts[sampled_metadata[0]["episode_id"]] += 1

    assert counts["ep_high"] > counts["ep_low"]
    artifact = agent.get_last_sampling_artifact()
    assert artifact is not None
    assert artifact["original_queue_order"]
    assert artifact["adjusted_queue_order"]
    assert artifact["reweight_factors"]["ep_high"] > artifact["reweight_factors"]["ep_low"]

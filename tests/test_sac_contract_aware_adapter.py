import numpy as np
import torch

from src.rl.sac import SACAgent
from src.rl.sac_contract_aware_adapter import (
    SACContractAwareAdapter,
    SACContractAwareAdapterConfig,
)


class _DummyEncoder(torch.nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(obs_dim, obs_dim)
        self.use_consistency = False
        self.use_contrastive = False

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(obs)


def test_sac_contract_aware_adapter_updates_from_batch(tmp_path):
    adapter = SACContractAwareAdapter(
        SACContractAwareAdapterConfig(
            enabled=True,
            latent_dim=4,
            action_dim=2,
            condition_dim=1,
            artifact_dir=str(tmp_path),
            log_interval=1,
        )
    )
    metrics = adapter.update_from_batch(
        latent_batch=np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.0, 0.3]], dtype=np.float32),
        action_batch=np.array([[0.1, 0.2], [0.2, 0.1]], dtype=np.float32),
        reward_batch=np.array([0.5, 0.2], dtype=np.float32),
        done_batch=np.array([0.0, 1.0], dtype=np.float32),
    )
    assert metrics["enabled"] is True
    assert metrics["total_loss"] >= 0.0
    assert (tmp_path / "sac_contract_aware_metrics.jsonl").exists()


def test_sac_agent_optionally_calls_contract_aware_adapter():
    encoder = _DummyEncoder(obs_dim=4)
    adapter = SACContractAwareAdapter(
        SACContractAwareAdapterConfig(
            enabled=True,
            latent_dim=4,
            action_dim=2,
            condition_dim=1,
        )
    )
    agent = SACAgent(
        encoder=encoder,
        latent_dim=4,
        action_dim=2,
        batch_size=2,
        device="cpu",
        contract_aware_adapter=adapter,
    )
    for idx in range(2):
        obs = np.array([idx, idx + 1, idx + 2, idx + 3], dtype=np.float32)
        next_obs = obs + 0.1
        action = np.array([0.2, 0.3], dtype=np.float32)
        agent.store_transition(obs, action, 0.5, next_obs, False, novelty=1.0)

    metrics = agent.update()
    assert metrics["contract_aware_enabled"] is True
    assert "contract_aware_total_loss" in metrics

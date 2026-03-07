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


def test_sac_contract_aware_live_mode_adds_alignment_loss_and_artifacts(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)

    adapter = SACContractAwareAdapter(
        SACContractAwareAdapterConfig(
            enabled=True,
            mode="live_loss",
            latent_dim=4,
            action_dim=2,
            condition_dim=2,
            artifact_dir=str(tmp_path),
            log_interval=1,
        )
    )
    agent = SACAgent(
        encoder=_DummyEncoder(obs_dim=4),
        latent_dim=4,
        action_dim=2,
        batch_size=2,
        device="cpu",
        contract_aware_adapter=adapter,
    )
    for idx in range(2):
        obs = np.array([idx, idx + 1, idx + 2, idx + 3], dtype=np.float32)
        next_obs = obs + 0.1
        agent.store_transition(
            obs,
            np.array([0.2, 0.3], dtype=np.float32),
            0.5,
            next_obs,
            False,
            novelty=1.0,
            episode_id=f"ep_{idx}",
            condition_vector=np.array([0.1, 0.2], dtype=np.float32),
            skill_mode="efficiency_throughput",
        )

    metrics = agent.update()
    assert metrics["contract_aware_enabled"] is True
    assert metrics["contract_aware_live_mode_enabled"] is True
    assert metrics["contract_aware_critic_alignment_loss"] >= 0.0
    assert metrics["contract_aware_reference_scalar_mae"] >= 0.0
    assert (tmp_path / "sac_contract_aware_metrics.jsonl").exists()
    assert (tmp_path / "sac_contract_aware_predictions.jsonl").exists()

import torch

from src.rl.contract_aware_critic import CriticBundleConfig, ContractAwareCriticBundle
from src.rl.contract_aware_losses import contract_aware_losses


def test_contract_aware_critic_shapes_and_losses():
    config = CriticBundleConfig(
        obs_dim=4,
        action_dim=2,
        condition_dim=3,
        skill_modes=["efficiency_throughput"],
        hidden_dim=32,
        head_hidden_dim=16,
    )
    critic = ContractAwareCriticBundle(config)
    obs = torch.randn(5, 4)
    action = torch.randn(5, 2)
    condition = torch.randn(5, 3)

    outputs = critic(obs, action, condition)
    assert outputs.objective_vector.shape == (5, len(config.objective_axes))
    assert outputs.econ_vector.shape == (5, len(config.econ_axes))
    assert outputs.compiled_scalar.shape == (5,)

    scalar_targets = torch.randn(5)
    objective_targets = torch.randn(5, len(config.objective_axes))
    econ_targets = torch.randn(5, len(config.econ_axes))
    losses = contract_aware_losses(
        outputs=outputs,
        scalar_targets=scalar_targets,
        objective_targets=objective_targets,
        econ_targets=econ_targets,
    )
    assert losses["total_loss"].item() >= 0.0
    assert set(losses.keys()) >= {"scalar_loss", "objective_loss", "econ_loss", "consistency_loss", "total_loss"}

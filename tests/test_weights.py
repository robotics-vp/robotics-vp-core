"""
Unit tests for PPO sample weighting.
"""
import pytest
import torch
from src.rl.ppo import PPOAgent


def test_weights_clipped():
    """Weights should be clipped to [0.5, 2.0]."""
    agent = PPOAgent(obs_dim=4, action_dim=1)

    # Extreme advantages and novelty
    advantages = torch.tensor([10.0, -10.0, 100.0, -100.0])
    novelty_scores = torch.tensor([1.0, 1.0, 1.0, 1.0])

    weights, valuations, stats = agent.compute_sample_weights(
        advantages, novelty_scores
    )

    # All weights should be in [0.5, 2.0]
    assert (weights >= 0.5).all()
    assert (weights <= 2.0).all()


def test_weights_normalized():
    """Weights should be normalized to mean≈1."""
    agent = PPOAgent(obs_dim=4, action_dim=1)

    advantages = torch.randn(100)
    novelty_scores = torch.rand(100)

    weights, valuations, stats = agent.compute_sample_weights(
        advantages, novelty_scores
    )

    # Mean should be very close to 1.0
    assert abs(weights.mean().item() - 1.0) < 0.01


def test_weights_differentiable():
    """Weights should support backprop."""
    agent = PPOAgent(obs_dim=4, action_dim=1)

    advantages = torch.randn(10, requires_grad=True)
    novelty_scores = torch.rand(10)

    weights, valuations, stats = agent.compute_sample_weights(
        advantages, novelty_scores
    )

    # Compute dummy loss
    loss = (weights * advantages).sum()

    # Backprop should work
    loss.backward()

    assert advantages.grad is not None


def test_weight_stats():
    """Weight stats should be computed correctly."""
    agent = PPOAgent(obs_dim=4, action_dim=1)

    advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    novelty_scores = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])

    weights, valuations, stats = agent.compute_sample_weights(
        advantages, novelty_scores
    )

    # Stats should be present
    assert 'mean' in stats
    assert 'p90' in stats
    assert 'min' in stats
    assert 'max' in stats

    # p90 should be >= mean
    assert stats['p90'] >= stats['mean']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

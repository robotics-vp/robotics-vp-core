# Phase B: Stable World Model Frozen

**Date Frozen**: 2025-11-15
**Checkpoint**: `checkpoints/world_model_stable_canonical.pt`

## Architecture: Contractive Residual Dynamics

```python
z_{t+1} = z_t + alpha * g(z_t, a_t)
```

**Key Properties**:
- **Residual structure**: Updates bounded by damping factor
- **Learned alpha**: 0.217 (22.7% of raw delta applied)
- **Spectral normalization**: All layers in g() have bounded singular values
- **Max delta clamp**: 0.15 (hard limit on residual magnitude)
- **Lipschitz bound**: 1.60

**Network**:
- Input: (z_t, a_t) = (128-dim latent, 2-dim action)
- Hidden: 256-dim, 3 layers
- Output: 128-dim residual delta
- LayerNorm + SiLU activations
- Spectral norm on all linear layers

## Training Configuration

```yaml
epochs: 200
lr: 1e-4
weight_decay: 1e-5

# Loss weights
lambda_trust: 1.0
lambda_var: 1.0
lambda_growth: 2.0

# Scheduled sampling
tf_start: 0.8
tf_end: 0.0

# Horizon
max_horizon: 60
```

## Final Metrics (60-step rollouts)

| Metric | Value | Target |
|--------|-------|--------|
| Trust score | 0.999997 | > 0.5 |
| Std ratio | 0.986x | 0.8x - 1.2x |
| Variance growth | 1.03x | < 1.5x |
| Episodes passing trust > 0.5 | 50/50 | 100% |
| Episodes passing trust > 0.9 | 50/50 | 100% |

## Horizon Stability Profile

| Horizon | Trust | Std Ratio | Var Growth |
|---------|-------|-----------|------------|
| H=5 | 0.999984 | 0.968x | 0.93x |
| H=10 | 0.999992 | 0.960x | 0.94x |
| H=20 | 0.999995 | 0.978x | 1.05x |
| H=40 | 0.999997 | 0.982x | 1.01x |
| H=60 | 0.999997 | 0.986x | 1.03x |

**All horizons pass trust > 0.9 for all 50 test episodes.**

## Comparison to Previous Models

| Model | Trust @ H=60 | Std Ratio | Issue |
|-------|--------------|-----------|-------|
| Trust-Aware | 0.000001 | 3.3x | Explodes |
| Horizon-Agnostic | 0.999991 | 0.58x | Mode collapse |
| **Stable (This)** | **0.999997** | **0.99x** | **Stable** |

## Files Associated

- `checkpoints/world_model_stable_canonical.pt` - Frozen checkpoint
- `src/world_model/contractive_dynamics.py` - Architecture definition
- `scripts/train_stable_world_model.py` - Training script
- `scripts/eval_world_model_rollouts.py` - Multi-horizon evaluator
- `results/stable_world_model.json` - Training results
- `results/world_model_rollout_evaluation.json` - Evaluation results
- `results/stable_wm_training_log.txt` - Full training log
- `results/world_model_comparison.md` - Detailed comparison

## Why This Works

1. **Contractive residual**: Can't have eigenvalues > 1
2. **Real scheduled sampling**: Actually mixes teacher forcing (old code ignored parameter)
3. **Multi-horizon loss**: Trained to be accurate at H=1 to H=60
4. **Multi-horizon trust**: Scored at H=10, 30, 60 (not just max)
5. **Variance growth penalty**: Detects instability patterns early

## Usage

**Loading**:
```python
from src.world_model.contractive_dynamics import StableWorldModel
import torch

ckpt = torch.load('checkpoints/world_model_stable_canonical.pt')
model = StableWorldModel(
    latent_dim=128,
    action_dim=2,
    hidden_dim=256,
    n_layers=3,
    alpha_init=0.3,
    max_delta=0.15,
)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

**Single step**:
```python
z_next, delta = model(z_t.unsqueeze(0), a_t.unsqueeze(0))
```

**Multi-step rollout**:
```python
z_trajectory = model.rollout(z_init, actions)  # (T+1, latent_dim)
```

## Safe Usage Guidelines

1. **Local branches only** (for now): Roll 5-10 steps from real z_t
2. **Gate with trust**: Keep only segments with min trust > 0.9
3. **Check variance**: Keep only segments with std ratio in [0.8, 1.2]
4. **Economic weighting**: Use ΔMPL/error/energy to weight importance
5. **A/B test**: Always compare augmented vs baseline before trusting uplift

## DO NOT

- Use for full 60-step episode generation without A/B validation
- Train policy directly on unfiltered synthetic data
- Trust synthetic without trust_net + variance gating
- Modify this checkpoint - create new experiments separately

## Status

**FROZEN** - This is canonical Phase B world model infrastructure.

Same as z_V encoder freeze: if something breaks later, this is the anchor point.

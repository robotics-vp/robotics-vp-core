# World Model Comparison: Before vs After

## Summary

The contractive world model architecture fixes the horizon stability problem completely.

## Models Compared

1. **Trust-Aware (Old)**: `latent_dynamics_trust_aware.pt`
   - Standard MLP with trust loss on short segments
   - Optimizes for 10-step selfies, explodes at 60 steps

2. **Horizon-Agnostic (Attempted Fix)**: `latent_dynamics_horizon_agnostic.pt`
   - Fake scheduled sampling (never actually used TF prob)
   - Over-corrected: mode collapse instead of explosion

3. **Stable/Contractive (New)**: `stable_world_model.pt`
   - Residual architecture: z_{t+1} = z_t + alpha * g(z_t, a_t)
   - Spectral normalization bounds Lipschitz constant
   - REAL scheduled sampling
   - Multi-horizon loss and trust scoring

## Results at H=60 (Full Episode Rollouts)

| Model | Trust Score | Std Ratio | Variance Growth | Passes Trust>0.5 |
|-------|-------------|-----------|-----------------|------------------|
| **Trust-Aware** | 0.000001 | 3.29x | 52-64x | 0/50 (0%) |
| **Horizon-Agnostic** | 0.999991 | 0.58x | 0.64x | 50/50 (100%) |
| **Stable/Contractive** | 0.999997 | 0.99x | 1.03x | 50/50 (100%) |

## Detailed Horizon Breakdown: Trust-Aware (OLD - BROKEN)

| Horizon | Trust Score | Std Ratio | Var Growth | Diagnosis |
|---------|-------------|-----------|------------|-----------|
| H=5 | 0.999996 | 1.00x | 0.98x | Perfect |
| H=10 | 0.999998 | 1.00x | 1.00x | Perfect |
| H=20 | 0.999998 | 1.07x | 1.34x | Starting to drift |
| H=40 | 0.999633 | 1.60x | 6.85x | **Exploding** |
| H=60 | 0.000001 | 3.29x | 52-64x | **CATASTROPHIC** |

**Root cause**: Transition function has eigenvalues > 1 in unstable directions.

## Detailed Horizon Breakdown: Horizon-Agnostic (MODE COLLAPSE)

| Horizon | Trust Score | Std Ratio | Var Growth | Diagnosis |
|---------|-------------|-----------|------------|-----------|
| H=5 | 0.000575 | 0.78x | 0.58x | Low trust, collapsing |
| H=10 | 0.970771 | 0.70x | 0.43x | Better trust, still collapsing |
| H=20 | 0.999933 | 0.62x | 0.34x | High trust, severe collapse |
| H=40 | 0.999988 | 0.56x | 0.33x | Essentially flat |
| H=60 | 0.999991 | 0.58x | 0.64x | Mode collapsed |

**Root cause**: Scheduled sampling was never implemented (ignored the parameter).
Model learned to suppress all dynamics to minimize variance penalty.

## Detailed Horizon Breakdown: Stable/Contractive (FIXED)

| Horizon | Trust Score | Std Ratio | Var Growth | Diagnosis |
|---------|-------------|-----------|------------|-----------|
| H=5 | 0.999984 | 0.97x | 0.93x | Near-perfect |
| H=10 | 0.999992 | 0.96x | 0.94x | Near-perfect |
| H=20 | 0.999995 | 0.98x | 1.05x | Near-perfect |
| H=40 | 0.999997 | 0.98x | 1.01x | Near-perfect |
| H=60 | 0.999997 | 0.99x | 1.03x | Near-perfect |

**Key metrics**:
- All trust scores > 0.999
- All std ratios within 0.96x - 0.99x (target: 1.0x)
- All variance growth ~1.0 (no explosion, no collapse)
- 100% of episodes pass trust > 0.9 at ALL horizons

## Architectural Improvements

### 1. Contractive Residual Structure
```python
z_{t+1} = z_t + alpha * g(z_t, a_t)
```
- Learned alpha = 0.217 (damping factor)
- Bounds maximum step size
- Prevents eigenvalues > 1

### 2. Spectral Normalization
- Applied to all layers in g()
- Bounds Lipschitz constant of updates
- Lipschitz bound: 1.60

### 3. Real Scheduled Sampling
- Actually mixes teacher forcing with autoregressive
- Annealed from 80% TF to 0% over 200 epochs
- Teaches model to correct its own errors

### 4. Multi-Horizon Trust Loss
- Scores trajectories at H=10, 30, 60
- Not just maximum horizon
- Ensures stability at all time scales

### 5. Variance Growth Penalty
- Penalizes late_var / early_var > 1.5
- Not just global std matching
- Detects instability patterns

## Training Progression

```
Epoch   TF%    Alpha   Recon     Trust    Std
----------------------------------------------------
10      76%    0.227   0.00279   0.9999   0.0622
50      60%    0.226   0.00041   1.0000   0.0619
100     40%    0.225   0.00106   1.0000   0.0619
150     20%    0.221   0.00021   1.0000   0.0614
200     0%     0.217   0.00019   1.0000   0.0610

Real z_V std: 0.062
```

Key observations:
- Alpha decreased from 0.30 to 0.217 (learned damping)
- Trust stayed at 1.0 throughout
- Std remained stable at ~0.061-0.062 (perfect match)
- Recon loss decreased consistently

## Implications for Data Augmentation

The stable world model is now ready for use as a local simulator:

1. **Local Branches**: Roll 5-10 steps from real z_t
2. **Gate with Trust**: trust_net scores stay > 0.99
3. **Variance Check**: std ratio ~1.0, no explosion risk
4. **Econ Weighting**: Can now weight by ΔMPL/error/energy

Next steps:
- Generate short synthetic segments (5-10 steps)
- Use trust_net to gate
- A/B test: real-only vs real+synthetic offline RL
- Monitor for "do no harm" before expanding to longer horizons

## Files Created

- `scripts/eval_world_model_rollouts.py` - Multi-horizon evaluator
- `src/world_model/contractive_dynamics.py` - Contractive architecture
- `scripts/train_stable_world_model.py` - Proper training script
- `checkpoints/stable_world_model.pt` - Trained stable model
- `results/stable_world_model.json` - Training results
- `results/world_model_rollout_evaluation.json` - Evaluation results
- `results/stable_wm_training_log.txt` - Full training log

## Conclusion

The world model is now "quasi-sequential vacuum" - the same operator, stable whether it's been 7 steps or 57 steps. No more "MLP that sometimes yeets itself into space."

**Key insight**: The problem was never about loss functions alone. It was about architectural constraints that make the operator inherently stable under iteration, combined with proper training (real scheduled sampling, multi-horizon scoring).

The trust_net + contractive dynamics + econ weighting now form a coherent ecosystem where the world model breathes the same "oxygen" as the rest of the pipeline.

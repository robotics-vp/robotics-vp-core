# Phase B Status: Data Augmentation via World Model

**Date**: 2025-11-15
**Status**: Infrastructure Complete, World Model Blocked

## Executive Summary

Phase B aimed to use a learned world model for synthetic data augmentation in offline RL. The infrastructure is complete and working correctly, but the world model generates out-of-distribution (OOD) latents that poison policy training. A learned trust gate (`trust_net`) successfully rejects this bad data, proving the safety mechanism works. The world model is the blocker.

## What Works

### 1. z_V Encoder and Rollout Infrastructure
- Contrastive encoder trained on real episodes
- Physics-grounded z_V rollout generation
- 50 real episodes collected with economic metrics
- Data properly stored in NPZ format

### 2. Data Bricks and Valuation
- Semantic clustering of episodes into "bricks"
- Economic impact profiles computed per brick (MPL, error rate, energy)
- Brick manifest with episode-level metadata
- Valuations correctly reflect physics-based improvements

### 3. Offline RL Pipeline
- SAC training on z_V space with latent actions
- Replay buffer with source tracking (real/synthetic)
- Weighted sampling and importance-weighted loss support
- Baseline achieves reasonable performance (Action MSE: 0.178)

### 4. Trust Gating (trust_net)
- MLP classifier: episode features → p(real)
- 100% validation accuracy, ROC-AUC = 1.0
- Trust gap: 0.9985 (real=0.9992, synthetic=0.0006)
- Successfully gates synthetic data to 0.06% effective contribution
- Trust-weighted policy matches baseline within 4% (proves "do no harm")

## What's Blocked: World Model OOD Problem

### Root Cause
The world model (GRU-based latent dynamics) generates synthetic latents with:
- **Variance 2x too high**: Real z_V std = 0.062, Synthetic z_V std = 0.117
- ALL 50 synthetic episodes fail basic distribution bounds
- MSE loss looks fine, but latents are OOD in aggregate

### Evidence
```
Real z_V statistics:
  Mean: -0.039 ± 0.067
  Std:  0.062 ± 0.0054 (tight band)
  Range: [-0.32, 0.30]

Synthetic z_V statistics:
  Mean: -0.045 ± 0.071 (close to real)
  Std:  0.117 ± 0.0148 (1.9x real!)
  Range: [-0.69, 0.65] (2x real range)
```

### Impact
- Hardcoded synthetic weighting (10%) poisoned policy training
- All augmented policies performed WORSE than baseline:
  - Baseline: MPL=18.5-20.0, Error=37-43%
  - All augmented: MPL=10-15, Error=47-77%
- trust_net correctly identifies and rejects all synthetic data

## Phase B Artifacts

### Code
- `src/valuation/trust_net.py` - Trust network classifier
- `scripts/train_trust_net.py` - Training and scoring pipeline
- `scripts/train_trust_weighted_offline.py` - Importance-weighted RL
- `scripts/filter_synthetic_zv.py` - Distribution filter (bandaid)
- `src/world_model/` - GRU dynamics model (needs fixing)

### Checkpoints
- `checkpoints/trust_net.pt` - Trained trust classifier
- `checkpoints/offline_baseline_actor.pt` - Real-only baseline
- `checkpoints/offline_trust_weighted_actor.pt` - Trust-gated policy

### Data
- `data/physics_zv_rollouts_trust.npz` - Real episodes with trust scores
- `data/synthetic_zv_rollouts_trust.npz` - Synthetic episodes with trust scores
- `data/bricks/data_bricks_manifest.json` - Brick metadata

### Results
- `results/trust_net_analysis.json` - Trust network metrics
- `results/trust_weighted_training.json` - Trust-weighted RL results

## Key Metrics

| Metric | Baseline | Trust-Weighted | Delta |
|--------|----------|----------------|-------|
| Action MSE | 0.178 | 0.171 | -4.13% |
| Real Contribution | 100% | 99.94% | -0.06% |
| Synthetic Contribution | 0% | 0.06% | - |
| Final Actor Loss | -4.35 | -4.28 | +1.6% |

**Conclusion**: Trust-weighted ≈ baseline confirms the gate works correctly.

## Experiment: Trust-Aware World Model Training

We attempted to fix the world model by using trust_net as a differentiable critic in the training loss:

```python
L_total = L_recon + λ_trust * L_trust + λ_feat * L_feat
```

Where:
- `L_trust = BCE(trust_net(z_syn), 1)` - make synthetic look real
- `L_feat = ||mean(feat_real) - mean(feat_syn)||^2` - feature matching

**Training Results (10-step rollouts):**
- Synthetic trust: 0.427 → 0.9999 (+134%)
- Synthetic std: 0.087 → 0.062 (matches real!)
- Looked promising during training

**Sampling Results (60-step rollouts):**
- NEW trust: 0.000000 (worse than old 0.000626)
- NEW std: 0.21 (3.4x real, worse than old 1.9x)
- 0/50 episodes pass trust threshold

**Root Cause: Compounding Errors**
The model learned to fool trust_net on short 10-step rollouts, but long 60-step rollouts accumulate errors exponentially. The variance explodes when rolling out for full episodes.

## Next Steps (Priority Order)

### 1. Fix Long-Horizon Rollout Stability (Critical Blocker)
The short-horizon training doesn't transfer to long-horizon sampling. Options:

**Option A: Scheduled Sampling with Long Horizons**
```python
# During training, roll out for full 60 steps
# Use teacher forcing with decreasing probability
for t in range(60):
    if random() < teacher_forcing_prob:
        z_input = z_real[t]  # Use ground truth
    else:
        z_input = z_pred[t]  # Use own predictions
    z_pred[t+1] = model(z_input, a[t])
# Apply trust loss to full 60-step rollout
```

**Option B: Variance Clipping**
```python
# Clip residuals to prevent explosion
delta = self.residual_head(h)
delta = torch.clamp(delta, -0.1, 0.1)  # Hard limit
z_next_pred = z_t + delta
```

**Option C: Autoregressive Training with Real Sequences**
- Train on actual full 60-step sequences from real data
- Backprop through time for entire episode
- More expensive but captures long-horizon dynamics

### 2. Match Training Horizon to Sampling Horizon
- If sampling uses 60 steps, train with 60-step rollouts
- Trust loss must be computed on same horizon as deployment
- Feature matching on full episodes, not short segments

### 3. Add Variance Regularization
```python
# During rollout
syn_var_per_step = z_sequence.var(dim=2).mean(dim=1)  # Per-episode variance over time
# Penalize variance growth
var_growth = syn_var_per_step[-1] / (syn_var_per_step[0] + 1e-6)
L_var = λ_var * ReLU(var_growth - 1.5)  # Penalize if >1.5x growth
```

### 4. Alternative: Shorter Horizons for Augmentation
Instead of generating full 60-step episodes:
- Generate 10-20 step segments where model is stable
- Use these shorter segments for offline RL
- Avoids compounding error problem

## Lessons Learned

1. **MSE is insufficient**: Low MSE doesn't mean in-distribution
2. **Static weighting is dangerous**: 10% synthetic was poisoning training
3. **Learned gates are essential**: trust_net catches what static rules miss
4. **Distribution matters more than point estimates**: Variance was 2x off
5. **Safety first**: Trust gate proved it "does no harm" before trusting augmentation

## Technical Debt

- [ ] World model needs distribution regularization
- [ ] Episode-level trust vs. transition-level trust (currently episode)
- [ ] trust_net uses simple features (mean, std, min, max, var, smoothness)
- [ ] Could extend to temporal patterns with TemporalTrustNet (GRU)
- [ ] No automatic retraining loop for flywheel yet

## Conclusion

Phase B infrastructure is complete and validated. The trust gate correctly rejects OOD synthetic data, proving safety. The blocker is the world model's distribution mismatch. Once fixed with proper regularization, the infrastructure is ready for real data augmentation gains.

**Status**: Ready for world model fix experiment.

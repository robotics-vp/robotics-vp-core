# Synthetic Data Augmentation Roadmap

## Current Status: Phase B World Model Complete

The stable world model (`checkpoints/world_model_stable_canonical.pt`) is now horizon-agnostic:
- Trust > 0.999 at all horizons (5-60 steps)
- Std ratio ~0.99x (perfect distribution match)
- Variance growth ~1.03x (stable, not exploding)

**Ready for conservative augmentation testing.**

## Horizon Expansion Levels

### Level 0: Baseline (Real-Only)
**Status**: ✓ Already done
- Real physics rollouts only
- Trust-weighted training
- Established baseline metrics: Action MSE, MPL, error rate

### Level 1: Ultra-Short Branches (5 steps)
**Status**: Ready to test
- Shortest safe horizon
- Minimal error accumulation
- Expected: Should definitely pass all gates

```bash
# Collect 5-step branches
python scripts/collect_local_synthetic_branches.py \
  --horizon 5 \
  --min-trust 0.95 \
  --branches-per-episode 10 \
  --output data/local_synth_branches_h5.npz

# A/B test
python scripts/train_offline_with_local_synth.py \
  --synth-data data/local_synth_branches_h5.npz \
  --synth-weight 0.05
```

**Pass criteria**:
- All branches pass trust > 0.95
- Augmented MSE within 2% of baseline

### Level 2: Short Branches (10 steps)
**Status**: Ready to test
- Default conservative horizon
- Good balance of diversity vs safety

```bash
# Collect 10-step branches
python scripts/collect_local_synthetic_branches.py \
  --horizon 10 \
  --min-trust 0.9 \
  --branches-per-episode 5 \
  --output data/local_synth_branches_h10.npz

# A/B test
python scripts/train_offline_with_local_synth.py \
  --synth-data data/local_synth_branches_h10.npz \
  --synth-weight 0.1
```

**Pass criteria**:
- >80% branches pass trust > 0.9
- Augmented MSE within 3% of baseline
- No "cliffs" in learning curves

### Level 3: Medium Branches (20 steps)
**Status**: Proceed only if Level 2 passes
- More temporal diversity
- Starting to push stability limits

```bash
# Collect 20-step branches
python scripts/collect_local_synthetic_branches.py \
  --horizon 20 \
  --min-trust 0.85 \
  --branches-per-episode 3 \
  --output data/local_synth_branches_h20.npz

# A/B test
python scripts/train_offline_with_local_synth.py \
  --synth-data data/local_synth_branches_h20.npz \
  --synth-weight 0.1
```

**Pass criteria**:
- >60% branches pass trust > 0.85
- Augmented MSE within 5% of baseline
- Monitor for variance drift in late steps

### Level 4: Long Branches (40+ steps)
**Status**: Only if Level 3 passes and shows improvement
- High-risk, high-reward
- Full episode-level diversity

**Requirements**:
- Previous levels show positive or neutral impact
- May need tighter std ratio bounds [0.9, 1.1]
- Consider brick-specific evaluation

### Level 5: Full Synthetic Episodes (60 steps)
**Status**: Future goal
- Complete episode generation
- Maximum diversity
- Only after all previous levels validated

## Execution Plan

### Step 1: Collect Level 2 Branches (Default)
```bash
python scripts/collect_local_synthetic_branches.py
```

Expected output:
- `data/local_synth_branches.npz` - Branch data
- `data/local_synth_branches_metadata.json` - Stats

### Step 2: Run A/B Test
```bash
python scripts/train_offline_with_local_synth.py
```

Expected output:
- `results/offline_local_synth_eval.json` - Comparison metrics
- `checkpoints/offline_baseline_actor.pt` - Baseline policy
- `checkpoints/offline_local_synth_actor.pt` - Augmented policy

### Step 3: Interpret Results

| Result | Interpretation | Next Action |
|--------|----------------|-------------|
| Augmented ≈ Baseline (±2%) | "Do no harm" - safe but no uplift | Increase synth_weight or horizon |
| Augmented < Baseline (3-5% worse) | Minor degradation | Tighten gating (higher trust, narrower std) |
| Augmented > Baseline (any improvement) | Synthetic adds value! | This is the goal state |
| Augmented << Baseline (>5% worse) | Poisoning | Reduce horizon, increase trust threshold |

### Step 4: Economic Metrics Evaluation

After A/B test, evaluate on physics env:
```python
# TODO: Add physics env evaluation
# - Run both policies in simulation
# - Measure MPL, error rate, energy consumption
# - Compare wage parity metrics
# - Break down by brick/condition
```

This is the real test: does synthetic improve economic metrics, not just MSE?

## Gating Strategy

### Trust Gating
```
min_trust >= threshold
```
- Conservative: 0.95 (very strict)
- Normal: 0.90 (default)
- Aggressive: 0.80 (risky)

### Variance Gating
```
std_ratio in [min_ratio, max_ratio]
```
- Conservative: [0.9, 1.1] (tight)
- Normal: [0.8, 1.2] (default)
- Aggressive: [0.7, 1.3] (risky)

### Economic Weighting
```
weight = trust * econ_weight * source_balance
```

Where `econ_weight` can be derived from:
- Brick MPL value
- ΔMPL contribution
- Error reduction potential
- Energy efficiency
- Novelty score

## Red Flags to Watch For

1. **Trust collapse at horizon N** - Model suddenly unstable
2. **Std ratio drift** - Gradually increasing variance
3. **MSE explosion** - Prediction errors growing fast
4. **Learning curve cliffs** - Sudden performance drops
5. **Brick-specific failures** - Some bricks poison, others don't

## Success Criteria by Phase

### Phase B1: Local Augmentation Works
- Synthetic passes trust/variance gates
- A/B test shows "do no harm"
- Pass rate > 50% for 10-step branches

### Phase B2: Synthetic Creates Value
- A/B test shows improvement (any amount)
- Economic metrics (MPL, error) improve
- Pass rate stable across bricks

### Phase B3: Scale to Longer Horizons
- 20-40 step branches usable
- Consistent economic uplift
- Automatic horizon selection based on brick

### Phase B4: Full Flywheel
- Synthetic feeds back to WM training
- Econ weights learned end-to-end
- Continuous improvement loop

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/eval_world_model_rollouts.py` | Diagnose WM stability |
| `scripts/collect_local_synthetic_branches.py` | Generate trusted branches |
| `scripts/train_offline_with_local_synth.py` | A/B test augmentation |
| `scripts/train_stable_world_model.py` | Train WM if needed |

## Files to Monitor

| File | What to Check |
|------|---------------|
| `data/local_synth_branches_metadata.json` | Pass rate, trust distribution |
| `results/offline_local_synth_eval.json` | A/B comparison metrics |
| `results/world_model_rollout_evaluation.json` | WM stability by horizon |

## Key Insight

The world model is now "quasi-sequential vacuum" - same operator, stable at any horizon. But just because it's stable doesn't mean it's useful. The A/B test with economic weighting is the real validation: does it create value, not just pass technical checks?

Start conservative (Level 2: 10-step branches), validate "do no harm", then carefully expand. If performance degrades at any level, that's your practical stability horizon regardless of what the technical metrics say.

The oxygen metaphor: the world model now breathes the same air as the rest of the ecosystem (trust_net, econ weights, brick values). Use that shared language to make principled decisions about when to trust synthetic data.

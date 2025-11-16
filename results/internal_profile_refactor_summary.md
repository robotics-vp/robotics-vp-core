# Internal Experiment Profile Refactor Summary

## Overview

Created centralized configuration for experimental knobs used in testing/collection scripts. **The actual RL weighting logic (trust_net × w_econ_lattice) remains 100% DL-driven and was NOT modified.**

## What Was Centralized

### A/B Test Knobs (NOT used in model weighting)
- `target_synth_share`: 0.20 (target synthetic contribution for A/B tests)
- `econ_weight_scale`: 1.0 (scale factor for experiments)
- `econ_weight_cap`: 1.0 (safety cap, not used in model)

### Branch Collection Parameters
- `max_branch_horizon`: 20 (rollout horizon)
- `branches_per_episode`: 5
- `min_trust_threshold`: 0.9 (gating threshold)
- `min_std_ratio`: 0.8
- `max_std_ratio`: 1.2

### Objective Conditioning
- `default_objective_vector`: [1.0, 0.7, 0.5, 0.8] (future conditioning)
- `objective_dim`: 4

### Lattice Training Defaults
- `lattice_n_keypoints`: 16
- `lattice_hidden_dim`: 32
- `lattice_n_bricks`: 5
- `lattice_epochs`: 100

### Training Parameters
- `ab_test_epochs`: 100
- `ab_test_batch_size`: 256
- `ab_test_lr`: 1e-3

### Data Paths
- `real_data_path`: data/physics_zv_rollouts.npz
- `synthetic_branches_path`: data/local_synth_branches.npz
- `brick_manifest_path`: data/bricks/data_bricks_manifest.json
- `w_econ_lattice_path`: checkpoints/w_econ_lattice.pt
- `world_model_path`: checkpoints/world_model_stable_canonical.pt
- `trust_net_path`: checkpoints/trust_net.pt

## Scripts Updated

1. **collect_local_synthetic_branches.py**
   - NOTE comment added
   - Imports `get_internal_experiment_profile`
   - Uses profile for: horizon, branches_per_episode, trust threshold, std ratio bounds, objective_dim, data paths
   - Uses `default_objective_vector` from profile for branch tagging

2. **train_offline_with_local_synth.py**
   - NOTE comment added
   - Imports `get_internal_experiment_profile`
   - Uses profile for: data paths, epochs, batch_size, lr, target_synth_share, econ_weight_scale

3. **train_w_econ_lattice.py**
   - NOTE comment added
   - Imports `get_internal_experiment_profile`
   - Uses profile for: n_bricks, epochs, batch_size, lr, n_keypoints, hidden_dim, data paths

## What Was NOT Modified

- **trust_net** - 100% learned, no changes
- **w_econ_lattice model** - 100% learned monotonic calibrators, no changes
- **RL weight calculation** - Still uses: `weight = trust × w_econ`
- **Any model internals** - All neural network architectures unchanged
- **Data weighting logic** - Still uses `compute_sample_weights()` with learned weights

## Confirmation: RL Weighting Logic NOT Modified

The actual weighting in offline RL still uses:
```python
# In compute_sample_weights():
t['weight'] = t['trust'] * base_trust_weight * t['econ_weight'] * econ_weight_scale
```

Where:
- `t['trust']` comes from trust_net (learned)
- `t['econ_weight']` comes from w_econ_lattice (learned)
- `base_trust_weight` and `econ_weight_scale` are experimental knobs (not model parameters)

The profile only provides **test-time experimental knobs**, not model parameters.

## Task-Specific Profiles

The system supports task-specific overrides:
```python
get_internal_experiment_profile("dishwashing")  # Different objective vector
get_internal_experiment_profile("bricklaying")  # Different horizon
```

## Next Steps

- TODO: Migrate to full PolicyProfile after demo
- Profiles can be extended with more task-specific configurations
- Consider adding validation/schema for profile values

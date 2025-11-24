# Stage 6: First Real Model Training Pipeline - Implementation Summary

## Overview

Stage 6 completes the transition from scaffolding to real neural network training. All components are flag-gated, deterministic, JSON-safe, and preserve economics code integrity.

## Deliverables

### 1. Configuration System ✅

**File**: `config/pipeline.yaml`
- Canonical task ID: `drawer_open` (propagates to all pipelines)
- Phase I data generation settings (SIMA-2, ROS→Stage2, Isaac)
- Training component flags (all default to `false` for safety)
- Determinism and safety configurations

**Module**: `src/config/pipeline.py`
- `get_canonical_task()` - Returns canonical task ID
- `get_phase1_config()` - Phase I data generation settings
- `get_training_config(component)` - Component-specific training config
- `is_neural_mode_enabled(component)` - Check neural mode flag
- `get_checkpoint_path(component)` - Checkpoint path resolver
- `get_determinism_config()` - Seed and CUDA determinism settings
- `get_safety_config()` - Gradient clipping and NaN/Inf checks

### 2. Phase I Data Build Pipeline ✅

**Script**: `scripts/run_phase1_data_build.py`

Orchestrates:
1. **SIMA-2 Stress Generation** - Runs `stress_test_sima2_pipeline.py` for canonical task
2. **ROS→Stage2 Pipeline** - Auto-detects ROS logs; falls back silently if absent
3. **Isaac Adapter** - Single rollout with synthetic data (placeholder for real Isaac)
4. **Manifest Builder** - Validates all data paths and writes `results/phase1_manifest.json`

Output:
- `results/sima2_phase1/` - SIMA-2 stress data
- `results/stage2_phase1/` - ROS→Stage2 outputs (if ROS logs detected)
- `results/isaac_phase1/` - Isaac rollout data
- `results/phase1_manifest.json` - Validated data manifest
- `results/phase1_data_build_summary.json` - Pipeline summary

**Smoke Test**: `scripts/smoke_test_phase1_data_build.py`
- Validates pipeline runs end-to-end
- Checks manifest structure and dataset counts
- Verifies determinism with fixed seed

### 3. Vision Backbone Training ✅

**Script**: `scripts/train_vision_backbone_real.py`

Implements:
- **SimCLR-style contrastive loss** (NT-Xent with temperature scaling)
- **Contrastive projection head** (Linear → ReLU → Linear → L2 normalize)
- **Optional reconstruction head** (decodes projections back to embeddings)
- **Data augmentation** (deterministic Gaussian noise with seed control)
- **Model freezing** (sets eval mode and marks frozen in checkpoint)

Architecture:
```
Features [B, D]
    ↓
Augment (noise) → Aug1, Aug2
    ↓
ContrastiveHead(Linear→ReLU→Linear) → z1, z2 (L2 normalized)
    ↓
NT-Xent Loss (positive pairs across augmentations)
    ↓
Optional: ReconstructionHead(z → features)
```

Output:
- `checkpoints/vision_backbone.pt` - Frozen contrastive model

Training Config (from `config/pipeline.yaml`):
```yaml
training:
  vision_backbone:
    use_neural: false  # Set to true to enable
    checkpoint_path: "checkpoints/vision_backbone.pt"
    freeze_after_training: true
```

**Smoke Test**: `scripts/smoke_test_train_vision_backbone_real.py`
- Validates contrastive loss decreases
- Checks frozen flag in checkpoint
- Verifies model state dictionaries

### 4. Spatial RNN Training ✅

**Script**: `scripts/train_spatial_rnn.py` (already implemented)

Implements:
- **Forward dynamics loss** (z_t, a_t → z_{t+1} prediction via ConvGRU)
- **Temporal smoothness regularizer** (L2 on hidden state deltas)
- **Multi-level support** (P3, P4, P5 pyramid levels)

Loss Function:
```
L = α * L_prediction + β * L_smoothness

L_prediction = MSE(h_{t+1}, target_{t+1})
L_smoothness = Σ ||h_{t+1} - h_t||²
```

Output:
- `checkpoints/spatial_rnn/checkpoint_epoch_N.pt`

Training Config:
```yaml
training:
  spatial_rnn:
    use_neural: false  # Set to true to enable
    checkpoint_path: "checkpoints/spatial_rnn.pt"
```

### 5. SIMA-2 Neural Segmenter Training ✅

**Script**: `scripts/train_sima2_segmenter.py` (already implemented)

Implements:
- **U-Net encoder/decoder architecture**
- **Focal loss** (addresses class imbalance) + **Dice loss** (overlap metric)
- **TrustMatrix weighting** (scales loss by segment trust scores)
- **F1 activation threshold** (neural segmenter activates when F1 ≥ 85%)

Architecture:
```
Input [B, 3, H, W]
    ↓
U-Net Encoder (conv blocks with skip connections)
    ↓
U-Net Decoder (upsampling + skip connections)
    ↓
Boundary Head → boundary_logits [B, 1, H, W]
Primitive Head → primitive_logits [B, num_classes, H, W]
```

Loss Function:
```
L_boundary = α * FocalLoss + (1-α) * DiceLoss
L_primitive = CrossEntropyLoss(primitive_logits, primitive_targets)
L_total = λ_boundary * L_boundary + λ_primitive * L_primitive
```

Output:
- `checkpoints/sima2_segmenter/checkpoint_epoch_N.pt`

Training Config:
```yaml
training:
  sima2_segmenter:
    use_neural: false  # Set to true to enable
    checkpoint_path: "checkpoints/sima2_segmenter.pt"
    f1_activation_threshold: 0.85
```

### 6. Hydra Policy Training ✅

**Script**: `scripts/train_hydra_policy.py` (already implemented)

Implements:
- **Multi-objective SAC** (task reward + econ value + energy regularizer)
- **Skill-mode routing** (HydraActor with per-skill heads)
- **ConditionVector wiring** (econ signals → policy conditioning)

Architecture:
```
Observation [B, obs_dim]
    ↓
ConditionVector (econ_slice, skill_mode, novelty_tier, ood_severity)
    ↓
HydraActor Trunk (shared layers)
    ↓
Skill-specific Heads (one per skill_mode)
    ↓
Action [B, action_dim]
```

Reward Components:
```
r_total = w_task * r_task + w_econ * r_econ + w_energy * r_energy

r_task: Task-specific reward (completion, efficiency)
r_econ: Economic value (MPL delta, wage parity)
r_energy: Energy regularizer (penalize high torque/power)
```

Output:
- `checkpoints/hydra_policy_phase1.pt`

Training Config:
```yaml
training:
  hydra_policy:
    use_neural: false  # Set to true to enable
    checkpoint_path: "checkpoints/hydra_policy.pt"
```

### 7. Golden-Path Runner ✅

**Script**: `scripts/run_stage6_train_all.py`

Orchestrates full Stage 6 pipeline:
1. Phase I data build
2. Vision backbone training
3. Spatial RNN training
4. SIMA-2 segmenter training
5. Hydra policy training
6. Write `results/stage6/success.json` marker

Features:
- **Automatic skip detection** (respects `--skip-*` flags)
- **Neural mode routing** (uses real vs stub scripts based on config)
- **Progress tracking** (prints status icons ✓/⚠/✗)
- **Elapsed time tracking** (per-step and total)
- **Dry-run mode** (`--dry-run` prints commands without executing)

Output:
- `results/stage6/stage6_training_results.json` - Full pipeline results
- `results/stage6/success.json` - Success marker (only if all steps pass)
- `checkpoints/vision_backbone.pt`
- `checkpoints/spatial_rnn/checkpoint_epoch_5.pt`
- `checkpoints/sima2_segmenter/checkpoint_epoch_10.pt`
- `checkpoints/hydra_policy_phase1.pt`

### 8. End-to-End Smoke Test ✅

**Script**: `scripts/smoke_test_stage6_end_to_end.py`

Validates:
- Full pipeline runs without errors
- All checkpoints created (for non-skipped steps)
- Success marker written when all steps pass
- Economics code untouched (imports still work)
- Determinism with fixed seed

## Usage

### Basic Usage (All Steps)

```bash
# Run full Stage 6 pipeline with defaults (stub mode)
python scripts/run_stage6_train_all.py --seed 0

# Enable neural mode for specific components (set in config/pipeline.yaml)
# training:
#   vision_backbone:
#     use_neural: true  # Enable real SimCLR training
#   spatial_rnn:
#     use_neural: true  # Enable real ConvGRU training
#   ...

python scripts/run_stage6_train_all.py --seed 0
```

### Skip Specific Steps

```bash
# Skip data build (use existing data)
python scripts/run_stage6_train_all.py --seed 0 --skip-data-build

# Skip heavy training steps for quick test
python scripts/run_stage6_train_all.py --seed 0 \
  --skip-spatial-rnn \
  --skip-segmenter
```

### Dry Run (Print Commands Only)

```bash
python scripts/run_stage6_train_all.py --seed 0 --dry-run
```

### Run Individual Steps

```bash
# Phase I data build
python scripts/run_phase1_data_build.py --seed 0

# Vision backbone (real)
python scripts/train_vision_backbone_real.py --seed 0 --force-neural --epochs 10

# Spatial RNN
python scripts/train_spatial_rnn.py --seed 0 --epochs 5 --mode convgru

# SIMA-2 segmenter
python scripts/train_sima2_segmenter.py --seed 0 --epochs 10 --use_dice

# Hydra policy
python scripts/train_hydra_policy.py --seed 0 --max-steps 100
```

## Smoke Tests

```bash
# Test Phase I data build
python scripts/smoke_test_phase1_data_build.py

# Test vision backbone training
python scripts/smoke_test_train_vision_backbone_real.py

# Test Stage 6 end-to-end
python scripts/smoke_test_stage6_end_to_end.py
```

## Contract Guarantees

### ✅ Determinism
- All scripts seed RNG (numpy, torch, CUDA)
- Same seed → same outputs (within floating-point precision)
- CUDA determinism enforced when enabled in config

### ✅ Flag-Gated
- Neural mode defaults to `false` (safe)
- Must explicitly enable via `config/pipeline.yaml` or `--force-neural`
- Stub versions run when neural mode disabled

### ✅ JSON-Safe
- All outputs serializable to JSON via `to_json_safe()`
- No numpy arrays, torch tensors, or custom objects in output files
- Checksums and digests for validation

### ✅ No Economics Code Modified
- Stage 6 scripts do NOT import or modify:
  - `src/economics/reward_engine.py`
  - `src/economics/econ_controller.py`
  - `src/economics/*` (any economics module)
- Smoke test validates economics imports still work

### ✅ Safety
- Gradient clipping (max_norm=1.0) on all neural training
- NaN/Inf checks (clip tensors, fail on inf if configured)
- Timeout limits on subprocess calls (600s default)

## Next Steps (Post-Stage 6)

After Stage 6 completes, the following are ready for integration:

1. **Real Image Loading** - Replace synthetic image generation in RegNet with actual RGB loading from VisionFrame paths
2. **Full Isaac Integration** - Replace placeholder Isaac rollout with real Isaac Gym rollouts
3. **Neural Mode Defaults** - Once validated, flip `use_neural: true` in config for production
4. **Multi-Task Training** - Extend beyond `drawer_open` to full task distribution
5. **Hyperparameter Tuning** - Grid search over lr, temperature, hidden_dim, etc.
6. **Distributed Training** - Add multi-GPU support for large-scale training

## Files Created

### Configuration
- `config/pipeline.yaml`
- `src/config/pipeline.py`

### Scripts
- `scripts/run_phase1_data_build.py`
- `scripts/train_vision_backbone_real.py`
- `scripts/run_stage6_train_all.py`

### Smoke Tests
- `scripts/smoke_test_phase1_data_build.py`
- `scripts/smoke_test_train_vision_backbone_real.py`
- `scripts/smoke_test_stage6_end_to_end.py`

### Documentation
- `STAGE6_IMPLEMENTATION.md` (this file)

## Verification Checklist

- [x] Task 1: Canonical task config created
- [x] Task 2: Phase I data build pipeline implemented
- [x] Task 3: Vision backbone with real SimCLR loss
- [x] Task 4: Spatial RNN training (already implemented)
- [x] Task 5: SIMA-2 segmenter training (already implemented)
- [x] Task 6: Hydra policy training (already implemented)
- [x] Task 7: Golden-path runner created
- [x] Task 8: End-to-end smoke test created
- [x] All scripts flag-gated (default off)
- [x] All outputs deterministic
- [x] All outputs JSON-safe
- [x] No economics code modified

---

**Status**: ✅ Stage 6 Implementation Complete

All deliverables created, tested, and documented. Ready for real training runs.

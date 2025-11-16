# Video-to-Policy Pipeline: Implementation Status

**Last Updated**: 2025-01-12

---

## ✅ COMPLETED: Step 1 & 2 - Video Infrastructure

### What's Working

**1. Video Encoder Module** (`src/encoders/video_encoder.py`)
- ✅ Simple2DCNN (651K params, CPU-friendly)
- ✅ Simple3DCNN (837K params)
- ✅ R3D18 (requires torchvision + GPU)
- ✅ Unified interface: `(B, T, C, H, W) → (B, latent_dim)`

**2. Encoder Builder** (`src/encoders/builder.py`)
- ✅ Config-driven encoder construction
- ✅ Supports both MLP (state) and Video encoders
- ✅ Device handling (CPU/GPU)

**3. DishwashingVideoEnv Wrapper** (`src/envs/video_wrappers.py`)
- ✅ Wraps DishwashingEnv to emit video observations
- ✅ Synthetic frame generation from state (colored bars)
- ✅ Frame buffer management (temporal stacking)
- ✅ Returns: `(T, C, H, W)` video instead of state vector
- ✅ Tested and validated

**4. Video Configuration** (`configs/dishwashing_video.yaml`)
- ✅ encoder.type = "video"
- ✅ env.type = "dishwashing_video"
- ✅ All video parameters specified

### Testing Results

```bash
$ python3 src/envs/video_wrappers.py

Testing DishwashingVideoEnv...

[Reset Test]
Observation shape: (8, 3, 64, 64)  ✅
Observation dtype: float32          ✅
Observation range: [0.000, 0.941]   ✅

[Step Test]
Observation shape: (8, 3, 64, 64)  ✅
Info keys: ['succ', 'errs', 'p_err', 'speed', 'care', 'rate_per_min']  ✅

✅ All tests passed!
```

---

## 🔄 IN PROGRESS: Step 3 - SAC Integration

### What Needs to Be Done

**Modify `train_sac.py` to support both state and video modes:**

1. **Environment Creation** (based on config)
```python
if cfg['env']['type'] == 'dishwashing':
    # State mode (current)
    env = DishwashingEnv(params)
    obs_dim = env._obs().shape[0]
    encoder = build_encoder(cfg['encoder'], obs_dim=obs_dim, device=device)

elif cfg['env']['type'] == 'dishwashing_video':
    # Video mode (new)
    from src.envs.video_wrappers import create_video_env
    env = create_video_env(
        base_env_class=DishwashingEnv,
        base_env_config={'params': params},
        video_config=cfg['env']['video']
    )
    encoder = build_encoder(cfg['encoder'], video_shape=(8,3,64,64), device=device)
```

2. **Observation Handling in Replay Buffer**
```python
# Current: buffer stores state vectors
# New: buffer should store video observations (8, 3, 64, 64)

# When sampling batch:
if encoder_type == 'video':
    obs_batch = torch.FloatTensor(obs_batch).to(device)  # (B, T, C, H, W)
else:
    obs_batch = torch.FloatTensor(obs_batch).to(device)  # (B, obs_dim)

# Encode to latent
z = encoder(obs_batch)  # (B, latent_dim) for both modes
```

3. **Novelty & ΔMPL on Latents**
```python
# OLD (state-based):
novelty = compute_novelty(state)

# NEW (latent-based):
z = encoder(obs)  # obs is either state or video
novelty = compute_novelty(z)  # operates on latent space

# ΔMPL estimation
delta_mpl_cust = data_value_estimator.predict(novelty)
```

4. **Episode Loop Updates**
```python
# Reset
obs = env.reset()  # (8,3,64,64) in video mode, (obs_dim,) in state mode

# Step
obs_next, info, done = env.step(action)  # Same signature for both modes

# Store in replay
buffer.add(obs, action, reward, obs_next, done)  # Works for both
```

### Key Design Principle

**SAC operates ONLY on latent embeddings `z`, never on raw observations.**

This means:
- Actor network: `π(a | z)` not `π(a | obs)`
- Critic network: `Q(z, a)` not `Q(obs, a)`
- Encoder handles modality: `z = f_ψ(obs)` where obs can be state or video

---

## 📋 TODO: Steps 4-6

### Step 4: Diffusion Novelty on Latents

**Modify**: `src/deep_learning/novelty_diffusion.py`

```python
class DiffusionNoveltyEstimator:
    """Operates on latent embeddings, not raw pixels"""

    def __init__(self, latent_dim=128):
        # Diffusion operates in latent space
        self.latent_dim = latent_dim
        self.prior = LatentDiffusion(latent_dim)

    def compute_novelty(self, z):
        """
        Args:
            z: (latent_dim,) embedding from encoder
        Returns:
            novelty: float
        """
        # Compute reconstruction error in latent space
        ...
```

**Integration** (train_sac.py):
```python
# Encode observation to latent
z = encoder(obs)

# Compute novelty in latent space
novelty = diffusion_novelty.compute_novelty(z)

# Predict ΔMPL from novelty
delta_mpl_cust = data_value_estimator.predict(novelty)

# Use in spread allocation
spread_info = compute_spread_allocation(
    w_hat_r=w_hat_r,
    w_h=wh,
    time_hours=time_hours,
    delta_mpl_cust=delta_mpl_cust,
    delta_mpl_total=delta_mpl_total
)
```

### Step 5: State vs Video Comparison

**Create**: `experiments/compare_state_vs_video.py`

```python
def compare_modalities():
    """
    Run short training with state vs video configs.
    Compare:
    - Final MP, error, wage parity
    - Consumer surplus distribution
    - Training stability (loss variance)
    """

    results = {}

    # Run state mode
    results['state'] = run_training('configs/dishwashing_feasible.yaml', episodes=200)

    # Run video mode
    results['video'] = run_training('configs/dishwashing_video.yaml', episodes=200)

    # Compare metrics
    compare_economics(results['state'], results['video'])
    compare_stability(results['state'], results['video'])

    # Generate report
    save_comparison_report(results, 'experiments/state_vs_video_comparison.json')
```

### Step 6: GPU + Physics Scaffolding

**Create**: `requirements-gpu.txt`
```
torch>=2.0.0+cu118
torchvision>=0.15.0+cu118
diffusers>=0.21.0
```

**Create**: `GPU_README.md` (instructions for RunPod/AWS)

**Create**: `src/envs/physics/__init__.py` (interface stub)

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VIDEO-TO-POLICY PIPELINE                  │
└─────────────────────────────────────────────────────────────┘

[Video Obs]                    [State Obs]
(8, 3, 64, 64)                 (obs_dim,)
     │                              │
     │                              │
     ▼                              ▼
┌──────────────┐            ┌──────────────┐
│ Video Encoder│            │ MLP Encoder  │
│ Simple2DCNN  │            │  (256-256)   │
└──────────────┘            └──────────────┘
     │                              │
     └──────────┬───────────────────┘
                ▼
         [Latent z] (128-dim)
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌────────┐ ┌────────┐ ┌────────────┐
│  SAC   │ │Novelty │ │   ΔMPL     │
│ Policy │ │Estimate│ │ Estimator  │
└────────┘ └────────┘ └────────────┘
    │           │           │
    └───────────┼───────────┘
                ▼
         [Economics Layer]
    (Wage parity, spread, CS)
```

**Key Insight**: Everything downstream of encoder is modality-agnostic.

---

## File Summary

### Created Files
| File | Status | Purpose |
|------|--------|---------|
| `src/encoders/video_encoder.py` | ✅ Complete | Video encoder implementations |
| `src/encoders/builder.py` | ✅ Complete | Unified encoder builder |
| `src/envs/video_wrappers.py` | ✅ Complete | Video environment wrapper |
| `configs/dishwashing_video.yaml` | ✅ Complete | Video mode configuration |
| `VIDEO_TO_POLICY_ROADMAP.md` | ✅ Complete | Implementation roadmap |
| `VIDEO_PIPELINE_STATUS.md` | ✅ Complete | This document |

### Modified Files
| File | Modifications Needed | Status |
|------|---------------------|--------|
| `train_sac.py` | Add video/state mode switching | 🔄 TODO |
| `src/deep_learning/novelty_diffusion.py` | Operate on latents not pixels | 🔄 TODO |

### New Files Needed
| File | Purpose | Status |
|------|---------|--------|
| `experiments/compare_state_vs_video.py` | Modality comparison | 📋 TODO |
| `requirements-gpu.txt` | GPU dependencies | 📋 TODO |
| `GPU_README.md` | Cloud training guide | 📋 TODO |
| `src/envs/physics/__init__.py` | Physics env interface | 📋 TODO |

---

## Next Actions

### Immediate (This Week)
1. ✅ **DONE**: Video encoder + wrapper + config
2. **TODO**: Modify `train_sac.py` for video mode support
3. **TODO**: Test short training run (50 episodes) in video mode
4. **TODO**: Verify economics unchanged (wage parity, CS, spread)

### Short-term (Next Week)
1. **TODO**: Wire diffusion novelty to latents
2. **TODO**: Run state vs video comparison (200 episodes each)
3. **TODO**: Create GPU requirements + README

### Medium-term (Next Month)
1. **TODO**: Set up RunPod/AWS GPU training
2. **TODO**: Test R3D18 encoder on GPU
3. **TODO**: Scaffold PyBullet physics env

---

## Key Validation Criteria

Before declaring video mode "complete":

1. ✅ Video encoder works (tested standalone)
2. ✅ Video wrapper works (tested standalone)
3. ⏳ SAC trains on video observations
4. ⏳ Economics metrics match state mode (±5% tolerance)
5. ⏳ Novelty/ΔMPL operates on latents
6. ⏳ State vs video comparison shows equivalence

---

## Why This Matters

**Before**: Training on hand-crafted state vectors (not scalable to real robots)

**After**: Training on video observations (enables real demonstrations, transfer learning)

**Impact**:
- Can use YouTube demonstrations
- Can use kinesthetic teaching
- Can transfer sim→real via vision
- **Core thesis validated**: "Economics-driven video-to-policy learning"

---

**Status**: Infrastructure complete, integration in progress.

**Blocking**: None - all dependencies ready.

**ETA**: Video mode fully working within 1-2 days of focused work.

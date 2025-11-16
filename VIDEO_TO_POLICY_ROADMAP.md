# Video-to-Policy (V2P) Implementation Roadmap

**Status**: Step 1 Complete (Video Encoder Interface)

---

## Overview

This document tracks the transition from state-based observations to video-based observations, enabling the core thesis: **"Video-conditioned policies whose reward is labor economics and whose data is priced via ΔMPL."**

### Strategic Rationale

The economic architecture is already solid:
- ✅ Wage indexer + consumer surplus guarantee
- ✅ Spread allocation based on ΔMPL contributions
- ✅ Online data value estimator
- ✅ Securit izable cashflow model

**Weak link**: We're still training on toy state vectors, not real modality (video).

**Approach**: Video pathway first, then physics. This de-risks the core claim before adding sim complexity.

---

## Step 1: Video Encoder Interface ✅ COMPLETE

### What We Built

**1. Video Encoder Module** (`src/encoders/video_encoder.py`)

Three architecture options:
- **Simple2DCNN**: 2D CNN per-frame + temporal pooling (CPU-friendly, 651K params)
- **Simple3DCNN**: Lightweight 3D convolutions for spatiotemporal features (837K params)
- **R3D18**: ResNet3D-18 pretrained on Kinetics (requires GPU + torchvision)

Interface:
```python
encoder = VideoEncoder(latent_dim=128, arch='simple2dcnn')
z = encoder(video)  # (B, T, C, H, W) -> (B, latent_dim)
```

**2. Unified Encoder Builder** (`src/encoders/builder.py`)

Config-driven encoder construction:
```python
encoder = build_encoder(config, obs_dim=10, device=device)  # MLP
encoder = build_encoder(config, video_shape=(8,3,64,64), device=device)  # Video
```

**3. Config Integration** (`configs/dishwashing_feasible.yaml`)

Added encoder configuration section:
```yaml
encoder:
  type: "mlp"       # Switch to "video" for visual observations
  latent_dim: 128

  mlp:
    hidden_dim: 256

  video:
    arch: "simple2dcnn"
    input_channels: 3
    frames: 8
    height: 64
    width: 64
```

### Key Design Decision

**SAC is now modality-agnostic.** The actor/critic networks only see latent embeddings `z`, not raw observations. This means:
- Switch modality by changing `encoder.type` in config
- No changes needed to SAC algorithm
- Same economic layer works for state or video

### Testing

```bash
# Test video encoder
python3 src/encoders/video_encoder.py

# Test builder
PYTHONPATH=. python3 src/encoders/builder.py
```

Output:
```
[Simple2DCNN] Input: (4, 8, 3, 64, 64) -> Output: (4, 128)
[Simple3DCNN] Input: (4, 8, 3, 64, 64) -> Output: (4, 128)
✅ All encoders working!
```

---

## Step 2: VideoWrapperEnv (NEXT)

### Goal

Wrap existing `DishwashingEnv` to emit synthetic video instead of state vectors.

### Implementation Plan

**Create**: `src/envs/video_wrappers.py`

```python
class DishwashingVideoEnv:
    """
    Wraps DishwashingEnv and returns (T, C, H, W) frames instead of state.

    For now: Generate synthetic visualizations of state (colored bars for metrics).
    Later: Swap in real sim/camera feeds.
    """

    def __init__(self, base_env, frames=8, height=64, width=64):
        self.base_env = base_env
        self.frames = frames
        self.height = height
        self.width = width
        self.frame_buffer = deque(maxlen=frames)

    def reset(self):
        state = self.base_env.reset()
        frame = self._state_to_frame(state)
        self._init_frame_buffer(frame)
        return self._get_video_obs()

    def step(self, action):
        state, reward, done, info = self.base_env.step(action)
        frame = self._state_to_frame(state)
        self.frame_buffer.append(frame)
        video_obs = self._get_video_obs()
        return video_obs, reward, done, info

    def _state_to_frame(self, state):
        """Convert state vector to (C, H, W) image"""
        # For now: Simple visualization (colored bars)
        # Later: Render from sim or use camera feed
        frame = self._render_state_bars(state)
        return frame

    def _get_video_obs(self):
        """Stack frames into (T, C, H, W) tensor"""
        return np.stack(list(self.frame_buffer), axis=0)
```

**Why Synthetic Video First?**
- Proves video → latent → RL → economics pathway works
- No dependency on Isaac/Mujoco/physics yet
- Fast iteration on CPU

**Config Update**:
```yaml
env:
  type: "dishwashing_video"
  base: "dishwashing"
  video:
    frames: 8
    height: 64
    width: 64
    render_mode: "synthetic"  # Later: "sim", "camera"
```

**Test**:
```python
env = DishwashingVideoEnv(base_env, frames=8, height=64, width=64)
obs = env.reset()  # (8, 3, 64, 64)
obs, reward, done, info = env.step(action)  # video observation
```

---

## Step 3: Diffusion Novelty on Video Latents

### Goal

Wire diffusion-based novelty estimation to work on video encoder latents instead of raw pixels.

### Implementation Plan

**Modify**: `src/deep_learning/novelty_diffusion.py`

Current:
```python
# Operates on state vectors
novelty = diffusion_novelty(state)
```

New:
```python
# Operates on video latent embeddings
z = encoder(video)  # (B, T, C, H, W) -> (B, latent_dim)
novelty = diffusion_novelty(z)
```

**Key Change**: Diffusion prior/denoiser operate in latent space (128-dim), not pixel space (8×3×64×64).

**Why Latent Space?**
- More efficient than pixel-space diffusion
- Latents are already semantically meaningful (encoded by CNN)
- Enables scaling to high-res video later

**Integration** (`train_sac.py`):
```python
# Get video observation
video = env.reset()  # (T, C, H, W)

# Encode to latent
z = encoder(video)  # (latent_dim,)

# Compute novelty in latent space
novelty = diffusion_novelty(z)

# Predict ΔMPL_cust from novelty
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

**Now the entire pipeline operates on video:**
- Video → Encoder → Latent → SAC policy
- Video → Encoder → Latent → Novelty → ΔMPL → Pricing
- Economics layer unchanged

---

## Step 4: GPU Infrastructure (Parallel Track)

### Goal

Prepare for GPU training without fully migrating yet.

### Implementation Plan

**1. Requirements File** (`requirements-gpu.txt`):
```
torch>=2.0.0+cu118
torchvision>=0.15.0+cu118
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
```

**2. Docker Config** (`docker/Dockerfile.gpu`):
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /workspace

# Install dependencies
COPY requirements-gpu.txt .
RUN pip install -r requirements-gpu.txt

# Copy code
COPY . .

# Expose port for TensorBoard
EXPOSE 6006

CMD ["python3", "train_sac.py"]
```

**3. RunPod/Cloud Setup** (`GPU_README.md`):
```markdown
# GPU Training Setup

## RunPod Template
- Base: pytorch/pytorch:2.0.1-cuda11.8
- GPU: RTX 4090 / A5000
- Storage: 50GB

## Launch Training
```bash
git clone <repo>
cd robotics-v-p-economics-model
pip install -r requirements-gpu.txt
python3 train_sac.py --config configs/dishwashing_feasible.yaml
```

## Config Switch for GPU
```yaml
encoder:
  type: "video"
  video:
    arch: "r3d18"
    pretrained: true  # Use Kinetics weights
```
```

**4. Local Testing**:
- Keep `simple2dcnn` for CPU development
- Switch to `r3d18` only when deploying to GPU

---

## Step 5: Physics Environment Scaffolding

### Goal

Scaffold physics sim without going deep into contact dynamics yet.

### Implementation Plan

**Create**: `src/envs/physics/`

**Option A: Isaac Gym**
- Pros: Fast GPU simulation, robot-native
- Cons: Steep learning curve, NVIDIA GPU required
- Use case: High-throughput policy training

**Option B: PyBullet**
- Pros: Easy to get started, CPU-friendly
- Cons: Slower than Isaac
- Use case: Quick prototyping, smaller experiments

**Option C: MuJoCo**
- Pros: Industry standard, good physics
- Cons: Commercial license (though now free)
- Use case: High-quality contact physics

**Recommended Start: PyBullet**

**Create**: `src/envs/physics/dishwashing_pybullet.py`

```python
class DishwashingPyBullet:
    """
    Simple tray + dish simulation with:
    - (speed, care) actions or force/precision equivalent
    - error/no-error outcomes based on contact forces
    - RGB camera observations (64x64)
    """

    def __init__(self, camera_config):
        self.p = pybullet
        self.p.connect(self.p.DIRECT)  # Headless

        # Load simple tray + dish URDF
        self.tray_id = self.p.loadURDF("tray.urdf")
        self.dish_id = self.p.loadURDF("dish.urdf")

        # Camera
        self.camera = Camera(height=64, width=64, fov=60)

    def step(self, action):
        speed, care = action

        # Apply forces (simple model)
        force = speed * MAX_FORCE * (1 - care * CARE_PENALTY)
        self.p.applyExternalForce(...)

        # Simulate
        for _ in range(SUBSTEPS):
            self.p.stepSimulation()

        # Check for errors (contact forces)
        contact_points = self.p.getContactPoints(...)
        error = self._check_for_damage(contact_points)

        # Render camera view
        rgb = self.camera.render()

        return rgb, reward, done, info
```

**Integration**:
```yaml
env:
  type: "dishwashing_physics"
  physics:
    backend: "pybullet"  # "isaac", "mujoco"
    camera:
      height: 64
      width: 64
      fov: 60
    render_headless: true
```

**Key Constraint**: "Env-compatible API + video outputs" is the main deliverable. Don't get stuck on perfect physics yet.

---

## Roadmap Summary

| Step | Status | CPU-Friendly | GPU Required | Deliverable |
|------|--------|--------------|--------------|-------------|
| 1. Video Encoder | ✅ Complete | ✅ Yes (Simple2D/3DCNN) | Optional (R3D18) | Modality-agnostic SAC |
| 2. VideoWrapperEnv | ⏳ Next | ✅ Yes (synthetic video) | No | Video → RL pipeline |
| 3. Diffusion Novelty | 🔜 Pending | ⚠️ Slow | Recommended | ΔMPL from video |
| 4. GPU Infrastructure | 🔜 Pending | N/A | ✅ Yes | Cloud training ready |
| 5. Physics Env | 🔜 Pending | ⚠️ PyBullet only | Isaac/GPU | Real sim observations |

---

## Why This Order?

### Video First, Physics Second

**Rationale**:
1. **De-risks core thesis**: Proves "video → latent → RL → economics" works
2. **Faster iteration**: No sim debugging, contact tuning, domain randomization
3. **CPU development**: Can work on laptop, defer GPU costs
4. **Clear handoff**: Once video pathway works, physics is "just another env"

### What We Prove at Each Step

**Step 2** (VideoWrapperEnv):
- SAC trains on video observations
- Economics layer unchanged
- Wage parity, spread allocation, consumer surplus all work with video

**Step 3** (Diffusion Novelty):
- Novelty computed from video latents
- ΔMPL estimation from novelty
- Data pricing based on video, not state

**Step 4** (GPU):
- Switch to R3D18 encoder for real video
- Faster training (1000 episodes in hours, not days)
- Ready for real demonstrations

**Step 5** (Physics):
- Replace synthetic video with sim renders
- Contact dynamics for error detection
- Domain randomization for transfer

---

## Current State

### Files Created/Modified (Step 1)

**New Files**:
- `src/encoders/video_encoder.py` - Video encoder implementations
- `src/encoders/builder.py` - Unified encoder builder
- `VIDEO_TO_POLICY_ROADMAP.md` - This document

**Modified Files**:
- `configs/dishwashing_feasible.yaml` - Added encoder configuration

### What Works Now

```bash
# Train with MLP encoder (current)
python3 train_sac.py --config configs/dishwashing_feasible.yaml

# Switch to video encoder (after Step 2)
# Edit config: encoder.type = "video"
python3 train_sac.py --config configs/dishwashing_feasible.yaml
```

### What's Next

**Immediate** (this week):
1. Implement `DishwashingVideoEnv` wrapper
2. Test video → encoder → SAC pathway
3. Verify economics layer works unchanged

**Short-term** (next 2 weeks):
1. Wire diffusion novelty to video latents
2. Validate ΔMPL estimation from video
3. Set up GPU infrastructure (RunPod/AWS)

**Medium-term** (next month):
1. Scaffold PyBullet dishwashing env
2. Replace synthetic video with sim renders
3. Train on GPU with R3D18 encoder

---

## Key Insight

**The economic architecture is complete. The next frontier is modality.**

We have:
- ✅ Wage parity convergence
- ✅ Lagrangian quality constraints
- ✅ ΔMPL-based spread allocation
- ✅ Consumer surplus guarantee
- ✅ Structural deflation theorem (validated)

But we're still training on:
- ❌ Hand-crafted state vectors
- ❌ Toy environment without physics
- ❌ No visual observations

**Video pathway unlocks**:
- Real human demonstrations (YouTube, kinesthetic teaching)
- Transfer from video → robot policy
- Data pricing based on visual novelty
- Thesis validation: "Economics-driven video-to-policy"

---

**Next Action**: Implement Step 2 (VideoWrapperEnv) to complete the video → RL → economics pipeline.

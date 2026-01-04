# Process Reward Module (Robo-Dopamine-style)

Provides **Potential-Based Reward Shaping (PBRS)** downstream of SceneTracks_v1 and Motion Hierarchy Network (MHN). Consumes latent features (not pixels), produces progress potential Φ and confidence, and computes policy-invariant shaped rewards.

## Architecture Overview

```
SceneTracks_v1 ──┬──► FeatureExtractor ──► EpisodeFeatures
                 │                              │
MHN Summary ─────┘                              │
                                                ▼
                                           HopPredictor ──► hops, uncertainties
                                                │
                                                ▼
                                    ┌─── ProgressPerspectives ───┐
                                    │   • Phi_I (incremental)    │
                                    │   • Phi_F (forward)        │
                                    │   • Phi_B (backward)       │
                                    └────────────┬───────────────┘
                                                 │
                        FusionOverride ──────────┼
                                                 │
                                                 ▼
                                            FusionNet ──► weights, confidence
                                                 │
                                                 ▼
                                           Φ* (fused potential)
                                                 │
                                                 ▼
                                         PBRS Wrapper
                                                 │
                                                 ▼
                            r_shape[t] = γ·Φ*[t+1] - Φ*[t]
```

## Key Components

### 1. Feature Extraction (`features.py`)

Extracts features from SceneTracks_v1 without requiring pixels:

**Always available:**
- `poses_R`, `poses_t`, `scales` — World-frame kinematics
- `visibility`, `occlusion` — Per-entity visibility
- `ir_loss`, `converged` — IR diagnostics

**Optional:**
- `z_shape`, `z_tex` — Latent embeddings (float16)
- MHN features — Motion hierarchy difficulty/plausibility

### 2. Hop Model (`hop_model.py`)

Predicts "hop" (progress) between BEFORE and AFTER states:

```python
hop_hat ∈ [-1, 1]  # Positive = progress, Negative = regress
```

**LabelProvider abstraction** supports multiple label sources:
- `OracleDistanceLabelProvider` — Feature-space distance to goal (bootstrap)
- `TaskSuccessLabelProvider` — Sparse task success signals
- `ProxyLabelProvider` — Custom domain-specific heuristics

### 3. Progress Perspectives (`progress_perspectives.py`)

Three complementary progress estimates:

| Perspective | Definition | Best When |
|-------------|------------|-----------|
| **Φ_I** | Incremental cumsum of hops | Smooth trajectories |
| **Φ_F** | Forward: distance from init | Early in episode |
| **Φ_B** | Backward: 1 - dist to goal | Goal is known |

### 4. Fusion (`fusion.py`)

Learns to combine perspectives — **NO simple averaging**:

```python
Φ* = Σ w_i · Φ_i   where w = softmax(FusionNet(inputs))
```

**Orchestrator control via `FusionOverride`:**
- `temperature` — Softmax sharpness
- `candidate_mask` — Enable/disable perspectives
- `risk_tolerance` — Minimum confidence threshold
- `entropy_penalty` — Penalize uncertain fusion
- `weight_smoothing` — Temporal smoothing factor

### 5. PBRS Shaping (`shaping.py`)

Policy-invariant reward shaping:

```python
r_shape[t] = γ · Φ*[t+1] - Φ*[t]
```

**Guaranteed properties:**
- Optimal policy unchanged (Ng et al., 1999)
- Telescoping: `Σ r_shape = γ^T · Φ[T] - Φ[0]`
- Optional confidence gating

## API

### Primary Entry Point

```python
from src.process_reward import process_reward_episode, ProcessRewardConfig

# Load scene tracks
from src.vision.scene_ir_tracker.serialization import deserialize_scene_tracks_v1
data = dict(np.load("episode.npz", allow_pickle=False))
scene_tracks = deserialize_scene_tracks_v1(data)

# Compute process reward
# For offline evaluation (allows goal_frame_idx=None which uses last frame as goal):
cfg = ProcessRewardConfig(gamma=0.99, use_confidence_gating=True, online_mode=False)
result = process_reward_episode(
    scene_tracks=scene_tracks,
    instruction="pick up the box",
    goal_frame_idx=None,  # Uses last frame (only valid in offline mode)
    cfg=cfg,
    mhn=mhn_summary,  # Optional
    orchestrator_overrides=FusionOverride(temperature=0.5),  # Optional
)

# For online RL training (requires explicit goal - safe by default):
cfg_online = ProcessRewardConfig(gamma=0.99)  # online_mode=True by default
result_online = process_reward_episode(
    scene_tracks=scene_tracks,
    instruction="pick up the box",
    goal_frame_idx=scene_tracks.num_frames - 1,  # Explicit goal required
    cfg=cfg_online,
)

# Use outputs
print(f"Shaped reward sum: {result.r_shape.sum():.3f}")
print(f"Final potential: {result.phi_star[-1]:.3f}")
print(f"Mean confidence: {result.conf.mean():.3f}")
```

### Per-Step API (for RL rollouts)

```python
from src.process_reward import process_reward_step

output = process_reward_step(
    scene_tracks_t=frame_data,
    t=current_timestep,
    goal_frame_idx=goal_idx,
    cfg=cfg,
    prev_phi=previous_output.phi_t1,
    prev_conf=previous_output.conf_t1,
    init_features=init_features,
    goal_features=goal_features,
)

shaped_reward = output.r_shape
```

## Configuration

### ProcessRewardConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.99 | Discount factor for PBRS |
| `use_confidence_gating` | True | Scale r_shape by confidence |
| `feature_dim` | 32 | Per-track feature dimension |
| `use_latents` | True | Use z_shape/z_tex if available |
| `use_mhn_features` | True | Use MHN summary features |
| `phi_clip_min/max` | 0.0/1.0 | Phi clipping bounds |
| `device` | "cpu" | PyTorch device |

### FusionOverride

| Parameter | Default | Description |
|-----------|---------|-------------|
| `temperature` | 1.0 | Softmax temperature (lower = sharper) |
| `candidate_mask` | (True, True, True) | [Φ_I, Φ_F, Φ_B] enable flags |
| `risk_tolerance` | 0.3 | Min confidence threshold |
| `entropy_penalty` | 0.1 | Penalty for uncertain fusion |
| `weight_smoothing` | 0.0 | Temporal smoothing [0, 1] |
| `min_weight_floor` | 0.01 | Minimum weight for enabled candidates |

## Output Structure

### ProcessRewardEpisodeOutput

```python
@dataclass
class ProcessRewardEpisodeOutput:
    phi_star: np.ndarray      # (T,) fused potential
    conf: np.ndarray          # (T,) confidence
    r_shape: np.ndarray       # (T-1,) shaped rewards (PBRS)
    perspectives: ProgressPerspectives
    diagnostics: FusionDiagnostics
    episode_id: Optional[str]
    metadata: Dict[str, Any]
```

### FusionDiagnostics

```python
@dataclass
class FusionDiagnostics:
    weights: np.ndarray       # (T, 3) fusion weights [I, F, B]
    entropy: np.ndarray       # (T,) weight entropy
    disagreement: np.ndarray  # (T,) max |Φ_i - Φ_j|
    gating_factor: np.ndarray # (T,) applied gating
```

## Smoke Test

```bash
# With synthetic data
python scripts/smoke_test_process_reward.py --synthetic --verbose

# With real data
python scripts/smoke_test_process_reward.py \
    --input data/episode.npz \
    --output results/process_reward_output.json
```

## Tests

```bash
# Run all process reward tests
pytest tests/process_reward/ -v

# Specific test suites
pytest tests/process_reward/test_pbrs.py -v        # PBRS telescoping
pytest tests/process_reward/test_fusion.py -v      # Fusion behavior
pytest tests/process_reward/test_integration.py -v # End-to-end
```

## Integration Points

### Training Loop Hook

```python
# In rollout labeling / episode processing
from src.process_reward import process_reward_episode

result = process_reward_episode(scene_tracks, instruction, cfg=cfg)

# Log shaped rewards alongside MPL outcomes
episode_log["shaped_rewards"] = result.r_shape
episode_log["phi_star_trace"] = result.phi_star
episode_log["process_reward_summary"] = result.summary()
```

### Reward Model Integration

The module is designed as an **optional component**. Enable via config flags:

```yaml
# In training config
reward_shaping:
  enabled: true
  gamma: 0.99
  use_confidence_gating: true
  fusion_override:
    temperature: 0.8
    risk_tolerance: 0.3
```

## Design Decisions

1. **Consumes latents, not pixels** — Operates downstream of SceneTracks_v1 and MHN
2. **No simple averaging** — FusionNet learns weighted combination
3. **Orchestrator-controllable** — All fusion parameters exposed via FusionOverride
4. **Policy-invariant PBRS** — Guaranteed to preserve optimal policy
5. **Multi-source labels** — LabelProvider abstraction supports oracle/human/LLM/task signals
6. **Fallback support** — HeuristicHopPredictor and HeuristicFusion work without PyTorch

## Dependencies

- **Required:** numpy
- **Optional:** torch (for HopNet, FusionNet neural models)
- **Upstream:** `src.vision.scene_ir_tracker.serialization` (SceneTracksLite)
- **Upstream:** `src.vision.motion_hierarchy.metrics` (MotionHierarchySummary)

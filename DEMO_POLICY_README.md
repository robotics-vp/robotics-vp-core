# DemoPolicy & Sim Demo Runner - Implementation Summary

**Completed:** 2-week demo inference wrapper and simulator runner (not YC demo)

## Overview

Implemented a clean inference wrapper (`DemoPolicy`) and demo simulation runner (`run_demo_in_sim.py`) that sit on top of the existing Stage 6+ stack without touching reward/econ math.

## Components Implemented

### 1. DemoPolicy (`src/inference/demo_policy.py`)

**Purpose:** Inference wrapper for trained models in demo/deployment scenarios.

**Features:**
- Loads trained components (vision backbone, spatial RNN, SIMA-2, Hydra policy)
- Respects neural/stub flags from `config/pipeline.yaml`
- Constructs `ConditionVector` and `PolicyObservation` correctly
- Returns actions compatible with env backends (PyBullet/Isaac)
- Fully deterministic given seed
- JSON-safe metadata/summaries

**API:**
```python
from src.inference import DemoPolicy, DemoPolicyConfig

# Default config (from pipeline.yaml)
policy = DemoPolicy()

# Custom config
config = DemoPolicyConfig(
    backend="pybullet",
    canonical_task_id="drawer_open",
    use_neural_vision=False,
    seed=0,
)
policy = DemoPolicy(config=config)

# Reset with seed
policy.reset(seed=42)

# Act on raw env obs
raw_obs = env.reset()
action_dict = policy.act(raw_obs)
action = action_dict["action"]  # numpy array
metadata = action_dict["metadata"]  # debugging info

# Get summary
summary = policy.get_summary()  # JSON-safe dict
```

**Configuration:**
- All neural flags default to `False` (stub mode)
- Loads from `config/pipeline.yaml` by default
- Override via `DemoPolicyConfig` constructor or dict

**Components loaded:**
- Vision backbone: RegNet + BiFPN (neural if enabled, else hash-based stub)
- Spatial RNN: ConvGRU (neural if enabled, else identity stub)
- SIMA-2 segmenter: U-Net (neural if enabled, else default OOD/recovery stub)
- Hydra policy: Multi-objective SAC (neural if enabled, else zero-action stub)
- ConditionVectorBuilder: For runtime condition vectors
- ObservationAdapter: For policy observation construction

**Determinism:**
- Given same seed, same sequence of raw_obs → same actions
- Seeds Python, NumPy, PyTorch RNGs
- Resets internal state (spatial RNN hidden state, step counter)

### 2. Sim Demo Runner (`scripts/run_demo_in_sim.py`)

**Purpose:** Run deterministic rollouts using DemoPolicy in sim env.

**Usage:**
```bash
python3 scripts/run_demo_in_sim.py \
  --env-backend pybullet \
  --num-episodes 5 \
  --max-steps 200 \
  --seed 0 \
  --output-dir results/demo_sim
```

**Outputs:**

1. **`episodes.jsonl`** (one line per episode):
   ```json
   {
     "episode_id": 0,
     "seed": 0,
     "success": false,
     "steps": 5,
     "total_reward": -0.15,
     "econ_summary": {
       "avg_mpl": 0.97,
       "avg_energy": 1.02,
       "total_errors": 3
     },
     "ood_stats": {
       "max_ood_risk": 0.1,
       "ood_step_count": 0,
       "ood_step_fraction": 0.0
     },
     "recovery_stats": {
       "max_recovery_priority": 0.0,
       "recovery_step_count": 0,
       "recovery_step_fraction": 0.0
     },
     "skill_mode_counts": {"recovery_heavy": 5}
   }
   ```

2. **`steps.jsonl`** (one line per step):
   ```json
   {
     "episode_id": 0,
     "step": 0,
     "action_summary": {"action_norm": 0.0, "action_mean": 0.0},
     "ood_step_flags": {"ood_risk_level": 0.1, "is_ood": false},
     "recovery_step_flags": {"recovery_priority": 0.0, "is_recovery": false},
     "reward_scalar": -0.07,
     "econ_step_summary": {"mpl_proxy": 1.09, "energy_proxy": 1.25}
   }
   ```

3. **Stdout summary:**
   ```
   Episodes: 5
   Success rate: 2/5 (40.0%)
   Avg episode length: 150.2 steps
   Avg MPL proxy: 0.98
   Avg energy proxy: 1.12
   ```

**Environment backends:**
- PyBullet: Uses existing `PyBulletBackend` + `DrawerVasePhysicsEnv`
- Isaac: Uses `IsaacAdapter` stub
- Fallback: `StubEnv` if real env unavailable

**Determinism:**
- Same `--seed` + same config → identical logs
- All randomness controlled via seed

### 3. Smoke Tests

#### `scripts/smoke_test_demo_policy_inference.py`

Tests DemoPolicy:
- Instantiation (default config, dict config, DemoPolicyConfig)
- `reset(seed)` determinism
- `act(raw_obs)` returns valid action dict
- Deterministic actions for fixed seed + obs
- `get_summary()` returns JSON-safe dict

**Run:**
```bash
python3 scripts/smoke_test_demo_policy_inference.py
```

**Output:**
```
================================================================================
[smoke_test_demo_policy_inference] All tests passed ✓
================================================================================
```

#### `scripts/smoke_test_run_demo_in_sim.py`

Tests run_demo_in_sim.py:
- Script runs without error
- Output directory created
- `episodes.jsonl` created and valid
- `steps.jsonl` created and valid
- All required fields present
- JSON-safe outputs
- Exit code 0

**Run:**
```bash
python3 scripts/smoke_test_run_demo_in_sim.py
```

**Output:**
```
================================================================================
[smoke_test_run_demo_in_sim] All tests passed ✓
================================================================================
```

## Hard Constraints (Verified)

✅ **Does NOT touch:**
- Reward math
- Economic controller
- Phase H core logic
- Pricing
- Ontology schemas

✅ **Everything is:**
- Deterministic
- JSON-safe
- Behind flags/configs
- No hidden behavior changes

✅ **Optimized for:**
- Clean integration (not "hacky demo code")
- Future reuse (YC demo later)

## Integration with Existing Stack

**Uses existing components:**
- `src/config/pipeline.py` - loads canonical task, neural flags, checkpoints
- `src/observation/adapter.py` - `ObservationAdapter` for policy obs
- `src/observation/condition_vector_builder.py` - `ConditionVectorBuilder`
- `src/vision/regnet_backbone.py` - RegNet + BiFPN vision backbone
- `src/vision/interfaces.py` - `VisionFrame`, `VisionLatent`
- `src/rl/trunk_net.py` - Spatial RNN (when neural)
- `src/rl/hydra_heads.py` - Hydra policy (when neural)
- `src/ontology/sima2_segmenter.py` - SIMA-2 (when neural)
- `src/envs/physics/pybullet_backend.py` - PyBullet backend
- `src/env/isaac_adapter.py` - Isaac adapter

**Respects existing configs:**
- `config/pipeline.yaml` - canonical_task_id, use_neural flags, checkpoint paths
- `config/vision.yaml` - vision config (if present)
- Economic domain configs - for econ logging only (no modification)

## File Structure

```
src/inference/
├── __init__.py                        # Exports DemoPolicy, DemoPolicyConfig
└── demo_policy.py                     # Main inference wrapper

scripts/
├── run_demo_in_sim.py                 # Sim demo runner
├── smoke_test_demo_policy_inference.py # DemoPolicy smoke test
└── smoke_test_run_demo_in_sim.py      # Sim runner smoke test

results/
└── demo_sim/                          # Default output directory
    ├── episodes.jsonl
    └── steps.jsonl
```

## Next Steps (Future)

When ready for YC demo or full deployment:

1. **Train neural components:**
   - Set `use_neural: true` in `config/pipeline.yaml`
   - Run Stage 6 training pipeline
   - Checkpoints will be loaded automatically

2. **Add real envs:**
   - Replace stub envs with full PyBullet/Isaac/UE5 envs
   - No code changes needed (already uses backend factory)

3. **Scale up:**
   - Increase `--num-episodes` and `--max-steps`
   - Add parallel rollout support if needed

4. **Analysis:**
   - Parse `episodes.jsonl` and `steps.jsonl` for metrics
   - Aggregate across runs for performance stats

## Testing

Both smoke tests pass:

```bash
# DemoPolicy inference test
python3 scripts/smoke_test_demo_policy_inference.py
# ✓ All tests passed

# Sim runner test
python3 scripts/smoke_test_run_demo_in_sim.py
# ✓ All tests passed
```

## Notes

- This is for the **2-week demo**, not the YC demo
- All defaults preserve existing behavior (neural flags off)
- Clean separation: inference wrapper doesn't touch training/econ code
- Future-proof: when neural components are trained, just flip flags

# Drawer+Vase Physics Environment

**Phase C: Fei-Fei Benchmark Implementation**

A PyBullet-based physics environment for the classic robotics challenge: open a drawer while avoiding collision with a fragile vase.

---

## Overview

The Drawer+Vase environment is a physics-first implementation that integrates with the Phase B synthetic data flywheel (trust + econ + λ weighting) and prepares for HRL/VLA layers in subsequent Phase C steps.

### Task

**Objective:** Open the top drawer of a cabinet without hitting or tipping over a fragile vase positioned near the drawer pull path.

**Challenge:** The vase creates an obstacle that requires careful planning and execution. High-speed or careless movements risk collision.

---

## Environment Details

### State Space (State Mode)

13-dimensional continuous vector:

| Index | Feature | Description |
|-------|---------|-------------|
| 0-2 | `ee_pos` | End-effector position (x, y, z) |
| 3-5 | `ee_vel` | End-effector velocity (vx, vy, vz) |
| 6 | `drawer_frac` | Drawer open fraction [0, 1] |
| 7-9 | `vase_pos` | Vase position (x, y, z) |
| 10 | `vase_upright` | Vase uprightness [0, 1] |
| 11 | `min_clearance` | Minimum clearance to vase |
| 12 | `grasp_state` | Whether grasping handle [0, 1] |

### Action Space

3-dimensional continuous vector: End-effector velocity command (dx, dy, dz) in [-1, 1].

### Vision Mode

RGB images (128x128x3) rendered from PyBullet camera. Compatible with Phase A.5 AlignedVideoEncoder for z_V encoding.

---

## Physics Model

### Objects

1. **Cabinet Body** (static)
   - Position: (0, 0, 0.5)
   - Size: 0.5m × 0.4m × 0.6m
   - Two drawers (top = movable, bottom = static)

2. **Top Drawer** (dynamic, constrained)
   - Mass: 1.0 kg
   - Friction: 0.3
   - Damping: 0.8
   - Max extension: 0.35m
   - Slides along Y-axis (prismatic joint)

3. **Vase** (dynamic, fragile)
   - Position: (0.3, 0, 0.8) - near drawer path
   - Radius: 0.04m
   - Height: 0.12m
   - Mass: 0.3 kg
   - Fragility threshold: 10 N·s
   - Tip angle threshold: 30°

4. **End-Effector** (controlled)
   - Start position: (-0.3, 0, 0.8)
   - Mass: 0.5 kg
   - Max force: 50 N
   - Max velocity: 0.3 m/s

### Collision Detection

- EE-Vase contact impulses tracked
- Drawer-Vase contact impulses tracked
- High-risk contacts (impulse > 1.0) counted
- Vase breaks if impulse > fragility threshold

### Energy Model

Simple proportional model based on EE velocity:
```
delta_energy_Wh = ||ee_vel|| × control_dt × 0.01
```

---

## Termination Conditions

| Condition | Reason | Done |
|-----------|--------|------|
| Vase broken (collision) | `vase_collision` | Yes |
| Vase tipped | `vase_tipped` | Yes |
| Drawer ≥ 90% open | `success` | Yes |
| High-risk contacts ≥ 5 | `sla_violation` | Yes |
| Steps ≥ max_steps | `max_steps` | Yes |

---

## Economic Integration

### EconParams Preset: `drawer_vase`

```python
DrawerVaseEconParams(
    value_per_successful_drawer_open=5.0,  # $ per success
    vase_break_cost=50.0,                   # $ per broken vase
    electricity_price_kWh=0.12,             # $/kWh
    other_costs_per_hr=2.0,                 # $/hr maintenance
    allowable_risk_tolerance=0.1,           # Max collision probability
    fragility_penalty_coeff=10.0,           # Penalty multiplier
)
```

### EpisodeInfoSummary Compatibility

Info dictionary includes all fields required for:
- `summarize_drawer_vase_episode()` → `EpisodeInfoSummary`
- `compute_econ_reward()` integration
- Phase B synthetic weighting pipeline

Key info fields:
- `drawer_fraction`: Task progress
- `vase_intact`: Safety constraint
- `min_clearance`: Risk metric
- `total_impulse`: Collision severity
- `n_high_risk_contacts`: SLA tracking
- `energy_Wh`: Operating cost
- `success`: Task completion
- `terminated_reason`: Why episode ended

---

## Configuration Files

### State Mode

`configs/drawer_vase_physics_state.yaml`
- Low-dimensional state observations
- Fast training
- No encoder overhead

### Vision Mode (Aligned)

`configs/drawer_vase_physics_aligned.yaml`
- RGB camera observations
- z_V encoding via AlignedVideoEncoder
- World model integration ready
- Full Phase B flywheel support

---

## Scripted Baseline

`policies/scripted/drawer_open_avoid_vase.py`

State machine policy:
1. **Approach**: Move towards drawer handle, avoiding vase
2. **Grasp**: Fine-tune to grab handle
3. **Pull**: Pull drawer open (negative Y direction)
4. **Retreat**: Return to safe position

Features:
- Vase avoidance via repulsive potential field
- Safe offset maintenance (0.15m from vase)
- Never directly collides with vase

---

## Usage

### Smoke Test

```bash
python scripts/smoke_test_drawer_vase.py
```

Tests:
- State mode observation format
- Vision mode rendering
- EconParams loading
- Drawer opening trajectory
- EpisodeInfoSummary generation

### Scripted Evaluation

```bash
python scripts/eval_drawer_vase_scripted.py --episodes 20
```

Logs:
- Success rate
- Vase collision rate
- Clearance distribution
- Energy usage
- Episode summaries

### Training (SAC)

```bash
python scripts/train_sac_v2.py --config configs/drawer_vase_physics_state.yaml
```

With Phase B synthetic augmentation:
```bash
python scripts/train_offline_with_local_synth.py \
    --config configs/drawer_vase_physics_aligned.yaml \
    --use-lambda-controller
```

---

## Phase B Integration

The environment is fully compatible with:

1. **Stable World Model**
   - Can generate synthetic branches from drawer+vase states
   - Trust scoring via `trust_net`
   - Horizon-stable rollouts

2. **W_Econ_Lattice**
   - Maps (ΔMPL, Δerror, ΔEP) → economic weight
   - J-trained (not heuristic)
   - Respects monotonicity constraints

3. **Lambda Controller**
   - Predicts optimal synthetic budget
   - Respects `max_synth_share` cap
   - Budget controller, not per-sample gate

4. **EpisodeInfoSummary**
   - Same interface as dishwashing env
   - Profit/loss tracking
   - Termination reason logging

5. **Internal Profile**
   - Centralized config via `get_internal_experiment_profile()`
   - Task-specific objective vectors
   - Safety bounds and caps

---

## Future Extensions (Phase C)

### HRL Integration

Low-level skills:
- `LOCATE_DRAWER`
- `LOCATE_VASE`
- `APPROACH_HANDLE`
- `GRASP_HANDLE`
- `PULL_WITH_CLEARANCE`
- `RETREAT_SAFE`

High-level policy selects skills + parameters.

### Vision + Affordance

- Risk map around vase (no-go zones)
- Affordance head for drawer handle detection
- Fragility priors from visual features

### VLA Transformer

- Language instruction: "Open the top drawer without hitting the vase"
- Transformer plans skill sequence
- SIMA-style teacher trajectories for training

---

## Files Created

- `src/envs/drawer_vase_physics_env.py` - Main environment
- `src/config/econ_params.py` - Updated with DrawerVaseEconParams
- `configs/drawer_vase_physics_state.yaml` - State mode config
- `configs/drawer_vase_physics_aligned.yaml` - Vision mode config
- `policies/scripted/drawer_open_avoid_vase.py` - Baseline policy
- `scripts/eval_drawer_vase_scripted.py` - Scripted evaluator
- `scripts/smoke_test_drawer_vase.py` - Smoke test utility
- `docs/drawer_vase_env.md` - This documentation

---

## Dependencies

- PyBullet (physics simulation)
- NumPy (computation)
- Gymnasium (env interface)

Install PyBullet:
```bash
pip install pybullet
```

---

## Success Criteria

1. **Environment runs** without crashes
2. **Scripted policy** achieves >80% success rate
3. **Vase collision rate** <5% for baseline
4. **EpisodeInfoSummary** generates correctly
5. **Phase B integration** works (trust + econ + λ)
6. **Drop-in replacement** for dishwashing task

---

## Notes

- The environment uses mock physics if PyBullet is not installed
- Vision mode requires OpenGL rendering
- The cabinet/drawer model is simplified (not URDF-based)
- Vase fragility is simplified (impulse threshold, not material model)
- Energy model is proportional (not dynamics-based)

This is a **physics-first** environment ready for HRL, VLA, and SIMA-2 integration.

---

*Phase C Environment created on 2025-11-16*
*Compatible with Phase B frozen architecture*

# Phase 3 External Pattern Absorption — 2026-05-20

## Boundary

This note records what Phase 3 borrows from public/local OSS robotics surfaces.
It does not promote any external stack into ontology. Unitree, Isaac, Holosoma,
GR00T, and LeRobot-style tools remain provider/pattern sources under native
Embodiment / Actuation WM contracts.

## Sources inspected

| Source | What we absorb | What remains external |
|--------|----------------|-----------------------|
| Unitree `unitree_rl_gym` G1 config | 12-DoF locomotion policy shape, observation/privileged-observation/action dimensions, PD-control/default-joint-angle pattern, domain-randomization axes | hardware latency, watchdog, real safety envelope, deployed drift |
| Unitree `unitree_rl_lab` | IsaacLab-style task naming, G1-29DoF training/play/deploy rhythm, sim2sim-before-sim2real discipline | actual IsaacLab runtime execution and sim2real deployment |
| Local Unitree/Isaac/Holosoma clones | visible G1 URDF/XML/USD/policy/task assets, 29-DoF whole-body naming patterns, dex-hand variants, local ONNX deploy policy proof | native provider execution, benchmark promotion, hardware calibration |
| NVIDIA Isaac sim-to-real/co-training guidance | mixed sim+real dataset discipline, systematic real-world behavior documentation, explicit sim-to-real gap symptoms | robot-specific execution and real-data collection |

## Code landed from this absorption

- `src/world_model/embodiment_actuation/morphology.py`
  - `G1MorphologyProfile`
  - `MorphologyJointSpec`
  - `MorphologyEvidenceReceipt`
  - G1 12-DoF locomotion, 29-DoF whole-body, and 29-DoF+dex3 joint families
  - public/local scan function for G1 config/model/task evidence
- `src/world_model/embodiment_actuation/neural_seams.py`
  - CPU-runnable local dynamics, inverse-retargeting, action-proposal, and
    drift/calibration seam modules
- `src/world_model/embodiment_actuation/training_corpus.py`
  - Phase 3.4 training-row and manifest builders with promotion blocked until
    GPU/provider/benchmark evidence exists
- `scripts/smoke_test_embodiment_phase34.py`
  - local proof that morphology evidence, training rows, and all seam forward
    passes can run without GPU

## Current local evidence result

The local smoke scan over the current host found:

- `variant`: `g1_29dof`
- `joint_count`: `29`
- `action_dimension`: `29`
- locomotion config evidence: `observed`
- morphology asset visibility: `observed`
- remaining calibration blockers: `external_blocked`
- Phase 3.4 training manifest: `promotion_eligible=false`

This closes local structural gaps around morphology and learned-seam sockets. It
does not close provider/runtime/hardware evidence.

## Remaining gaps after this pass

| Gap | Status |
|-----|--------|
| Latency profile | external runtime/hardware evidence |
| Safety watchdog profile | external runtime/hardware evidence |
| Hardware joint-limit validation | external hardware evidence |
| Sim-real drift measurement | external provider/hardware evidence |
| GPU-backed training for 3.4 seams | deferred GPU/provider season |
| Benchmark promotion | deferred until real evaluation evidence exists |

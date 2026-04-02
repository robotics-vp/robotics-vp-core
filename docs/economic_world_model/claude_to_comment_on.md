# Claude Commentary Artifact

## Current Status

- **Date**: 2026-04-02
- **Branch**: `codex/multi-wm-architecture-plan`
- **Implementation center of gravity**: Phase 1 Sim / Synth / Physics WM closure
- **Active specs**:
  - `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
  - `docs/economic_world_model/codex_tranche_tier1_tier3_verification.md`
  - `docs/economic_world_model/phase1_closure_standard.md`
  - `docs/economic_world_model/doctrine_runtime_ladder_reuse.md`

This file is the single current-state handoff. Historical tranche detail belongs in:
- `docs/economic_world_model/progress_log.md`
- `docs/economic_world_model/implementation_notes.md`

## Tranche Spec Coverage

| Area | Current state |
|------|---------------|
| Tier 1 / Tier 3 internal closure surfaces | closed on the audited path |
| Repo-root host-reality scan | closed on the audited path |
| Public upstream root consumption without GPU | materially advanced on the audited path |
| Holosoma repo-local runtime/motion/policy/retargeting consumption | materially advanced on the audited path |
| Explicit-context-vs-autodiscovery precedence | closed on the audited path |

## What Was Fixed In This Pass

- Public upstream roots were pulled onto this host under `/Users/amarmurray/code` for the Phase-1 lanes the branch already knows how to consume:
  - `IsaacLab`
  - `unitree_sim_isaaclab`
  - `unitree_rl_gym`
  - `HumanoidVerse`
  - `xr_teleoperate`
  - `unitree_IL_lerobot`
  - `unitree_sdk2`
  - `unitree_models`
  - `holosoma`
- `src/world_model/sim_synth_physics/runtime_targets.py` and `src/world_model/sim_synth_physics/runtime_layouts.py` now derive Holosoma motion, policy, and retargeting subroots from a real local `holosoma` repo instead of requiring those surfaces to be re-entered manually as separate roots.
- `src/world_model/sim_synth_physics/runtime_layouts.py` now prefers actual Holosoma model/checkpoint surfaces over arbitrary demo-data `.pt` files, so local policy selection no longer confuses retargeting demo artifacts for runtime policy truth.
- `src/world_model/sim_synth_physics/adapters/holosoma_deployment.py` and `src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py` now treat repo-derived motion and retargeting surfaces as real local evidence. The branch no longer ignores those surfaces once the repo is present on disk.
- `src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py`, `src/world_model/sim_synth_physics/runtime_bundles.py`, and `src/world_model/sim_synth_physics/runtime_bridge.py` now make explicit deployment context outrank background autodiscovery. Local public clones add evidence, but they no longer silently steal profile or bridge selection from an explicit caller-provided context.

## What Was Not Changed

- No Perception / Grounding implementation surfaces
- No Embodiment / Actuation implementation surfaces outside the existing Phase-1 backend/runtime lane
- No frozen Phase B math or controller logic
- No fake readiness patch for missing Unitree calibration/watchdog assets

## Verified Local Evidence

Verified on the current host and codebase:

- `scripts/scan_phase1_runtime_layouts.py` runs successfully as a repo-root CLI.
- The scan emits explicit local summaries for both backend lanes.
- Downstream Phase 1 surfaces preserve:
  - usable/install-blocked profile truth
  - selected verified / partial target ids
  - selected policy / deploy / runtime-report refs and sources
  - launch missing preconditions
  - host-preflight missing, ready, and verified component sets

Current host scan after this pass:

- **Isaac/Unitree**
  - `usable_profiles = ['isaaclab_core', 'unitree_sim_isaaclab', 'unitree_rl_gym', 'humanoidverse', 'xr_teleoperate', 'unitree_model_assets']`
  - `selected_profile = unitree_sim_isaaclab`
  - `selected_policy_ref = /Users/amarmurray/code/unitree_rl_gym/deploy/pre_train/g1/motion.pt`
  - `selected_runtime_report = /Users/amarmurray/code/unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml`
  - `selected_verified_target_ids = ['unitree_sdk2_root', 'unitree_asset_root']`
- **Holosoma**
  - `usable_profiles = ['holosoma_repo', 'holosoma_motion_bank', 'holosoma_policy_bank', 'retargeting_bundle']`
  - `install_ready_profiles = ['holosoma_repo', 'holosoma_motion_bank', 'holosoma_policy_bank', 'retargeting_bundle']`
  - `selected_profile = holosoma_repo`
  - `selected_policy_ref = /Users/amarmurray/code/holosoma/src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx`
  - `selected_verified_target_ids = ['holosoma_motion_root']`
  - `host_preflight_status = preflight_ready`

## Explicitly Blocked External Truth

- **Isaac/Unitree**
  - upstream runtime roots and target roots are now real local evidence on this host
  - remaining host-preflight blockers are specific asset-contract surfaces:
    - `asset::unitree_robot_description`
    - `asset::whole_body_joint_map`
    - `asset::actuator_latency_profile`
    - `asset::joint_limit_profile`
    - `asset::safety_watchdog_profile`
  - these remain blocked because the branch still needs real asset-manifest/calibration/watchdog truth, not just runtime repos
- **Holosoma**
  - repo-local runtime, model, motion, and retargeting surfaces are now materially visible to the branch
  - the remaining honest blocker is no longer “missing repo roots”; it is meaningful runtime/install execution beyond repo-local evidence plus any richer deploy/report assets the eventual runtime needs
- **GPU-backed materialization**
  - GGDS / LDM / video materialization still requires the actual GPU/model/runtime lane, which is not present on this host

## Phase 1 Closure Assessment

| Finding | Category | Rationale |
|---------|----------|-----------|
| Holosoma repo-local motion/policy/retargeting surfaces were present on disk but not consumed as canonical local evidence | A -> closed | runtime-target/layout/pack/deployment now derive those subroots and use them |
| Holosoma policy selection could choose demo-data `.pt` artifacts as runtime policy | A -> closed | policy selection now prefers model/checkpoint surfaces rather than arbitrary demo files |
| Public local clones could silently outrank explicit Isaac deployment context | A -> closed | explicit deployment context now wins over background autodiscovery in deployment, runtime bundle, and bridge selection |
| Real Unitree robot asset-manifest/calibration/watchdog contracts | B | blocked by missing external asset-contract reality, not runtime-root discovery |
| Meaningful Isaac/Unitree execution and GPU-backed sim/materialization | B | blocked by absent runtime/GPU/provider reality |
| Meaningful Holosoma runtime execution beyond repo-local evidence | B | blocked by runtime/install/provider reality beyond what a shallow local repo clone alone proves |

Category A count: 0 (across the audited Phase 1 closure surfaces after the non-GPU host-consumption pass)
Category B count: 3
Category C unresolved: 0

Closure recommendation: **Phase 1 internal closure remains intact, and meaningful non-GPU progress is still possible when real local runtime roots/assets arrive.**

## Recommendation To Claude

- **Yes: meaningful Phase-1 progress still exists without a GPU.**
- The current best non-GPU move is still operational rather than architectural:
  1. keep consuming real local runtime roots/assets/checkpoints through the existing scan/binding path as they arrive
  2. provide or derive real Unitree asset-manifest surfaces so the remaining Isaac asset blockers collapse from generic `asset::...` gaps into verified local evidence
  3. keep Holosoma in the same lane if richer deploy/runtime-report/install truth appears
- Parallel Phase 2 preparation is now more justified, but it should remain secondary unless:
  - no more real Phase-1 runtime/assets are arriving soon
  - or the GPU-backed GGDS / LDM / video lane becomes the next real bottleneck

## Procedural Note

Keep this file as a single clean current-state artifact. When a new meaningful tranche lands:
- overwrite this file with the new current truth
- keep historical tranche detail in `progress_log.md`
- keep implementation detail in `implementation_notes.md`

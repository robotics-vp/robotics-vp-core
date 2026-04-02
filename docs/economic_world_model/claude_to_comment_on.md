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
| Public upstream root consumption without GPU | closed on the audited path |
| Holosoma repo-local runtime/motion/policy/retargeting consumption | closed on the audited path |
| Unitree public asset derivation / host rerun | closed on the audited path |
| Explicit-context-vs-autodiscovery precedence | closed on the audited path |

## What Was Fixed In This Pass

- Public upstream repos already present under `/Users/amarmurray/code` are now used for more than runtime-root discovery.
- `src/world_model/sim_synth_physics/asset_manifest.py` now derives honest Unitree asset surfaces from verified local public roots when explicit manifest values are absent or are only missing placeholder paths:
  - `unitree_robot_description`
  - `whole_body_joint_map`
  - `joint_limit_profile`
  - recommended:
    - `control_frequency_profile`
    - `teleop_recovery_contract`
- The derivation stays narrow and sourceable:
  - `unitree_models` for the primary G1 USD robot description
  - `HumanoidVerse` for the primary G1 joint-map / joint-limit config
  - `unitree_sim_isaaclab` for control-frequency evidence
  - `xr_teleoperate` for teleop-recovery evidence
- `src/world_model/sim_synth_physics/adapters/backend_isaac.py`, `src/world_model/sim_synth_physics/asset_contracts.py`, `src/world_model/sim_synth_physics/backend_runtime_execution.py`, and `scripts/scan_phase1_runtime_layouts.py` now thread the Isaac runtime-target contract into asset normalization, so the same derived asset truth shows up in:
  - backend binding
  - asset contracts
  - runtime materialization payloads
  - the repo-root host scan
- Missing explicit manifest paths no longer block a lane when a verified derived local file exists. Real explicit manifest refs still win.

## What Was Not Changed

- No Perception / Grounding implementation surfaces
- No Embodiment / Actuation implementation surfaces outside the existing Phase-1 backend/runtime lane
- No frozen Phase B math or controller logic
- No fake readiness patch for missing Unitree latency/watchdog assets

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
  - `verified_asset_count = 5`
  - verified/derived local asset surfaces now include:
    - `unitree_robot_description`
    - `whole_body_joint_map`
    - `joint_limit_profile`
    - `control_frequency_profile`
    - `teleop_recovery_contract`
  - `host_preflight_missing_components = ['asset::actuator_latency_profile', 'asset::safety_watchdog_profile']`
- **Holosoma**
  - `usable_profiles = ['holosoma_repo', 'holosoma_motion_bank', 'holosoma_policy_bank', 'retargeting_bundle']`
  - `install_ready_profiles = ['holosoma_repo', 'holosoma_motion_bank', 'holosoma_policy_bank', 'retargeting_bundle']`
  - `selected_profile = holosoma_repo`
  - `selected_policy_ref = /Users/amarmurray/code/holosoma/src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx`
  - `selected_verified_target_ids = ['holosoma_motion_root']`
  - `host_preflight_status = preflight_ready`

## Explicitly Blocked External Truth

- **Isaac/Unitree**
  - public repos materially reduced the blocked set
  - remaining host-preflight blockers are now only:
    - `asset::actuator_latency_profile`
    - `asset::safety_watchdog_profile`
  - I searched the now-local public repos specifically for latency/watchdog/safety artifacts after this pass
  - there are public control-frequency and soft-emergency-stop signals, but there is still no clean whole-body latency-contract or safety-watchdog artifact I would count as those required surfaces without overclaiming
- **Holosoma**
  - repo-local runtime, model, motion, and retargeting surfaces are now materially visible to the branch
  - the remaining honest blocker is meaningful runtime/install execution beyond repo-local evidence plus any richer deploy/report assets the eventual runtime needs
- **GPU-backed materialization**
  - GGDS / LDM / video materialization still requires the actual GPU/model/runtime lane, which is not present on this host

## Phase 1 Closure Assessment

| Finding | Category | Rationale |
|---------|----------|-----------|
| Holosoma repo-local motion/policy/retargeting surfaces were present on disk but not consumed as canonical local evidence | A -> closed | runtime-target/layout/pack/deployment now derive those subroots and use them |
| Holosoma policy selection could choose demo-data `.pt` artifacts as runtime policy | A -> closed | policy selection now prefers model/checkpoint surfaces rather than arbitrary demo files |
| Public local clones could silently outrank explicit Isaac deployment context | A -> closed | explicit deployment context now wins over background autodiscovery in deployment, runtime bundle, and bridge selection |
| Unitree public robot-description / joint-map / joint-limit surfaces were present in local public repos but not consumed as canonical asset truth | A -> closed | asset normalization now derives and verifies those surfaces from real local public roots |
| Real Unitree whole-body latency / watchdog contracts | B | blocked by missing external asset-contract reality, not by missing repo consumption or receipt plumbing |
| Meaningful Isaac/Unitree execution and GPU-backed sim/materialization | B | blocked by absent runtime/GPU/provider reality |
| Meaningful Holosoma runtime execution beyond repo-local evidence | B | blocked by runtime/install/provider reality beyond what a local repo clone alone proves |

Category A count: 0 (across the audited late-Phase-1 closure surfaces after the public-asset derivation pass)
Category B count: 3
Category C unresolved: 0

Closure recommendation: **On the audited late-Phase-1 surfaces, the remaining blockers are now honestly external. The useful non-GPU Isaac asset gap is narrow and explicit.**

## Recommendation To Claude

- Public repos **did** help materially:
  - they closed the non-GPU gap for robot description, joint-map, and joint-limit surfaces
  - they did **not** provide a clean required whole-body latency or watchdog contract
- The current best non-GPU move is now narrowly operational:
  1. if real `actuator_latency_profile` or `safety_watchdog_profile` artifacts appear, feed them through the existing scan/binding path
  2. if richer Holosoma deploy/runtime-report/install truth appears, feed that through the same path
  3. otherwise stop inventing more Phase-1-local structure and treat the remainder as external runtime/assets/GPU reality
- Parallel Phase 2 preparation is now justified, and implementation priority can begin preparing to shift when no new Phase-1 external runtime/assets are arriving soon.

## Procedural Note

Keep this file as a single clean current-state artifact. When a new meaningful tranche lands:
- overwrite this file with the new current truth
- keep historical tranche detail in `progress_log.md`
- keep implementation detail in `implementation_notes.md`

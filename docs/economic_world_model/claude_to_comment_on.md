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
| Compiler-side execution-contract closure (Tier 1.1/1.2) | closed on the audited path |
| Gen2Sim admission receipt emission (Tier 1.3) | closed on the audited path |
| Training corpus receipt-chain preservation (Tier 1.4) | closed on the audited path |
| Promotion/demotion machinery (Tier 3.2) | closed on the audited path |
| Branch planner fallback honesty (Tier 3.3) | closed on the audited path |
| Inferential yield scoring vs provenance/backend quality (Tier 3.4) | closed on the audited path |
| Randomization/calibration humanoid-axis completeness (Tier 3.5) | closed on the audited path |
| Shadow execution ladder threading (Tier 3.6) | closed on the audited path |
| Runtime layout/profile evidence | closed on the audited path |
| Upstream runtime-pack evidence | closed on the audited path |
| Selected-profile install/preflight truth | closed on the audited path |
| Selected-target install-shape truth | closed on the audited path |
| Policy-root/profile selection against real local installs | closed on the audited path |
| Checkpoint/report/deploy ref evidence selection | closed on the audited path |
| Selected-output validation against chosen runtime refs | closed on the audited path |
| Repo-root host-reality scan | closed on the audited path |
| Launch/work-order/training blocked-truth propagation | closed on the audited path |
| Render/provider lane (Tier 3.1) | verified sufficient on current path; no new gap found |

## What Was Fixed In This Pass

- `src/world_model/sim_synth_physics/runtime_launch.py` no longer strips `asset::...` host-preflight blockers out of launch readiness. A lane with missing verified robot-asset surfaces now stays honestly blocked instead of looking launch-ready.
- `src/world_model/sim_synth_physics/runtime_work_orders.py` now preserves more of the verified/blocked local host truth:
  - runtime-layout install-ready / install-partial / install-blocked profile groups
  - host-preflight ready / verified component sets
  - launch-specific missing preconditions and notes
- `src/world_model/sim_synth_physics/training_corpus.py` now preserves the same stronger local truth in backend-selector and branch-planner rows instead of flattening it into status-only metadata.
- The world-model tests that expected launch readiness now provide real temporary robot-asset files rather than relying on nonexistent paths. That removed a test-side pseudo-readiness assumption instead of weakening the new closure rule.

## Current Branch Truth

- Phase 1 Sim / Synth / Physics remains the active implementation center.
- The branch has not drifted upward into Perception / Grounding or Embodiment implementation.
- The late-Phase-1 closure surfaces now consistently distinguish:
  - verified local evidence
  - partial local evidence
  - symbolic or declared-only refs
  - explicit blocked truth
- Launch, work-order, training, and outcome surfaces now agree on blocked truth rather than preserving it in one receipt and softening it somewhere downstream.
- The host-reality scan now runs directly from repo root and produces an explicit local summary for both Isaac/Unitree and Holosoma lanes.

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

Host audit on this machine:

- no relevant Isaac/Unitree/Holosoma env vars are set
- no external `isaaclab`, `unitree_sdk2py`, or `holosoma` Python modules are importable
- no external Isaac/Unitree/Holosoma runtime roots were found in the common local clone directories the branch audits

## Explicitly Blocked External Truth

Current host scan says:

- **Isaac/Unitree**
  - `usable_profiles = []`
  - `binding_status = binding_blocked`
  - blocked on:
    - `policy_ref`
    - `launch_root`
    - `target::unitree_sdk2_root`
    - `target::unitree_asset_root`
    - `asset::unitree_robot_description`
    - `asset::whole_body_joint_map`
    - `asset::actuator_latency_profile`
    - `asset::joint_limit_profile`
    - `asset::safety_watchdog_profile`
- **Holosoma**
  - `usable_profiles = []`
  - `binding_status = binding_blocked`
  - blocked on:
    - `policy_ref`
    - `launch_root`
    - `target::holosoma_motion_root`
- **GPU-backed materialization**
  - GGDS / LDM / video materialization still requires the actual GPU/model/runtime lane, which is not present on this host

These are now honestly external blockers because the branch:
- can discover local installs when they exist
- can verify install shape instead of just path existence
- can prefer verified local refs over weaker candidates
- can carry blocked truth through launch, work-order, and training surfaces

## Phase 1 Closure Assessment

| Finding | Category | Rationale |
|---------|----------|-----------|
| Internal launch path could still ignore asset-side host-preflight blockers | A -> closed | launch readiness now consumes `asset::...` blockers instead of filtering them out |
| Work orders and trainer rows still flattened some local install/launch truth | A -> closed | downstream rows now preserve install-profile groups, host-preflight ready/verified components, and launch missing preconditions |
| Real Isaac / Unitree installs, assets, checkpoints, host setup | B | blocked by absent external runtime/provider reality on this host |
| Real Holosoma runtime, motion/policy/retargeting assets | B | blocked by absent external runtime/provider reality on this host |
| GPU-backed GGDS / LDM / video materialization | B | blocked by absent GPU/model/runtime reality |

Category A count: 0 (across the audited Phase 1 closure surfaces)
Category B count: 3
Category C unresolved: 0

Closure recommendation: **effectively at “all remaining blockers external” on the audited Phase 1 surfaces.**

## What Was Not Changed

- `src/world_model/sim_synth_physics/inferential.py`
- `src/world_model/sim_synth_physics/randomization.py`
- `src/world_model/sim_synth_physics/calibration.py`
- No Perception / Grounding implementation surfaces
- No Embodiment / Actuation implementation surfaces
- No frozen Phase B math or controller logic

## Recommendation To Claude

- **Phase 1 is now effectively at the point where the remaining audited blockers are external on this host.**
- Parallel Phase 2 preparation is now more justified, but it should still remain secondary until:
  - real Isaac/Unitree or Holosoma installs/assets/checkpoints land and need to be consumed
  - or the GPU-backed GGDS / LDM / video lane becomes available
- The next best move is therefore operational rather than architectural:
  1. bring real Isaac/Unitree runtime roots, SDK/assets, and checkpoints onto the host and rerun the scan/loop
  2. do the same for Holosoma runtime, motion, policy, and retargeting assets
  3. bring up the GPU-backed materialization lane and rerun the same closure checks

## Procedural Note

Keep this file as a single clean current-state artifact. When a new meaningful tranche lands:
- overwrite this file with the new current truth
- keep historical tranche detail in `progress_log.md`
- keep implementation detail in `implementation_notes.md`

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

This file is the **current-state handoff only**. Historical tranche detail belongs in:
- `docs/economic_world_model/progress_log.md`
- `docs/economic_world_model/implementation_notes.md`

## Current Branch Truth

The branch is still moving in the right direction.

- Phase 1 Sim / Synth / Physics remains the active implementation center.
- The branch has not drifted into premature Perception / Grounding implementation.
- The latest meaningful closure work has made the Phase 1 runtime/compiler surfaces more mechanically honest:
  - `PhysicsExecutionContract` is now canonical compiled state inside `SimSynthPhysicsWorldState`
  - the compiler emits a WM-owned `compiled_receipt_inventory` and `runtime_depth_projection`
  - training-corpus consumers preserve compiled contract and compiled ladder truth instead of flattening it away
- The latest runtime-evidence hardening pass made upstream runtime/profile/policy surfaces more concrete:
  - profile surfaces now carry primary refs and evidence density
  - Isaac asset truth distinguishes declared vs verified
  - Holosoma motion truth distinguishes existing vs missing motion sources

## What Changed Topologically

- Compiler-side backend routing is no longer only a runtime byproduct.
- The runtime ladder is now visible in both compiled-state truth and runtime evidence truth.
- Upstream runtime packs are no longer just root/candidate shaped; they now preserve:
  - primary deploy refs
  - primary policy refs
  - primary runtime-report refs
  - candidate counts
  - git metadata where a real local clone exists
- Training/export rows now preserve that upstream evidence instead of collapsing it into vague readiness bits.

## What Fake Readiness Was Removed

- Compiler closure no longer depended on runtime-only reconstruction of backend routing.
- Isaac upstream readiness no longer treats declared assets as equivalent to verified assets.
- Holosoma motion readiness no longer treats named motion sources as equivalent to locally present motion sources.
- The branch is more explicit now about the difference between:
  - compiled posture
  - runtime posture
  - upstream evidence posture
  - genuinely externalized runtime / asset / GPU blockers

## Tranche Spec Coverage

| Area | Current state |
|------|---------------|
| Tier 1 / Tier 3 compiler-side closure | **closed on the audited path** |
| Training-corpus preservation of compiled/runtime truth | **materially closed on the audited path** |
| Upstream runtime/profile/policy evidence specificity | **materially improved** |
| Isaac declared-vs-verified asset truth | **materially improved** |
| Holosoma existing-vs-missing motion truth | **materially improved** |
| Render/provider receipt chain | **not reworked in the latest pass** |
| Promotion/demotion history surface | **still secondary / not the latest target** |

## What Was Not Changed

This handoff does **not** claim new closure in:

- `src/world_model/sim_synth_physics/render_providers.py`
- `src/world_model/sim_synth_physics/promotion.py`
- any Perception / Grounding implementation surface
- any frozen Phase B math or controller logic

## Phase 1 Closure Assessment

### Category A: internal incompleteness that should still be fixable in-repo

No obvious Category A break is currently claimed in the latest audited compiler/runtime/evidence cluster.

That does **not** mean global Phase 1 closure. It means the latest audited cluster is no longer blocked by an obvious missing receipt/state/contract gap.

### Category B: honest externalized remainder

- real Isaac / Unitree upstream runtime, assets, checkpoints, and host install
- real Holosoma host/runtime, motion corpora, retargeting assets, and policies
- real GPU-backed GGDS / LDM / video materialization
- benchmark density from actual backend execution against those real substrates

### Category C: non-blocking secondary refinement

- additional promotion/demotion provenance depth
- more outcome-density / benchmark-density once Category B substrates are present

## Explicit Internal vs External Statement

### Internal incompleteness fixed in the latest meaningful passes

- compiler-side canonicalization of `PhysicsExecutionContract`
- compiler-side projection of deeper runtime-ladder truth
- training/export preservation of compiled closure
- richer upstream runtime/profile/policy evidence preservation
- declared-vs-verified Isaac asset truth
- existing-vs-missing Holosoma motion-source truth

### What remains internal

- no fresh Category A gap is being claimed from the latest audited cluster
- there is still room for more density and stronger evidence, but not an obvious missing canonical surface in the latest reviewed path

### What is now honestly externalized

- actual upstream runtime availability
- actual host installs and assets
- actual checkpoints/policies/runtime reports
- actual GPU/model availability for GGDS / LDM / video materialization

## Recommendation to Claude

- **Phase 1 remains the active implementation center.**
- **Parallel Perception prep is allowed but secondary.**
- Do **not** treat the current audited-cluster closure as total Phase 1 closure.
- The next highest-leverage Phase 1 work should keep consuming the richer upstream evidence surfaces against real local/runtime/asset reality for:
  1. Isaac / Unitree
  2. Holosoma

## Procedural Note

Keep this file as a single clean current-state artifact. When a new meaningful tranche lands:
- overwrite this file with the new current truth
- keep historical tranche detail in `progress_log.md`
- keep implementation detail in `implementation_notes.md`

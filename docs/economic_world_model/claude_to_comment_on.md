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
| Runtime layout/profile evidence | **materially improved** |
| Upstream runtime-pack evidence | **materially improved** |
| Selected-profile install/preflight truth | **closed on the audited path** |
| Training/work-order preservation of install truth | **materially improved** |
| Holosoma selected-profile false blocker | **fixed** |
| Isaac partially discovered preferred-profile fallback | **fixed** |
| Promotion/demotion history surface | **unchanged / secondary** |
| Render/provider lane | **unchanged in this tranche** |

## Current Branch Truth

- Phase 1 Sim / Synth / Physics remains the active implementation center.
- The branch has not drifted upward into Perception / Grounding implementation.
- The latest meaningful closure work did **not** add a new runtime rung. It tightened the existing runtime layout -> runtime pack -> runtime binding path.
- Runtime profiles now carry explicit install/preflight truth, not only root/candidate truth:
  - selected install entrypoint paths
  - matched/missing entrypoints
  - primary entrypoint ref
  - install-preflight status
  - install missing/verified components
- Isaac/Unitree and Holosoma upstream runtime packs now preserve that profile-local install truth and expose it by profile id.
- Runtime bindings now resolve install/preflight truth against the **actually selected profile**, not just the pack’s preferred profile.
- Training-corpus rows and runtime work orders now preserve that install/preflight truth instead of flattening it away.

## What Changed Topologically

- Runtime layouts are no longer only “root exists + candidate refs” surfaces.
- Upstream runtime packs are no longer only preferred-profile summaries; they now carry reusable per-profile install truth.
- Runtime bindings no longer inherit profile-install blockers from the wrong profile when the selected mode changes.
- The Holosoma motion-train lane can now honestly select `holosoma_motion_bank` without inheriting `holosoma_repo` install gaps like `profile_entrypoint`.
- Partially discovered Isaac/Unitree upstream profiles can now still be selected as the best local upstream profile instead of collapsing to an empty preferred profile when deployment-level readiness stays strict.

## What Fake Readiness Was Removed

- “Repo root exists” is no longer treated as enough install truth for a selected runtime profile.
- “Pack preferred profile is blocked” no longer automatically means “selected binding profile is blocked.”
- Holosoma local motion-train no longer looks blocked by repo-entrypoint gaps that belong only to a different selected profile.
- Training/work-order consumers no longer have to infer selected-profile install truth from pack status alone.

## What Was Not Changed

This tranche did **not** claim new closure in:

- `src/world_model/sim_synth_physics/render_providers.py`
- `src/world_model/sim_synth_physics/promotion.py`
- any Perception / Grounding implementation surface
- frozen Phase B math or controller logic

## Phase 1 Closure Assessment

### Category A: internal incompleteness that should still be fixable in-repo

No fresh Category A gap is being claimed on the audited runtime layout/pack/binding/install cluster.

That does **not** mean total Phase 1 closure. It means this specific install/preflight cluster is now structurally honest on the audited path.

### Category B: honest externalized remainder

- real Isaac / Unitree installs, assets, checkpoints, and host setup
- real Holosoma host/runtime, motion corpora, retargeting assets, and policies
- real GPU-backed GGDS / LDM / video materialization
- benchmark density from actual backend execution on those real substrates

### Category C: non-blocking secondary refinement

- deeper promotion/demotion provenance
- denser benchmark and outcome evidence once Category B substrates are real

## Explicit Internal vs External Statement

### Internal incompleteness fixed today

- profile-level install/preflight evidence was added to runtime layouts
- upstream runtime packs now preserve profile-local install truth rather than only preferred-profile truth
- bindings now resolve install truth against the actually selected profile
- runtime work orders and trainer rows now preserve that selected-profile install truth
- the Holosoma motion-train selected-profile blocker leak was removed
- the Isaac partially discovered preferred-profile collapse was removed

### What remains internal

- no fresh obvious Category A gap is being claimed in the audited install/preflight cluster
- there is still room for more density, but not an obvious missing canonical install/preflight surface on this path

### What is now honestly externalized

- whether real local Isaac/Unitree upstream repos/assets/checkpoints are present and usable
- whether real local Holosoma runtime/motion/policy/retargeting assets are present and usable
- whether GPU/model runtime is available for GGDS / LDM / video materialization

## Recommendation to Claude

- **Phase 1 remains the active implementation center.**
- **Parallel Perception prep is allowed but secondary.**
- Do **not** treat this audited-cluster closure as total Phase 1 closure.
- The next highest-leverage Phase 1 work should keep pushing Category B reality:
  1. strengthen local install/preflight evidence against actual discovered Isaac/Unitree clones/assets/checkpoints
  2. do the same for real Holosoma host/runtime/motion/policy/retargeting assets
  3. keep preserving that truth through launch/work-order/training surfaces without inventing a new ladder

## Procedural Note

Keep this file as a single clean current-state artifact. When a new meaningful tranche lands:
- overwrite this file with the new current truth
- keep historical tranche detail in `progress_log.md`
- keep implementation detail in `implementation_notes.md`

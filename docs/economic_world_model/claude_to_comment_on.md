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
| Runtime layout/profile evidence | materially improved |
| Upstream runtime-pack evidence | materially improved |
| Selected-profile install/preflight truth | closed on the audited path |
| Selected-target install-shape truth | materially closed on the audited path |
| Policy-root/profile selection against real local installs | materially closed on the audited path |
| Runtime-layout usable-profile propagation | materially closed on the audited path |
| Checkpoint/report/deploy ref evidence selection | materially closed on the audited path |
| Selected-output validation against chosen runtime refs | materially closed on the audited path |
| Inferential yield scoring vs provenance/backend quality (Tier 3.4) | closed on the audited path |
| Randomization/calibration humanoid-axis completeness (Tier 3.5) | closed on the audited path |
| Promotion/demotion machinery (Tier 3.2) | Category A gap closed |
| Shadow execution ladder threading (Tier 3.6) | Category A gap closed on audited path |
| Branch planner fallback honesty (Tier 3.3) | materially closed on audited path |
| Render/provider lane (Tier 3.1) | verified sufficient on current path; no new gap found |
| Compiler-side execution-contract closure (Tier 1.1/1.2) | closed on the audited path |
| Gen2Sim admission receipt emission (Tier 1.3) | closed on the audited path |
| Training corpus receipt-chain preservation (Tier 1.4) | materially improved and audited on current path |

## Current Branch Truth

- Phase 1 Sim / Synth / Physics remains the active implementation center.
- The branch has not drifted upward into Perception / Grounding implementation.
- Runtime targets no longer stop at `path exists` on the selected-target path:
  - bindings now distinguish verified selected targets from partial selected targets
  - host preflight can fail on install-shape truth even when a root exists
  - work orders and trainer rows preserve that distinction instead of flattening it
- Deployment and runtime-pack readiness now use usable profiles and verified targets instead of raw existing roots:
  - install-blocked profiles no longer count as runtime-ready just because the repo root exists
  - verified targets, not just existing target paths, now drive deployment/runtime-pack readiness
  - empty explicit policy roots no longer outrank discovered runtime roots that actually contain checkpoints
- Runtime-layout contracts now expose `usable_profiles` directly, and bundle/bridge/work-order/trainer paths preserve that stronger truth instead of forcing downstream consumers to reconstruct it from weaker `ready_profiles` semantics.
- Runtime packs and bindings now prefer verified local checkpoint / deploy-config / runtime-report refs over earlier missing candidates, and they preserve both the chosen ref source and candidate-evidence summaries instead of flattening selection back into first-candidate ordering.
- Runtime output contracts and outcome receipts now validate harvested outputs against the selected policy / deploy-config / runtime-report refs, so “runtime outputs harvested” also says whether the outputs actually align with the chosen runtime artifacts.
- Shadow execution consumes selected runtime-binding truth instead of only carrying runtime-ladder metadata in the receipt.
- Branch plans and trainer rows explicitly distinguish:
  - learned payload applied
  - learned trace present but heuristic retained
  - heuristic retained because of demotion or helper unavailability
- Promotion now has a real demotion path, so a once-promoted helper cannot stay promoted forever when evidence degrades.

## What Changed Topologically

- No new WM, runtime rung, or speculative abstraction was introduced.
- Runtime-target contracts now carry install-shape verification metadata:
  - `verification_status`
  - `verified`
  - `matched_markers`
  - `missing_markers`
  - `primary_marker_ref`
- Isaac and Holosoma runtime bindings now preserve:
  - `selected_verified_target_ids`
  - `selected_partial_target_ids`
  - selected-target evidence with install-shape truth
- Runtime work orders and trainer exports now keep that selected-target truth, so downstream consumers no longer need to infer whether a root was merely named or actually install-shaped enough to trust.
- Policy contracts now preserve selected-root truth over multiple candidate roots, so an explicit-but-empty policy root can no longer hide a discovered local runtime root that actually contains checkpoints.
- Deployment contracts and upstream runtime packs now distinguish:
  - usable profiles
  - install-blocked profiles
  - verified targets
  - merely existing targets
  rather than collapsing all of that into root-exists posture.
- Runtime bundles, bridge receipts, work orders, and trainer rows now also preserve `runtime_layout_usable_profiles`, so the stronger profile truth survives into execution-facing and training-facing artifacts.
- Upstream runtime packs now also preserve candidate-evidence summaries plus the source of the chosen primary policy / deploy / runtime-report ref, and runtime bindings preserve the selected ref source, so “why this exact ref was chosen” is replayable instead of implicit.
- Runtime outcome receipts now also preserve `selected_ref_validation`, and work-order / trainer surfaces keep that status so selected-runtime mismatch truth survives beyond the raw harvested artifact list.

## What Fake Readiness Was Removed

- Empty SDK, asset, motion, and retargeting roots no longer look launch-ready just because the directory exists.
- Selected-target preflight no longer quietly inherits a weaker “path exists” notion when the stronger install-shape evidence says the target is still partial.
- Holosoma and Isaac selected-target rows no longer collapse verified and merely declared targets into one readiness class inside work orders or trainer exports.
- Explicit-but-empty policy roots no longer make a lane look more real than a discovered runtime root with actual checkpoints.
- Install-blocked runtime profiles no longer count as deployable just because the repo root exists.
- Downstream Phase 1 consumers no longer have to treat `ready_profiles` as if it already meant “usable profile”; that distinction is now explicit and replayable.
- A missing first candidate can no longer outrank a later verified local checkpoint or runtime report just because it appeared earlier in a list.
- A harvested runtime output set can no longer look fully satisfactory without also saying whether it matched the selected policy/report surfaces the lane actually intended to use.

## What Was Not Changed

- `src/world_model/sim_synth_physics/inferential.py`
- `src/world_model/sim_synth_physics/randomization.py`
- `src/world_model/sim_synth_physics/calibration.py`
- No Perception / Grounding implementation surfaces
- No frozen Phase B math or controller logic

## Phase 1 Closure Assessment

| Finding | Category | Rationale |
|---------|----------|-----------|
| Promotion had no demotion path | A -> closed | Evidence-based demotion is implemented in all helper resolvers |
| Shadow execution bypassed the deeper runtime-binding ladder | A -> closed | Shadow env/work-order artifacts now consume selected runtime-binding truth |
| Branch planner fallback was traceable but not explicit | A -> closed on audited path | Branch plans and trainer rows now state whether learned payloads were applied or only traced |
| Selected-target runtime roots could look ready from path existence alone | A -> closed on audited path | bindings/preflight now consume install-shape verification instead of path existence alone |
| Install-blocked profiles and empty explicit policy roots could still overstate readiness | A -> closed on audited path | deployment/runtime-pack selection now uses usable profiles, verified targets, and real checkpoint-bearing roots |
| Usable-profile truth was still being reconstructed ad hoc downstream | A -> closed on audited path | runtime-layout contracts, bundles, bridge receipts, work orders, and trainer exports now preserve it explicitly |
| Checkpoint/report selection still depended on first-candidate ordering in runtime packs and bindings | A -> closed on audited path | primary refs now prefer verified local artifacts and preserve candidate-evidence/source truth |
| Harvested runtime outputs did not say whether they matched the selected runtime refs | A -> closed on audited path | output contracts/outcome receipts now validate selected policy/deploy/report refs against harvested artifacts |
| Real Isaac / Unitree installs, assets, checkpoints | B | Remaining blocker is external host/runtime/asset reality |
| Real Holosoma runtime, motion/policy/retargeting assets | B | Remaining blocker is external host/runtime/asset reality |
| GPU-backed GGDS / LDM / video materialization | B | Remaining blocker is GPU/model/runtime availability |
| Inferential yield scoring vs backend fidelity (Tier 3.4) | A -> closed on audited path | direct verification now covers priority/provenance weighting, agenda score uplift, and branch confidence reaction to backend provenance flags |
| Randomization/calibration humanoid-axis completeness (Tier 3.5) | A -> closed on audited path | direct verification now covers humanoid randomization axes, adaptation policy derivation, and calibration/adaptation reaction to route status plus runtime evidence |

Category A count: 0 (on audited items)
Category B count: 3
Category C unresolved: 0

Closure recommendation: **not yet closed**. The explicit Tier 3.4 / 3.5 verification gap is now closed on the audited path, and the practical remainder is increasingly the real external runtime/asset/GPU blocker set rather than internal Phase-1-local ambiguity.

## Recommendation to Claude

- **Phase 1 remains the active implementation center.**
- Parallel Perception prep is allowed but secondary.
- The next highest-leverage Phase 1 work is still Category B burn-down through honest external consumption:
  1. keep consuming real Isaac/Unitree local install/asset/checkpoint reality
  2. keep doing the same for Holosoma runtime/motion/policy/retargeting assets
  3. keep converting remaining “ready in contract only” external-runtime claims into concrete host/runtime/asset evidence or explicit blocked truth

## Procedural Note

Keep this file as a single clean current-state artifact. When a new meaningful tranche lands:
- overwrite this file with the new current truth
- keep historical tranche detail in `progress_log.md`
- keep implementation detail in `implementation_notes.md`

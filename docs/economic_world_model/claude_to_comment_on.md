# Claude Commentary Artifact

## Current Status

- **Tranche**: Phase 1 Sim / Synth / Physics WM compiler-side closure pass
- **Date**: 2026-04-02
- **Branch**: `codex/multi-wm-architecture-plan`
- **Implementation center of gravity**: Phase 1 Sim / Synth / Physics WM closure
- **Active specs**:
  - `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
  - `docs/economic_world_model/codex_tranche_tier1_tier3_verification.md`
  - `docs/economic_world_model/phase1_closure_standard.md`

## What Was Implemented

- Canonicalized `PhysicsExecutionContract` into compiled world state:
  - `src/world_model/sim_synth_physics/state.py`
  - `src/world_model/sim_synth_physics/compiler.py`
  - `src/world_model/sim_synth_physics/runtime.py`
- The compiler now emits a WM-owned compiled receipt inventory and runtime-depth projection:
  - includes compiler-owned receipts, runtime-owned receipts, per-branch receipts, and a pre-runtime ladder projection
  - includes route status, binding status, bridge status, runtime target readiness, deployment-ready modes, ready profiles, policy readiness, and upstream runtime-pack status
- `SimSynthPhysicsRuntime.compile_world_state(...)` now compiles the world state with the runtime fallback backend so the compiled execution contract matches the runtime posture
- `SimSynthPhysicsRuntime.execute_world_state(...)` now reuses the compiled execution contract instead of rebuilding it when the world state already carries it
- `training_corpus.py` now preserves the new compiled contract and compiled receipt inventory in harvested bundles and trainer rows:
  - `physics_execution_contract`
  - `compiled_receipt_inventory_id`
  - projected binding / bridge / pack status

## What Changed Topologically

- `PhysicsExecutionContract` is no longer only a runtime-owned artifact. It is now part of `SimSynthPhysicsWorldState`, which makes backend routing a canonical compiled-state surface instead of a later runtime reconstruction.
- The compiler no longer stops at isolated state objects plus implied runtime depth. It now produces an explicit compiler-owned receipt inventory describing:
  - what receipts the compiled state can produce
  - what runtime receipts are expected later
  - which deeper runtime-ladder surfaces are already visible pre-runtime
- Training-corpus consumers no longer need to infer that compiler-side closure happened only from runtime receipts. They now preserve compiled route and compiled ladder truth directly.

## What Fake Readiness Was Removed

- The compiler no longer looked “complete” while still hiding backend routing in runtime-only logic.
- The compiled world state no longer suggested a flat backend binding when the deeper pre-runtime ladder truth existed only in scattered metadata.
- Training rows no longer flatten the new compiler-side closure away by ignoring the compiled contract and compiled receipt inventory.

## What Is Still Only Contract-Shaped

- Real Isaac / Isaac Lab / Unitree runtime, assets, checkpoints, and host bring-up remain external.
- Real Holosoma host/runtime, motion corpora, retargeting assets, and policies remain external.
- Real GGDS / LDM / GPU-backed materialization remains external.
- Concrete runtime-binding selected-profile / selected-launch-root truth still becomes richer after runtime bundle / launch preparation; the compiler now projects the pre-runtime ladder honestly, but it does not fabricate post-launch truth.

## Tranche Spec Coverage

| Item | Result | Notes |
|------|--------|-------|
| 1.1 Compiler full-artifact assembly audit | **fixed on audited gap** | Compiler now assembles and preserves `PhysicsExecutionContract` plus a compiler-owned receipt inventory and runtime-depth projection. |
| 1.2 Physics contracts execution binding completeness | **fixed today** | `PhysicsExecutionContract` is now compiled into `SimSynthPhysicsWorldState`, routed with the configured fallback backend, and available to downstream consumers without runtime-only reconstruction. |
| 1.3 Gen2Sim admission explicit receipt emission | **already closed** | No new change needed in this pass. |
| 1.4 Training corpus receipt consumption | **fixed on audited gap** | Corpus harvest now preserves the compiled execution contract and compiled receipt inventory instead of flattening the compiler-side closure away. |
| 3.1 Render provider contract chains | **passed on audited scope** | No new internal gap found in this pass. |
| 3.2 Promotion / demotion machinery | **partial but not blocking in this pass** | No separate historical demotion receipt was added; current bounded posture remains unchanged. |
| 3.3 Branch planner routing completeness | **passed on audited scope** | No new internal gap found in this pass. |
| 3.4 Inferential yield scoring | **passed on audited scope** | No new internal gap found in this pass. |
| 3.5 Randomization / calibration reaction to evidence | **partial / mostly external remainder** | Interfaces are present; remaining realism depends on concrete runtime evidence and assets. |
| 3.6 Shadow execution receipt completeness | **already improved** | No new shadow-execution code was needed in this pass. |
| 4.2 Compiler round-trip test | **strengthened today** | World-state tests now verify compiled execution-contract and receipt-inventory round-trip. |

## What Was Not Changed

These audited files were intentionally not changed in this tranche:

- `src/world_model/sim_synth_physics/render_providers.py`
- `src/world_model/sim_synth_physics/promotion.py`
- `src/world_model/sim_synth_physics/branch_planner.py`
- `src/world_model/sim_synth_physics/inferential.py`
- `src/world_model/sim_synth_physics/randomization.py`
- `src/world_model/sim_synth_physics/calibration.py`
- `src/world_model/sim_synth_physics/runtime_outcome_parsers.py`

## Phase 1 Closure Assessment

| Finding | Category | Rationale |
|---------|----------|-----------|
| Compiler did not carry `PhysicsExecutionContract` as canonical compiled state | **resolved A** | Closed today by compiling the contract into `SimSynthPhysicsWorldState` and reusing it in runtime execution. |
| Compiler-side state did not reflect deeper runtime-binding depth directly | **resolved A** | Closed today on the audited path via compiler-owned receipt inventory plus runtime-depth projection. |
| Training corpus flattened compiler-side closure truth | **resolved A** | Closed today by harvesting the compiled execution contract and compiled receipt inventory into trainer rows. |
| Real Isaac / Unitree runtime, assets, checkpoints, and host bring-up are absent | **B** | The WM now names this truth explicitly; the blocker is external runtime/assets availability. |
| Real Holosoma host/runtime, motion/retargeting assets, and policies are absent | **B** | The ladder is wired; the provider/runtime substrate is still external. |
| Real GGDS / LDM / GPU materialization is absent | **B** | The provider and receipt path are wired; concrete execution still depends on GPU/model/runtime availability. |
| Promotion lacks a separate historical demotion receipt | **C→B** | Current helper resolution already returns to bounded non-promoted posture; remaining gap is history/provenance, not structural Phase-1 dishonesty. |

Category A count: 0
Category B count: 4
Category C unresolved: 0

Closure recommendation: **parallel Perception prep allowed, but Phase 1 should remain the active implementation center until real-runtime/asset evidence is stronger**

## Explicit Internal vs External Statement

### Internal incompleteness fixed today

- compiler-side canonicalization of `PhysicsExecutionContract`
- compiler-side projection of deeper runtime-ladder truth into compiled state
- trainer/export preservation of that compiler-side closure

### What remains internal

- no new audited Category A finding remains in this compiler-side cluster
- the remaining internal work is now narrower and judgment-based, not an obvious missing receipt/state break in the audited path

### What is now honestly externalized

- real Isaac / Unitree runtime bring-up, assets, checkpoints, and host installation
- real Holosoma host/runtime, motion/retargeting assets, and policies
- real GGDS / LDM / GPU materialization
- benchmark density that depends on actual backend execution evidence

## Tests and Verification

Targeted verification run:

```text
python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q
python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py
python3 -m pytest -q tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py
git diff --check
```

Result:
- `26 passed`

## Recommendation to Claude

- **Phase 1 remains the active implementation center**, but it is now much closer to the “honestly external” boundary for the audited compiler/runtime surfaces.
- **Parallel Perception prep is allowed but secondary.** The compiler-side Category A cluster is closed on the audited path.
- The next highest-leverage Phase 1 cut is:
  1. use real upstream runtime / asset / checkpoint evidence to harden the concrete Isaac / Unitree lane against the now-compiled closure surfaces
  2. bring Holosoma runtime execution and outcome density up to the same evidence standard

# Claude Commentary Artifact

## Current Status

- **Tranche**: Phase 1 Sim / Synth / Physics WM Tier 1 / Tier 3 verification pass
- **Date**: 2026-04-02
- **Branch**: `codex/multi-wm-architecture-plan`
- **Implementation center of gravity**: Phase 1 Sim / Synth / Physics WM closure
- **Active specs**:
  - `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
  - `docs/economic_world_model/codex_tranche_tier1_tier3_verification.md`
  - `docs/economic_world_model/phase1_closure_standard.md`

## What Was Implemented

- Added a typed `Gen2SimAdmissionReceipt` and threaded it through the Phase 1 runtime/export path:
  - `src/world_model/sim_synth_physics/receipts.py`
  - `src/world_model/sim_synth_physics/gen2sim_admission.py`
  - `src/world_model/sim_synth_physics/runtime.py`
  - `src/world_model/sim_synth_physics/training_corpus.py`
- Updated shadow execution so it now carries the deeper runtime-ladder truth instead of jumping directly from `PhysicsExecutionContract` to shadow materialization:
  - `src/world_model/sim_synth_physics/shadow_execution.py`
  - shadow receipts now preserve runtime execution / adapter / launch / outcome ids and statuses plus `shadow_harvest_mode`
- Tightened training-corpus preservation for the branch-planner lane:
  - branch rows now keep gen2sim, adaptation, calibration, and shadow receipt truth instead of flattening to render/runtime only
- Added focused verification coverage:
  - `tests/test_sim_synth_branch_helpers.py`
  - `tests/test_sim_synth_training_corpus.py`
  - `tests/test_sim_synth_physics_world_model.py`

## What Changed Topologically

- `gen2sim` admission is no longer only a state object inside the compiled WM state. It now has a typed receipt that survives runtime execution, artifact emission, and trainer-facing harvest.
- Shadow execution is no longer a side path that ignores the deeper runtime ladder. It now carries forward:
  - runtime execution posture
  - adapter mediation posture
  - adapter realization posture
  - launch posture
  - outcome harvest posture
- Branch-planner export is no longer the “flattened” receipt consumer compared with backend-selector export. It now carries the same lower-WM honesty for:
  - gen2sim admission
  - adaptation
  - calibration
  - shadow execution

## What Fake Readiness Was Removed

- `gen2sim` admission no longer terminates at state-only truth with no typed receipt chain.
- Shadow execution no longer looks like an independent backend preview that can ignore the deeper runtime lane.
- Branch-planner trainer rows no longer look more complete than they are by omitting calibration / adaptation / shadow / gen2sim context while still consuming deeper runtime artifacts.

## What Is Still Only Contract-Shaped

- Real Isaac / Isaac Lab / Unitree runtime, assets, checkpoints, and host bring-up remain external.
- Real Holosoma host/runtime, motion corpora, retargeting assets, and policies remain external.
- Real GGDS / LDM / GPU-backed materialization remains external.
- Compiler-side runtime depth is still not fully reflected as canonical compiled state; the deeper runtime-binding truth still becomes explicit mainly in runtime artifacts rather than in the compiled world state itself.

## Tranche Spec Coverage

| Item | Result | Notes |
|------|--------|-------|
| 1.1 Compiler full-artifact assembly audit | **partial** | Compiler assembles the documented Phase 1 state objects, but compiled state still stops short of carrying `PhysicsExecutionContract` and deeper runtime-binding depth as canonical compiled artifacts. |
| 1.2 Physics contracts execution binding completeness | **gap found** | `PhysicsExecutionContract` is still built at runtime rather than carried inside `SimSynthPhysicsWorldState`. |
| 1.3 Gen2Sim admission explicit receipt emission | **fixed today** | Added `Gen2SimAdmissionReceipt`; runtime, artifacts, and training-corpus harvest now preserve it. |
| 1.4 Training corpus receipt consumption | **fixed on audited gap** | Branch-planner rows now preserve gen2sim / adaptation / calibration / shadow truth; backend-selector rows already carried most of this. |
| 3.1 Render provider contract chains | **passed on audited scope** | `RenderProviderReceipt` already exists via materialization; no additional provider-specific receipt rung was required. |
| 3.2 Promotion / demotion machinery | **partial but not blocking in this pass** | Current helper resolution still returns `shadow_candidate` whenever benchmark readiness is absent; no separate historical demotion receipt was added today. |
| 3.3 Branch planner routing completeness | **passed on audited scope** | Heuristic fallback posture remains visible through branch metadata and generation mode; no new gap found today. |
| 3.4 Inferential yield scoring | **passed on audited scope** | No new internal receipt/contract gap found in this pass. |
| 3.5 Randomization / calibration reaction to evidence | **partial / mostly external remainder** | Calibration already reacts to runtime evidence; remaining realism depends on concrete runtime evidence and assets. |
| 3.6 Shadow execution receipt completeness | **fixed today** | Shadow receipts now thread the deeper runtime lane and distinguish preview vs harvested posture. |
| 4.2 Compiler round-trip test | **added partial coverage** | Added a world-state `to_dict()` round-trip test for the core Phase 1 state surface. |

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
| `gen2sim` admission had no typed receipt chain | **resolved A** | This was internal incompleteness and is now closed by `Gen2SimAdmissionReceipt` plus runtime/export wiring. |
| Shadow execution bypassed deeper runtime-ladder truth | **resolved A** | This was internal inconsistency between Tier 2 and Tier 3 and is now closed on the audited path. |
| Branch-planner corpus rows flattened the deeper receipt chain | **resolved A** | This was internal trainer/export incompleteness and is now closed on the audited path. |
| Compiler does not carry `PhysicsExecutionContract` as canonical compiled state | **A** | Still internal; no external dependency blocks adding or threading this artifact. |
| Compiler-side state does not yet reflect the deeper runtime-binding depth directly | **A** | Still internal; runtime artifacts are honest, but compiled-state closure is not fully complete. |
| Render providers lack a separate provider-module-local receipt | **C→B** | Existing `RenderProviderReceipt` emitted by materialization appears sufficient; no internal receipt break was found on the audited path. |
| Promotion lacks a separate historical demotion receipt | **C→B** | Current helper resolution already demotes back to `shadow_candidate` when benchmark readiness is absent; the remaining gap is more about explicit history than structural honesty. |
| Concrete Isaac / Holosoma runtime, assets, checkpoints, and GPU-backed render execution are absent | **B** | These are now honestly externalized provider/runtime/asset blockers. |

Category A count: 2
Category B count: 3
Category C unresolved: 0

Closure recommendation: **not closed yet; parallel Perception prep is allowed but secondary**

## Explicit Internal vs External Statement

### Internal incompleteness fixed today

- `gen2sim` receipt-chain completeness
- shadow-execution receipt-chain honesty relative to the deeper runtime ladder
- branch-planner receipt preservation for gen2sim / adaptation / calibration / shadow truth

### What remains internal

- compiler-side canonicalization of `PhysicsExecutionContract`
- compiler-side propagation of deeper runtime-binding truth as compiled canonical state rather than only runtime artifacts

### What is now honestly externalized

- real Isaac / Unitree runtime bring-up, assets, and checkpoints
- real Holosoma host/runtime, motion/retargeting assets, and policies
- real GGDS / LDM / GPU materialization
- benchmark density that depends on actual backend execution evidence

## Tests and Verification

Targeted verification run:

```text
python3 -m compileall src/world_model/sim_synth_physics tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py -q
python3 -m ruff check src/world_model/sim_synth_physics tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py
python3 -m pytest -q tests/test_sim_synth_branch_helpers.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py
python3 -m pytest -q tests/test_sim_synth_physics_scripts.py tests/test_sim_synth_runtime_launch.py
git diff --check
```

Result:
- `30 passed`
- `10 passed`

## Recommendation to Claude

- **Phase 1 remains the active implementation center.**
- **Parallel Perception prep is allowed but secondary** because Category A count is down to a narrow compiler-side cluster.
- The next highest-leverage Phase 1 cut is:
  1. make `PhysicsExecutionContract` a canonical compiled-state artifact or equivalent compiler-owned receipt surface
  2. thread deeper runtime-binding truth into the compiled state / compiler-facing receipt inventory rather than only the runtime result

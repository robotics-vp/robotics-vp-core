# Claude Commentary Artifact

## Current Status

- **Tranche**: Sim / Synth / Physics WM closure, Phase 1 Isaac/Unitree adapter-realization tranche
- **Date**: 2026-04-01
- **Branch**: `codex/multi-wm-architecture-plan`
- **Active tranche spec**: `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
- **Held tranche spec**: `docs/economic_world_model/codex_tranche_perception_wm_schema.md`

## What Was Implemented

- Added `src/world_model/sim_synth_physics/adapters/isaac_unitree_adapter_realization.py`.
- The Isaac/Unitree lane now has a typed realization surface over:
  - executable-adapter request
  - executable-adapter consumer
  - adapter-execution mediation
- `backend_runtime_execution.py` now rebuilds that realization after execution mediation is finalized and preserves it in live runtime artifacts and metadata.
- `runtime.py` now writes `backend_runtime_adapter_realization.json` as a root-level loop artifact.
- `training_corpus.py` now carries adapter realization path/status into backend-selector and branch-planner rows.
- `scripts/run_isaac_unitree_executable_adapter.py` now emits `adapter_realization` alongside the existing request / consumer / execution / launch payloads.

## What Changed Topologically

- No new WM boundary was introduced.
- The Isaac/Unitree runtime lane now has the following explicit maturity chain:
  - executable-adapter request
  - executable-adapter consumer
  - adapter-execution mediation
  - adapter realization
  - launch preparation / execution
  - harvested runtime outcomes

This matters because the branch can now distinguish:
- who is asking for execution
- who is currently responsible for it
- how execution is mediated
- and how that lane is concretely realized today

instead of flattening the last two steps into generic launch state.

## What Topological Surface Became More Real

- The local Isaac/Unitree route is no longer just “backend factory happens later.”
- The branch now says concretely whether the lane is realized as:
  - `local_backend_factory`
  - `external_launch_delegate`
  - or blocked realization

That is a real Phase 1 improvement because it turns the local-runtime seam from an implicit implementation detail into a typed, replayable, auditable subsystem surface.

## What Fake Readiness Was Removed

- Previously, the branch could know:
  - request
  - consumer
  - execution mediation
  - launch status

but still leave the actual realization method implicit.

- This tranche removes that implicitness.
- The branch no longer has to pretend that a ready adapter-execution path and a concretely realized local adapter are the same thing.

## What Is Still Only Contract-Shaped

- The final concrete Isaac Lab / Isaac Sim / Unitree adapter implementation is still missing.
- The new realization surface is not that final adapter; it is the explicit bridge between current runtime truth and that future adapter.
- The local bridge remains real-or-unavailable.

## Contracts and Artifacts Added or Altered

| Surface | File | Classification | Notes |
|--------|------|----------------|-------|
| `backend_executable_adapter_realization_v1` | `src/world_model/sim_synth_physics/adapters/isaac_unitree_adapter_realization.py` | WM-native realization contract | Names whether the adapter is realized through local backend-factory handoff or external delegate. |
| `backend_runtime_adapter_realization.json` | `src/world_model/sim_synth_physics/runtime.py` | Root-level loop artifact | Promotes realization out of nested metadata into a first-class artifact. |
| realization metadata inside adapter receipt | `src/world_model/sim_synth_physics/adapters/isaac_unitree_adapter_execution.py` | Receipt metadata extension | Keeps realization topology attached to adapter receipt truth. |
| realization fields in harvested corpora | `src/world_model/sim_synth_physics/training_corpus.py` | Training/export extension | Preserves realization path/status for backend-selector and branch-planner rows. |

## Tests and Verification

New focused test:
- `tests/test_isaac_unitree_adapter_realization.py`

Updated tests:
- `tests/test_sim_synth_runtime_launch.py`
- `tests/test_sim_synth_physics_world_model.py`
- `tests/test_sim_synth_training_corpus.py`

Targeted verification run:

```text
python3 -m compileall src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_adapter_realization.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q
python3 -m ruff check src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_adapter_realization.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py
python3 -m pytest -q tests/test_isaac_unitree_adapter_execution.py tests/test_isaac_unitree_adapter_realization.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py
git diff --check
```

Result:
- `41 passed`

## What Remains Missing

- The active Phase 1 tranche is still not done.
- The honest remaining blockers are:
  - the final concrete Isaac Lab / Isaac Sim / Unitree adapter implementation over the request / consumer / execution / realization chain
  - equivalent Holosoma realization depth
  - GPU-backed GGDS / video materialization

## Open Doctrinal Questions

- Once the final concrete Isaac/Unitree adapter lands, should the realization surface remain inside Sim / Synth / Physics only, or should part of it become shared with the later Embodiment / Actuation WM?
- Should there be a later normalized robot-operation receipt above profile-specific adapter realizations, or should that wait until real low-level control and Phase 4 timing/safety seams exist?

## Recommendation to Claude

- Keep Sim / Synth / Physics as the implementation center of gravity.
- Do not move upward in the roadmap yet.
- The next highest-leverage code target is:
  - the final concrete Isaac/Unitree adapter implementation over this now-explicit request / consumer / execution / realization chain
  - then equivalent Holosoma realization / runtime-execution deepening

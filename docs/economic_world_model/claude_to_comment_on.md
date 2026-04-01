# Claude Commentary Artifact

## Current Status

- **Tranche**: Sim / Synth / Physics WM closure, Phase 1 Isaac/Unitree adapter-execution mediation tranche
- **Date**: 2026-04-01
- **Branch**: `codex/multi-wm-architecture-plan`
- **Active tranche spec**: `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
- **Held tranche spec**: `docs/economic_world_model/codex_tranche_perception_wm_schema.md`

## What Was Implemented

- Added `src/world_model/sim_synth_physics/adapters/isaac_unitree_adapter_execution.py`.
- The Isaac/Unitree runtime lane now has a typed adapter-execution mediation layer over the already-landed:
  - executable-adapter request
  - executable-adapter consumer
- `src/world_model/sim_synth_physics/backend_runtime_execution.py` now:
  - prepares adapter execution from request + consumer
  - finalizes that mediation against either:
    - external launch results
    - local bridge handoff
  - emits both:
    - `backend_executable_adapter_execution_v1`
    - `backend_runtime_adapter_receipt_v1`
- `src/world_model/sim_synth_physics/runtime.py` now surfaces the adapter receipt as a first-class loop artifact and carries it into:
  - loop result serialization
  - loop summary output
  - training-feedback manifests
  - outcome metadata
- `src/world_model/sim_synth_physics/training_corpus.py` now harvests the adapter receipt and preserves it in backend-selector and branch-planner row metadata.
- `scripts/run_isaac_unitree_executable_adapter.py` now emits:
  - `adapter_execution`
  - `adapter_receipt`
  in addition to the existing request / consumer / launch report surfaces.

## What Changed Topologically

- No new WM boundary was introduced.
- The Sim / Synth / Physics WM gained a new explicit maturity rung inside the Isaac/Unitree runtime lane.
- The lane is now:
  - canonical world-state compilation
  - backend binding
  - deployment contract
  - runtime bridge
  - runtime bundle / launch spec
  - executable-adapter request
  - executable-adapter consumer
  - executable-adapter execution mediation
  - launch preparation / execution
  - harvested runtime outcomes

This matters because the lane no longer collapses “who is responsible for execution” and “what actually happened at execution time” into the same launch-shaped artifact.

## What Fake Readiness Was Removed

- Previously, once the request and consumer existed, the next visible truth in many places was just launch status.
- That could make the lane look flatter and more executable than it really was.
- This tranche removed that compression by introducing explicit adapter-execution statuses such as:
  - `local_bridge_ready`
  - `local_bridge_missing`
  - `local_bridge_handed_off`
  - `external_launch_ready`
  - `external_launch_completed`
  - `external_launch_failed`

So the branch can now say:
- the request exists
- a consumer exists
- the execution mediation exists
- but the final real adapter is still missing

without pretending those are the same thing.

## What Is Still Only Contract-Shaped

- The final concrete Isaac Lab / Isaac Sim / Unitree adapter realization is still missing.
- The new adapter-execution layer is honest mediation, not the final low-latency runtime bridge.
- The local bridge path is still real-or-unavailable:
  - if `src.motor_backend.workcell_isaaclab_backend` is absent, the lane says so explicitly
  - it does not pretend local execution exists just because the request/consumer chain is typed

## Contracts and Receipts Added or Altered

| Contract / Receipt | File | Classification | Notes |
|-------------------|------|----------------|-------|
| `backend_executable_adapter_execution_v1` | `src/world_model/sim_synth_physics/adapters/isaac_unitree_adapter_execution.py` | WM-native execution-mediation contract | Sits between executable request/consumer and launch/outcome. |
| `backend_runtime_adapter_receipt_v1` | `src/world_model/sim_synth_physics/receipts.py` | Canonical loop receipt | Preserves adapter status, execution path, and whether execution was actually attempted. |
| adapter receipt in loop result | `src/world_model/sim_synth_physics/runtime.py` | Loop-result extension | Makes adapter mediation a first-class runtime artifact instead of metadata-only. |
| adapter receipt in harvested corpora | `src/world_model/sim_synth_physics/training_corpus.py` | Downstream training/export extension | Keeps adapter truth visible to backend-selector / branch-planner rows. |

## Tests and Verification

New focused test:
- `tests/test_isaac_unitree_adapter_execution.py`

Updated tests:
- `tests/test_sim_synth_runtime_launch.py`
- `tests/test_sim_synth_physics_world_model.py`
- `tests/test_sim_synth_training_corpus.py`

Targeted verification run:

```text
python3 -m compileall src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_adapter_execution.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q
python3 -m ruff check src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_adapter_execution.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py
python3 -m pytest -q tests/test_isaac_unitree_adapter_execution.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_physics_world_model.py
git diff --check
```

Result:
- `39 passed`

## What Remains Missing

- The active tranche is still not done.
- The honest remaining Phase 1 blockers are still:
  - a concrete Isaac Lab / Isaac Sim / Unitree adapter that consumes the request / consumer / adapter-execution chain against real upstream runtime/assets
  - equivalent Holosoma runtime-execution deepening to the same standard
  - GPU-backed GGDS / video materialization

## Open Doctrinal Questions

- Should the final concrete Isaac/Unitree adapter remain fully owned by the Sim / Synth / Physics WM, or should part of that executable-adapter realization later be shared with the Embodiment / Actuation WM once low-level control becomes real?
- When the real adapter lands, should the adapter receipt stay profile-shaped, or should there also be a more normalized robot-operation receipt above the profile-specific adapter surfaces?

## Recommendation to Claude

- Keep Sim / Synth / Physics as the implementation center of gravity.
- Do not promote Perception implementation yet.
- Treat the next highest-leverage code target as:
  - the final concrete Isaac/Unitree adapter realization over this new request / consumer / adapter-execution chain
  - then equivalent Holosoma runtime-execution deepening

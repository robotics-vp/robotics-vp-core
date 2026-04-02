# Claude Commentary Artifact

## Current Status

- **Tranche**: Sim / Synth / Physics WM closure, Phase 1 local-materialization + Holosoma parity tranche
- **Date**: 2026-04-01
- **Branch**: `codex/multi-wm-architecture-plan`
- **Active tranche spec**: `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
- **Held tranche spec**: `docs/economic_world_model/codex_tranche_perception_wm_schema.md`

## What Was Implemented

- Added `src/world_model/sim_synth_physics/adapters/local_backend_factory_adapter.py`.
- Added:
  - `src/world_model/sim_synth_physics/adapters/holosoma_executable_adapter.py`
  - `src/world_model/sim_synth_physics/adapters/holosoma_executable_consumer.py`
  - `src/world_model/sim_synth_physics/adapters/holosoma_adapter_execution.py`
  - `src/world_model/sim_synth_physics/adapters/holosoma_adapter_realization.py`
- `runtime_bundles.py` now emits executable-adapter request/consumer surfaces for Holosoma as well as Isaac/Unitree.
- `backend_runtime_execution.py` now:
  - uses a typed local backend-factory invocation/result surface before concrete runtime execution
  - preserves that local materialization truth into runtime metadata
  - emits Holosoma adapter execution / realization / receipt metadata instead of leaving Holosoma as a concrete-runtime special case
- `training_corpus.py` now preserves local adapter invocation/result statuses for downstream rows.

## What Changed Topologically

- No new WM boundary was introduced.
- The Isaac/Unitree lane now has:
  - request
  - consumer
  - adapter execution
  - adapter realization
  - local backend-factory invocation/result
  - launch
  - harvested outcome
- Holosoma now has:
  - request
  - consumer
  - adapter execution
  - adapter realization
  - local backend-factory invocation/result
  - concrete runtime execution or external launch

This matters because the branch can now distinguish:
- contract-shaped readiness
- explicit local materialization attempts
- explicit external-launch delegation
- and real concrete runtime execution

instead of letting those collapse into generic backend status.

## What Topological Surface Became More Real

- The local runtime seam is no longer hidden inside direct `make_motor_backend(...)` calls.
- Holosoma is no longer allowed to stay a structurally looser special case while Isaac/Unitree gets typed adapter truth.

## What Fake Readiness Was Removed

- “Local backend exists because the code eventually calls the factory” is no longer treated as implicit readiness.
- “Holosoma concrete runtime exists” is no longer allowed to bypass the typed request/consumer/execution/realization ladder.
- The train-from-motion Holosoma lane no longer falsely blocks on `policy_checkpoint` when no policy is the honest bounded mode.

## What Is Still Only Contract-Shaped

- The final concrete Isaac Lab / Isaac Sim / Unitree upstream runtime/assets/policies are still missing.
- The final concrete Holosoma host/runtime/motion/policy/retargeting assets are still missing.
- The new local backend-factory invocation/result surface is not the final hardware adapter; it is the explicit bridge to that future adapter/runtime.

## Contracts and Artifacts Added or Altered

| Surface | File | Classification | Notes |
|--------|------|----------------|-------|
| `backend_local_factory_invocation_v1` / `backend_local_factory_result_v1` | `src/world_model/sim_synth_physics/adapters/local_backend_factory_adapter.py` | WM-native local materialization contract | Makes local backend-factory realization explicit instead of implicit. |
| Holosoma request / consumer / execution / realization | `src/world_model/sim_synth_physics/adapters/holosoma_*` | WM-native runtime ladder | Brings Holosoma up to the same structural standard as Isaac/Unitree. |
| local adapter statuses in corpus rows | `src/world_model/sim_synth_physics/training_corpus.py` | Training/export extension | Preserves whether local materialization was only invoked, blocked, or actually materialized. |

## Tests and Verification

New focused tests:
- `tests/test_local_backend_factory_adapter.py`
- `tests/test_holosoma_executable_adapter.py`
- `tests/test_holosoma_adapter_execution.py`
- `tests/test_holosoma_adapter_realization.py`

Updated tests:
- `tests/test_sim_synth_runtime_bundles.py`
- `tests/test_sim_synth_physics_world_model.py`
- `tests/test_sim_synth_training_corpus.py`

Targeted verification run:

```text
python3 -m compileall src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_local_backend_factory_adapter.py tests/test_holosoma_executable_adapter.py tests/test_holosoma_adapter_execution.py tests/test_holosoma_adapter_realization.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py -q
python3 -m ruff check src/world_model/sim_synth_physics scripts/run_isaac_unitree_executable_adapter.py tests/test_local_backend_factory_adapter.py tests/test_holosoma_executable_adapter.py tests/test_holosoma_adapter_execution.py tests/test_holosoma_adapter_realization.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py
python3 -m pytest -q tests/test_local_backend_factory_adapter.py tests/test_holosoma_executable_adapter.py tests/test_holosoma_adapter_execution.py tests/test_holosoma_adapter_realization.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py
git diff --check
```

Result:
- `35 passed`

## What Remains Missing

- The active Phase 1 tranche is still not done.
- The honest remaining blockers are now narrower:
  - final concrete Isaac Lab / Isaac Sim / Unitree runtime/assets/policies
  - final concrete Holosoma host/runtime/motion/policy/retargeting assets
  - GPU-backed GGDS / video materialization

## Open Doctrinal Questions

- Once both backend lanes have real upstream assets, should the local backend-factory invocation/result surface remain Phase-1-local, or become part of a later shared embodiment/runtime deployment substrate?
- When the real robot path exists, should the later Embodiment / Actuation WM own a normalized adapter-operation receipt above these backend-specific ladders, or should that wait until Phase 4 timing/safety/control separation is real?

## Recommendation to Claude

- Keep Sim / Synth / Physics as the implementation center of gravity.
- Do not move upward in the roadmap yet.
- The next highest-leverage code target is:
  - final concrete Isaac/Unitree upstream runtime/assets/policies behind the new materialization ladder
  - then equivalent Holosoma host/runtime asset realization under the same standard

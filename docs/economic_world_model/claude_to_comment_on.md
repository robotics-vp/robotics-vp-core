# Claude Commentary Artifact

## Current Status

- **Tranche**: Sim / Synth / Physics WM closure, Phase 1 Isaac/Unitree executable-adapter tranche
- **Date**: 2026-04-01
- **Branch**: `codex/multi-wm-architecture-plan`
- **Active tranche spec**: `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
- **Held tranche spec**: `docs/economic_world_model/codex_tranche_perception_wm_schema.md`

## What Was Implemented

- Added `src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_adapter.py`.
- Extended the Isaac/Unitree runtime artifact path so `runtime_bundles.py` emits a typed `backend_executable_adapter_request_v1` for the preferred runtime profile.
- The executable-adapter request now survives into:
  - `backend_runtime_bundle_v1`
  - `backend_launch_spec_v1`
  - launch preparation in `runtime_launch.py`
  - runtime execution metadata via `backend_runtime_execution.py`
- Added `scripts/run_isaac_unitree_executable_adapter.py` as a dedicated WM-facing runner over those artifacts.
- Added focused tests:
  - `tests/test_isaac_unitree_executable_adapter.py`
  - updates in `tests/test_sim_synth_runtime_bundles.py`
  - updates in `tests/test_sim_synth_runtime_launch.py`
  - existing world-model tests continue to cover the runtime path

## What Changed Topologically

- No new WM boundary was introduced.
- The Sim / Synth / Physics WM gained a more concrete executable-adapter layer for the Isaac/Unitree lane.
- The effective topology is now:
  - canonical world-state compilation
  - backend binding
  - deployment contract
  - runtime bridge
  - runtime bundle / launch spec
  - executable-adapter request
  - launch preparation / execution
  - harvested runtime outcomes
- This matters because the Unitree/Isaac lane is now less dependent on implicit shell-command semantics and more explicitly typed before a full in-process adapter exists.

## Contracts and Schemas Added or Altered

| Contract/Schema | File | Classification | Notes |
|----------------|------|----------------|-------|
| `backend_executable_adapter_request_v1` | `src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_adapter.py` | WM-native executable-adapter contract | Carries deployment mode, adapter entrypoint, robot variant, asset/calibration posture, env overrides, and output expectations. |
| `executable_adapter_request` in runtime bundle / launch spec | `src/world_model/sim_synth_physics/runtime_bundles.py` | Runtime bundle extension | Makes the executable lane explicit instead of leaving it in command strings only. |
| Launch-plan consumption of adapter request | `src/world_model/sim_synth_physics/runtime_launch.py` | Launch preparation extension | Merges adapter-specific env overrides, notes, and missing preconditions into launch preparation. |
| Dedicated Isaac/Unitree adapter runner | `scripts/run_isaac_unitree_executable_adapter.py` | Runner surface | Gives the lane a concrete WM-facing runner over the typed artifacts. |

## Tests and Receipts Added

| Test / Receipt | File | What It Verifies |
|----------------|------|------------------|
| `test_isaac_unitree_executable_adapter_*` | `tests/test_isaac_unitree_executable_adapter.py` | Adapter request mode, env overrides, asset refs, and output expectations. |
| `test_build_isaac_runtime_bundle_*` additions | `tests/test_sim_synth_runtime_bundles.py` | Runtime bundle/launch spec now carry the executable-adapter request. |
| `test_prepare_backend_runtime_launch_ready` additions | `tests/test_sim_synth_runtime_launch.py` | Launch preparation now respects adapter env overrides and request metadata. |
| `test_run_isaac_unitree_executable_adapter_script_writes_adapter_request` | `tests/test_sim_synth_runtime_launch.py` | Dedicated runner preserves the request and launch receipt end to end. |

## What Remains Missing

- Phase 1 is still not structurally closed.
- The honest remainder is now:
  - a concrete Isaac Lab / Isaac Sim / Unitree executable adapter that consumes the typed executable-adapter request against real upstream runtime/assets
  - the equivalent Holosoma runtime-execution deepening
  - GPU-backed GGDS / video materialization
  - broader Tier 1 audit items in `codex_tranche_sim_synth_closure.md`, especially full compiler/receipt-chain completeness and cross-backend outcome parser parity

## Open Doctrinal Questions

- When the concrete Unitree runtime adapters land, should the executable-adapter request stay Phase-1-owned, or should it later become a shared contract between Sim / Synth / Physics and Embodiment / Actuation?
- How aggressively should the executable-adapter request normalize profile-specific launch semantics now, versus leaving more shape in upstream-profile-specific commands until the real adapters are present?
- When the concrete Unitree runtime adapters land, should harvested runtime outputs remain profile-shaped, or should an additional normalized robot-operation receipt sit above profile-specific outputs immediately?

## Docs / Roadmap / README Changes Needed

- The active roadmap docs that needed updating have been updated:
  - `docs/economic_world_model/multi_wm_architecture_plan.md`
  - `docs/economic_world_model/roadmap.md`
  - `docs/economic_world_model/progress_log.md`
  - `docs/economic_world_model/implementation_notes.md`
- No broader doctrine rewrite appears necessary after this sub-tranche.

## Verification Results

Targeted verification passed for this tranche:

```text
python3 -m compileall src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_adapter.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/backend_runtime_execution.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_executable_adapter.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py -q
python3 -m ruff check src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_adapter.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/backend_runtime_execution.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_executable_adapter.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py
python3 -m pytest -q tests/test_isaac_unitree_executable_adapter.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py
```

Result:
- `35 passed`

Not run for this sub-tranche:
- full repo `pytest`
- full `python3 -m compileall src/`
- full `ruff check .`

## Current Assessment

- Active tranche remains: `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
- Held tranche remains: `docs/economic_world_model/codex_tranche_perception_wm_schema.md`
- Recommendation to Claude:
  - keep Sim / Synth / Physics as the implementation center of gravity
  - do not promote Phase 2 implementation yet
  - next highest-leverage code target is the concrete Isaac/Unitree adapter consumer over the new executable-adapter request, followed by the equivalent Holosoma runtime-execution deepening

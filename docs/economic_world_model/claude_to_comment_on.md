# Claude Commentary Artifact

## Current Status

- **Tranche**: Sim / Synth / Physics WM closure, Phase 1 Isaac/Unitree executable-adapter consumer tranche
- **Date**: 2026-04-01
- **Branch**: `codex/multi-wm-architecture-plan`
- **Active tranche spec**: `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
- **Held tranche spec**: `docs/economic_world_model/codex_tranche_perception_wm_schema.md`

## What Was Implemented

- Added `src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_consumer.py`.
- Extended the Isaac/Unitree runtime artifact path so `runtime_bundles.py` emits a typed `backend_executable_adapter_consumer_v1` beside the executable-adapter request.
- The consumer now survives into:
  - `backend_runtime_bundle_v1`
  - `backend_launch_spec_v1`
  - launch preparation in `runtime_launch.py`
  - runtime execution metadata and artifacts via `backend_runtime_execution.py`
  - the standalone runner in `scripts/run_isaac_unitree_executable_adapter.py`
- Added focused tests:
  - `tests/test_isaac_unitree_executable_consumer.py`
  - updates in `tests/test_sim_synth_runtime_bundles.py`
  - updates in `tests/test_sim_synth_runtime_launch.py`
  - updates in `tests/test_sim_synth_physics_world_model.py`

## What Changed Topologically

- No new WM boundary was introduced.
- The Sim / Synth / Physics WM gained a more concrete executable-adapter consumer layer for the Isaac/Unitree lane.
- The effective topology is now:
  - canonical world-state compilation
  - backend binding
  - deployment contract
  - runtime bridge
  - runtime bundle / launch spec
  - executable-adapter request
  - executable-adapter consumer
  - launch preparation / execution
  - harvested runtime outcomes
- This matters because the Unitree/Isaac lane can now distinguish:
  - the request being made
  - the consumer path currently responsible for that request
  - the still-missing final real adapter realization

## Contracts and Schemas Added or Altered

| Contract/Schema | File | Classification | Notes |
|----------------|------|----------------|-------|
| `backend_executable_adapter_consumer_v1` | `src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_consumer.py` | WM-native executable consumer contract | Carries consumer mode, consumer status, local-vs-external responsibility, and remaining preconditions. |
| `executable_adapter_consumer` in runtime bundle / launch spec | `src/world_model/sim_synth_physics/runtime_bundles.py` | Runtime bundle extension | Makes the consumer over the executable request explicit. |
| Launch-plan consumption of adapter consumer | `src/world_model/sim_synth_physics/runtime_launch.py` | Launch preparation extension | Uses consumer metadata to drive launch mediation instead of only the request. |
| Dedicated Isaac/Unitree adapter runner | `scripts/run_isaac_unitree_executable_adapter.py` | Runner surface | Now exposes both request and consumer over the typed artifacts. |

## Tests and Receipts Added

| Test / Receipt | File | What It Verifies |
|----------------|------|------------------|
| `test_isaac_unitree_executable_consumer_*` | `tests/test_isaac_unitree_executable_consumer.py` | Consumer mode, consumer status, and blocked-vs-ready mediation. |
| `test_build_isaac_runtime_bundle_*` additions | `tests/test_sim_synth_runtime_bundles.py` | Runtime bundle/launch spec now carry the executable-adapter consumer. |
| `test_prepare_backend_runtime_launch_ready` additions | `tests/test_sim_synth_runtime_launch.py` | Launch preparation now respects adapter consumer env overrides and metadata. |
| `test_run_isaac_unitree_executable_adapter_script_writes_adapter_request` | `tests/test_sim_synth_runtime_launch.py` | Dedicated runner now preserves both request and consumer. |
| runtime execution metadata addition | `tests/test_sim_synth_physics_world_model.py` | Live runtime result preserves executable-adapter consumer metadata. |

## What Remains Missing

- Phase 1 is still not structurally closed.
- The honest remainder is now:
  - a concrete Isaac Lab / Isaac Sim / Unitree adapter that consumes the typed executable-adapter consumer against real upstream runtime/assets
  - the equivalent Holosoma runtime-execution deepening
  - GPU-backed GGDS / video materialization
  - broader Tier 1 audit items in `codex_tranche_sim_synth_closure.md`, especially full compiler/receipt-chain completeness and cross-backend outcome parser parity

## Open Doctrinal Questions

- When the concrete Unitree runtime adapters land, should the executable-adapter consumer stay Phase-1-owned, or should it later become a shared contract between Sim / Synth / Physics and Embodiment / Actuation?
- How aggressively should the executable-adapter consumer normalize profile-specific launch semantics now, versus leaving more shape in upstream-profile-specific consumers until the real adapters are present?
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
python3 -m compileall src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_consumer.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/backend_runtime_execution.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_executable_consumer.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py -q
python3 -m ruff check src/world_model/sim_synth_physics/adapters/isaac_unitree_executable_consumer.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/backend_runtime_execution.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_executable_consumer.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py
python3 -m pytest -q tests/test_isaac_unitree_executable_consumer.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py
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
  - next highest-leverage code target is the concrete Isaac/Unitree adapter implementation that consumes this new executable-adapter consumer against real upstream runtime/assets, followed by the equivalent Holosoma runtime-execution deepening

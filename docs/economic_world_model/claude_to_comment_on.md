# Claude Commentary Artifact

## Current Status

- **Tranche**: Sim / Synth / Physics WM closure, Phase 1 upstream-runtime-pack + Holosoma deployment tranche
- **Date**: 2026-04-01
- **Branch**: `codex/multi-wm-architecture-plan`
- **Active tranche spec**: `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
- **Held tranche spec**: `docs/economic_world_model/codex_tranche_perception_wm_schema.md`

## What Was Implemented

- Added:
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_pack.py`
  - `src/world_model/sim_synth_physics/adapters/holosoma_deployment.py`
  - `src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py`
- `src/world_model/sim_synth_physics/adapters/backend_isaac.py` now emits:
  - `deployment_contract`
  - `upstream_runtime_pack`
- `src/world_model/sim_synth_physics/adapters/backend_holosoma.py` now emits:
  - `deployment_contract`
  - `upstream_runtime_pack`
  - a more honest base-asset posture where retargeting / reward-overlay / policy surfaces are not treated as universally required for every Holosoma mode
- `runtime_bundles.py` now carries `upstream_runtime_pack` through the backend runtime bundle and launch spec and writes `backend_upstream_runtime_pack.json`
- `runtime_bridge.py`, `runtime.py`, `runtime_work_orders.py`, and `training_corpus.py` now preserve upstream-runtime-pack truth into:
  - bridge receipts
  - loop summaries
  - training feedback
  - runtime work orders
  - backend-selector rows
  - branch-planner rows
- `scripts/scan_phase1_runtime_layouts.py` now emits:
  - Isaac deployment contract
  - Isaac upstream runtime pack
  - Holosoma deployment contract
  - Holosoma upstream runtime pack

## What Changed Topologically

- No new WM was introduced.
- Phase 1 backend closure now has another explicit rung between backend binding and concrete runtime realization:
  - backend binding
  - deployment contract
  - upstream runtime pack
  - executable-adapter request
  - executable-adapter consumer
  - adapter execution
  - adapter realization
  - local materialization / external launch
  - harvested runtime outcomes

This matters because the branch can now say:
- which backend mode is actually intended (`sim_eval`, `motion_train`, `teleop_bridge`, `lerobot_eval`, `physical_deploy`, `retarget_eval`)
- whether the relevant upstream runtime/profile/policy/asset surfaces are pack-ready, pack-partial, or pack-blocked
- and which missing components remain before a backend lane stops being “contract-shaped”

## What Topological Surface Became More Real

- The upstream runtime surface is no longer inferred only from roots, layouts, or generic launch readiness.
- Isaac/Unitree now has a canonical WM-owned runtime pack over runtime targets, runtime layouts, deploy surfaces, policy-bank surfaces, telemetry surfaces, and robot-asset refs.
- Holosoma now has a canonical deployment contract plus runtime pack over runtime roots, motion sources, retargeting posture, reward-overlay posture, policy surfaces, and telemetry surfaces.

## What Fake Readiness Was Removed

- “Backend root exists, so the runtime is basically ready” is no longer enough.
- “Holosoma always needs policy + retargeting + reward overlay” is no longer treated as universally true.
- “Deployment posture can be reconstructed later from runtime targets and launch strings” is no longer assumed.

## What Is Still Only Contract-Shaped

- The actual upstream Isaac Lab / Isaac Sim / Unitree repos, assets, checkpoints, and host bring-up are still external reality.
- The actual Holosoma host/runtime, motion corpora, retargeting assets, and policy assets are still external reality.
- The new runtime-pack contracts are explicit representations of those dependencies; they are not substitutes for them.

## Contracts and Artifacts Added or Altered

| Surface | File | Classification | Notes |
|--------|------|----------------|-------|
| `isaac_unitree_runtime_pack_v1` | `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_pack.py` | WM-native upstream-runtime contract | Makes runtime profile / deploy / policy / asset / telemetry surfaces explicit for Isaac/Unitree. |
| `holosoma_deployment_contract_v1` | `src/world_model/sim_synth_physics/adapters/holosoma_deployment.py` | WM-native deployment contract | Separates `sim_eval`, `motion_train`, and `retarget_eval` with explicit missing-precondition logic. |
| `holosoma_runtime_pack_v1` | `src/world_model/sim_synth_physics/adapters/holosoma_runtime_pack.py` | WM-native upstream-runtime contract | Makes runtime root / motion / retargeting / policy / telemetry surfaces explicit for Holosoma. |
| `backend_upstream_runtime_pack.json` | `src/world_model/sim_synth_physics/runtime_bundles.py` | Runtime artifact | Writes the canonical pack artifact beside runtime bundles and launch specs. |
| upstream-runtime-pack metadata in work orders and corpus rows | `src/world_model/sim_synth_physics/runtime_work_orders.py`, `src/world_model/sim_synth_physics/training_corpus.py` | Downstream consumption | Preserves pack status / ready surfaces / missing components so later training or GPU bring-up does not have to rediscover them. |

## Tests and Verification

New focused tests:
- `tests/test_holosoma_deployment.py`
- `tests/test_holosoma_runtime_pack.py`
- `tests/test_isaac_unitree_runtime_pack.py`
- `tests/test_scan_phase1_runtime_layouts.py`

Updated tests:
- `tests/test_sim_synth_runtime_bundles.py`
- `tests/test_sim_synth_physics_world_model.py`
- downstream verification:
  - `tests/test_sim_synth_training_corpus.py`
  - `tests/test_sim_synth_runtime_work_orders.py`
  - `tests/test_sim_synth_runtime_launch.py`

Targeted verification run:

```text
python3 -m compileall src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py tests/test_holosoma_deployment.py tests/test_holosoma_runtime_pack.py tests/test_isaac_unitree_runtime_pack.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py -q
python3 -m ruff check src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py tests/test_holosoma_deployment.py tests/test_holosoma_runtime_pack.py tests/test_isaac_unitree_runtime_pack.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py
python3 -m pytest -q tests/test_holosoma_deployment.py tests/test_holosoma_runtime_pack.py tests/test_isaac_unitree_runtime_pack.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py
python3 -m pytest -q tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_work_orders.py tests/test_sim_synth_runtime_launch.py
git diff --check
```

Result:
- `32 passed`
- `14 passed`

## What Remains Missing

- The active Phase 1 tranche is still not done.
- The honest remaining blockers are narrower and more external:
  - real Isaac Lab / Isaac Sim / Unitree upstream runtime/assets/policies behind the new runtime pack
  - real Holosoma host/runtime/motion/policy/retargeting assets behind the new runtime pack
  - GPU-backed GGDS / video materialization

## Open Doctrinal Questions

- Once both backend lanes have real upstream runtime packs, should Phase 1 keep pack-specific metadata backend-local, or should a later embodiment/deployment layer normalize them into a shared runtime-pack ontology?
- When the Embodiment / Actuation WM becomes active, should compute/battery placement state condition these runtime packs directly, or should that wait until Phase 4 control-rate / companion-compute separation is real?

## Recommendation to Claude

- Keep Sim / Synth / Physics as the implementation center of gravity.
- Do not move upward in the roadmap yet.
- The next highest-leverage code target is:
  - make the Isaac/Unitree runtime pack bind to a real upstream runtime / asset / policy host surface
  - then do the same for the Holosoma runtime pack and deployment modes

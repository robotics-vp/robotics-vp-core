# Claude Commentary Artifact

## Current Status

- **Tranche**: Sim / Synth / Physics WM closure, Phase 1 runtime-binding + concrete Holosoma local-runtime correction
- **Date**: 2026-04-01
- **Branch**: `codex/multi-wm-architecture-plan`
- **Active tranche spec**: `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
- **Held tranche spec**: `docs/economic_world_model/codex_tranche_perception_wm_schema.md`

## What Was Implemented

- Added:
  - `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py`
  - `src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py`
- Threaded runtime binding through:
  - `src/world_model/sim_synth_physics/runtime_bundles.py`
  - `src/world_model/sim_synth_physics/runtime_launch.py`
  - `src/world_model/sim_synth_physics/runtime_work_orders.py`
  - `src/world_model/sim_synth_physics/runtime.py`
  - `src/world_model/sim_synth_physics/training_corpus.py`
  - `scripts/scan_phase1_runtime_layouts.py`
  - `scripts/run_isaac_unitree_executable_adapter.py`
- Fixed the Holosoma train-from-motion patch path in:
  - `src/world_model/sim_synth_physics/backend_runtime_execution.py`
  - the branch now rebuilds the Holosoma executable-adapter request from the patched runtime binding instead of mutating a stale `sim_eval` request in place

## What Changed Topologically

- No new WM was introduced.
- Phase 1 backend closure now has an explicit runtime-binding rung between upstream runtime packs and executable-adapter requests:
  - backend binding
  - deployment contract
  - upstream runtime pack
  - runtime binding
  - executable-adapter request
  - executable-adapter consumer
  - adapter execution
  - adapter realization
  - local materialization / external launch
  - harvested runtime outcomes

## What Topological Surface Became More Real

- The backend lanes no longer jump straight from upstream runtime-pack posture to executable-adapter requests.
- The WM now binds concrete mode-relevant surfaces first:
  - selected policy surface
  - selected motion surface
  - selected retargeting surface
  - selected launch root / command
  - selected target refs
- Those selections and their missing components are now canonical loop artifacts rather than implicit recomputation from pack metadata or launch status.

## What Fake Readiness Was Removed

- Pack-level missing components are no longer treated as universally relevant to every execution mode.
- Local Holosoma eval is no longer falsely blocked by missing external repo/launch surfaces when the branch already has:
  - a local runtime bridge
  - an explicit policy ref
- Local Holosoma train-from-motion is no longer falsely blocked by stale `policy_surface` / `policy_checkpoint` gaps when the branch already has:
  - a local runtime bridge
  - motion datapacks and/or inline motion clips

## What Is Still Only Contract-Shaped

- The actual Isaac Lab / Isaac Sim / Unitree upstream repos, assets, checkpoints, and host bring-up remain external reality.
- The actual Holosoma host/runtime, motion corpora, retargeting assets, and policy assets remain external reality.
- The new runtime-binding layer is a real canonical WM surface, but it still binds against provider-owned runtime packs rather than replacing them.

## Contracts and Artifacts Added or Altered

| Surface | File | Classification | Notes |
|--------|------|----------------|-------|
| `backend_runtime_binding_v1` for Isaac/Unitree | `src/world_model/sim_synth_physics/adapters/isaac_unitree_runtime_binding.py` | WM-native binding contract | Selects deployment-mode-relevant policy / launch / target / asset surfaces from the upstream pack before adapter request build. |
| `backend_runtime_binding_v1` for Holosoma | `src/world_model/sim_synth_physics/adapters/holosoma_runtime_binding.py` | WM-native binding contract | Selects policy / motion / retargeting / launch surfaces per mode and stops irrelevant external-pack gaps from blocking valid local modes. |
| `backend_runtime_binding.json` | `src/world_model/sim_synth_physics/runtime_bundles.py` | Runtime artifact | Writes the canonical binding artifact beside runtime bundles and launch specs. |
| Runtime-binding metadata in launch/work-order/corpus paths | `runtime_launch.py`, `runtime_work_orders.py`, `runtime.py`, `training_corpus.py` | Downstream consumption | Preserves binding status plus selected profile/policy/root/missing-component truth across the live loop. |
| Rebuilt Holosoma motion-train request | `backend_runtime_execution.py` | Behavioral correction | Rebuilds a motion-train request from the patched binding instead of mutating a stale `sim_eval` request. |

## Tests and Verification

New focused tests:
- `tests/test_isaac_unitree_runtime_binding.py`
- `tests/test_holosoma_runtime_binding.py`

Updated tests:
- `tests/test_scan_phase1_runtime_layouts.py`
- `tests/test_sim_synth_runtime_bundles.py`
- `tests/test_sim_synth_runtime_launch.py`
- `tests/test_sim_synth_physics_world_model.py`
- `tests/test_sim_synth_training_corpus.py`
- `tests/test_sim_synth_runtime_work_orders.py`

Targeted verification run:

```text
python3 -m compileall src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_work_orders.py -q
python3 -m ruff check src/world_model/sim_synth_physics scripts/scan_phase1_runtime_layouts.py scripts/run_isaac_unitree_executable_adapter.py tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_work_orders.py
python3 -m pytest -q tests/test_isaac_unitree_runtime_binding.py tests/test_holosoma_runtime_binding.py tests/test_scan_phase1_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_runtime_launch.py tests/test_sim_synth_physics_world_model.py tests/test_sim_synth_training_corpus.py tests/test_sim_synth_runtime_work_orders.py
git diff --check
```

Result:
- `44 passed`

## What Remains Missing

- The active Phase 1 tranche is still not done.
- The honest remaining blockers are now narrower and more external:
  - real Isaac/Unitree runtime, assets, checkpoints, and host bring-up behind the new runtime-pack -> runtime-binding -> adapter ladder
  - real Holosoma host/runtime, motion/retargeting assets, and policy assets behind the same ladder
  - GPU-backed GGDS / video materialization

## Open Doctrinal Questions

- Once both backend lanes have real upstream runtime packs and real runtime bindings, should later embodiment/deployment layers normalize them into a shared runtime-binding ontology, or keep them backend-local until the Embodiment / Actuation WM is active?
- When compute/battery canonical state becomes live in later lower WMs, should runtime-binding compilation consume those resource receipts directly, or should that wait until the Phase 4 real-time / companion-compute layers make placement and QoS consequences fully real?

## Recommendation to Claude

- Keep Sim / Synth / Physics as the implementation center of gravity.
- Do not move upward in the roadmap yet.
- The next highest-leverage code target is:
  - bind the Isaac/Unitree runtime-binding ladder to real upstream runtime / asset / checkpoint hosts
  - then do the same for Holosoma runtime binding and deployment modes

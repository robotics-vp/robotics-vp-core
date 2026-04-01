# Claude Commentary Artifact

## Current Status

- **Tranche**: Sim / Synth / Physics WM closure, Phase 1 Isaac/Unitree deployment-contract tranche
- **Date**: 2026-04-01
- **Branch**: `codex/multi-wm-architecture-plan`
- **Active tranche spec**: `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
- **Held tranche spec**: `docs/economic_world_model/codex_tranche_perception_wm_schema.md`

## What Was Implemented

- Landed the branch-truth reconciliation for the current multi-WM program:
  - `.agent/claude_copilot.md`
  - `docs/economic_world_model/neuralization_bridge_doctrine.md`
  - `docs/economic_world_model/codex_tranche_sim_synth_closure.md`
  - `docs/economic_world_model/codex_tranche_perception_wm_schema.md`
- Added `src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py`.
- Extended the Isaac backend binding and runtime path:
  - `src/world_model/sim_synth_physics/adapters/backend_isaac.py`
  - `src/world_model/sim_synth_physics/runtime_targets.py`
  - `src/world_model/sim_synth_physics/runtime_layouts.py`
  - `src/world_model/sim_synth_physics/runtime_bundles.py`
  - `src/world_model/sim_synth_physics/runtime_bridge.py`
  - `src/world_model/sim_synth_physics/runtime_launch.py`
  - `src/world_model/sim_synth_physics/runtime_outcomes.py`
  - `src/world_model/sim_synth_physics/backend_runtime_execution.py`
- Added focused tests:
  - `tests/test_isaac_unitree_deployment.py`
  - updates in `tests/test_sim_synth_runtime_targets.py`
  - updates in `tests/test_sim_synth_runtime_layouts.py`
  - updates in `tests/test_sim_synth_runtime_bundles.py`
  - updates in `tests/test_sim_synth_physics_world_model.py`

## What Changed Topologically

- No new WM boundary was introduced.
- The Sim / Synth / Physics WM gained a more explicit external-runtime deployment layer for the Isaac/Unitree lane.
- The Isaac backend no longer collapses all non-local-runtime posture into a generic shadow/unavailable state.
- External launch readiness, teleop readiness, LeRobot eval readiness, and physical-deploy preconditions are now represented as typed deployment-contract state that survives into:
  - backend binding metadata
  - runtime bundles
  - runtime bridge receipts
  - launch specs
  - runtime outcome harvesting
- The effective topology is now:
  - canonical world-state compilation
  - backend binding
  - deployment contract
  - runtime bridge
  - launch spec / output contract
  - harvested runtime outcomes
  rather than binding directly into a flatter launch/output path.

## Contracts and Schemas Added or Altered

| Contract/Schema | File | Classification | Notes |
|----------------|------|----------------|-------|
| `isaac_unitree_deployment_contract_v1` | `src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py` | WM-native deployment contract | Adds `sim_eval`, `teleop_bridge`, `lerobot_eval`, and `physical_deploy` readiness plus missing-precondition logic. |
| `deployment_contract` in Isaac binding metadata | `src/world_model/sim_synth_physics/adapters/backend_isaac.py` | Binding metadata extension | Makes `external_launch_ready` and `external_launch_assets_missing` first-class binding states. |
| `unitree_lerobot` runtime profile | `src/world_model/sim_synth_physics/runtime_layouts.py` | Runtime layout contract extension | Adds LeRobot deploy/policy/data candidate discovery. |
| Extended Isaac runtime target family | `src/world_model/sim_synth_physics/runtime_targets.py` | Runtime target contract extension | Adds `unitree_lerobot_root` alias plus `xr_teleoperate_root` / `unitree_il_lerobot_root` as valid external runtime roots. |
| Deployment-aware runtime bundle / launch spec | `src/world_model/sim_synth_physics/runtime_bundles.py` | Runtime bundle extension | Preferred profile selection now respects deployment posture instead of generic layout ordering only. |
| Extended bridge transport profiles | `src/world_model/sim_synth_physics/runtime_bridge.py` | Runtime bridge extension | Adds `unitree_xr_teleop_bridge` and `unitree_lerobot_eval_bridge`. |
| `unitree_lerobot` output source family | `src/world_model/sim_synth_physics/runtime_outcomes.py` | Runtime outcome contract extension | Lets harvested outputs preserve LeRobot-specific runtime artifact expectations. |

## Tests and Receipts Added

| Test / Receipt | File | What It Verifies |
|----------------|------|------------------|
| `test_build_isaac_unitree_deployment_contract_*` | `tests/test_isaac_unitree_deployment.py` | Deployment-mode readiness, missing-precondition logic, and profile preference. |
| `test_isaac_runtime_targets_accept_lerobot_alias` | `tests/test_sim_synth_runtime_targets.py` | `unitree_lerobot_root` alias is treated as canonical Isaac runtime-target input. |
| `test_isaac_runtime_layouts_detect_lerobot_profile` | `tests/test_sim_synth_runtime_layouts.py` | LeRobot repo shape is recognized as a real runtime profile. |
| `test_build_isaac_runtime_bundle_can_prefer_lerobot_profile` | `tests/test_sim_synth_runtime_bundles.py` | Deployment contract can steer bundle/launch selection toward LeRobot eval. |
| `test_world_state_marks_isaac_external_launch_ready_for_lerobot_and_teleop` | `tests/test_sim_synth_physics_world_model.py` | World-state compilation preserves teleop + sdk2_python + teleimager posture into bridge readiness. |
| `backend_runtime_bridge_receipt_v1` metadata extension | `src/world_model/sim_synth_physics/runtime_bridge.py` | Receipt change | Bridge receipts now preserve `deployment_contract`. |
| `backend_runtime_bundle_v1` / `backend_launch_spec_v1` metadata extension | `src/world_model/sim_synth_physics/runtime_bundles.py` | Receipt artifact change | Runtime bundles and launch specs now carry deployment posture. |

## What Remains Missing

- Phase 1 is still not structurally closed.
- The honest remainder is now increasingly external/runtime/provider constrained, but some implementation remains:
  - a concrete Isaac Lab / Isaac Sim / Unitree executable adapter path, not only deployment-aware launch preparation
  - a concrete Holosoma host/runtime/policy/motion/retargeting execution path under the same contract quality
  - GPU-backed GGDS / video materialization
  - broader Tier 1 audit items in `codex_tranche_sim_synth_closure.md`, especially full compiler/receipt-chain completeness and cross-backend outcome parser parity
- `claude_to_comment_on.md` was previously a dormant template; this file is now the first real tranche handoff artifact, but the habit still needs to be maintained after future sub-tranches.

## Open Doctrinal Questions

- Should `physical_deploy` stay inside the Phase 1 sim/synth deployment contract as a readiness surface, or eventually move partly into the Embodiment / Actuation WM once that WM owns more of the robot-facing control envelope?
- How aggressively should the Isaac/Unitree runtime bundle prefer teleop versus LeRobot eval when both are ready? The current ordering is deployment-contract driven but still heuristic.
- When the concrete Unitree runtime adapters land, should harvested runtime outputs remain profile-shaped, or should an additional normalized robot-operation receipt sit above profile-specific outputs immediately?

## Docs / Roadmap / README Changes Needed

- No new master-plan rewrite appears necessary after this tranche.
- The active roadmap docs that needed updating have been updated:
  - `docs/economic_world_model/progress_log.md`
  - `docs/economic_world_model/implementation_notes.md`
- The next documentation need is likely a narrower follow-up note once the concrete Isaac/Unitree executable adapter lands, because that is the next place where Phase 1 may cross from structural closure work into honest external-runtime constraints.

## Verification Results

Targeted verification passed for this tranche:

```text
python3 -m compileall src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py src/world_model/sim_synth_physics/adapters/backend_isaac.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_targets.py src/world_model/sim_synth_physics/runtime_layouts.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_bridge.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/runtime_outcomes.py src/world_model/sim_synth_physics/backend_runtime_execution.py tests/test_isaac_unitree_deployment.py tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py -q
python3 -m ruff check src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py src/world_model/sim_synth_physics/adapters/backend_isaac.py src/world_model/sim_synth_physics/adapters/__init__.py src/world_model/sim_synth_physics/runtime_targets.py src/world_model/sim_synth_physics/runtime_layouts.py src/world_model/sim_synth_physics/runtime_bundles.py src/world_model/sim_synth_physics/runtime_bridge.py src/world_model/sim_synth_physics/runtime_launch.py src/world_model/sim_synth_physics/runtime_outcomes.py src/world_model/sim_synth_physics/backend_runtime_execution.py tests/test_isaac_unitree_deployment.py tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py
python3 -m pytest -q tests/test_isaac_unitree_deployment.py tests/test_sim_synth_runtime_targets.py tests/test_sim_synth_runtime_layouts.py tests/test_sim_synth_runtime_bundles.py tests/test_sim_synth_physics_world_model.py
```

Result:
- `34 passed`

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
  - next highest-leverage code target is the concrete Isaac/Unitree executable-adapter lane, followed by the equivalent Holosoma runtime-execution deepening

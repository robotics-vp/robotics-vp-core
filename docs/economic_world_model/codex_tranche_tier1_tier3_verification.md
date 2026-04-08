# Codex Verification Tranche: Tier 1 + Tier 3 Checks

## Classification

- **Type**: verification / completeness audit (not new architecture)
- **Branch**: `codex/multi-wm-architecture-plan`
- **Priority**: required after real-runtime bring-up tranche completes
- **Prerequisite**: Isaac/Unitree and Holosoma runtime-binding ladder bound to real upstream hosts
- **Sequencing**: this tranche does NOT add new rungs or surfaces. It verifies that existing surfaces are correctly connected, emit the right receipts, and react to the now-deeper Tier 2 runtime truth.

## Motivation

The Tier 2 backend adapter/runtime ladder has been substantially deepened through 8+ tranches (executable adapter → consumer → execution → realization → runtime pack → runtime binding). Tiers 1 and 3 were not addressed during that work. Before claiming Phase 1 structural closure, the compiler pipeline, receipt chains, and inferential/promotion surfaces must be verified against the deeper Tier 2 runtime truth.

This is verification-first work: read what exists, check completeness, fix gaps. Not architecture.

## Tier 1 Verification Checks

### 1.1 Compiler full-artifact assembly audit

**File**: `src/world_model/sim_synth_physics/compiler.py`

**Check**: Does `compile_sim_synth_physics_world_state()` now assemble all canonical state objects in documented order?

Verify:
- [ ] Physics context compiled via `_compile_physics_context()`
- [ ] Adaptation policy compiled via `compile_physics_adaptation_policy()`
- [ ] Backend execution binding compiled via `compile_backend_execution_binding()`
- [ ] Robot asset contract compiled via `compile_robot_asset_contract()`
- [ ] Runtime bridge compiled via `compile_backend_runtime_bridge()`
- [ ] Synthetic branch plans compiled via `compile_synthetic_branch_plans()`
- [ ] Gen2Sim admission compiled via `compile_gen2sim_admission_state()`
- [ ] Diffusion conditioning compiled (if applicable to branch plans)
- [ ] Inferential learnability contracts summarized via `summarize_inferential_learnability_contracts()`
- [ ] All expected receipts listed in `EXPECTED_RECEIPTS` are actually producible from the compiled state

**Check**: Does the compiler output carry or reference the Tier 2 runtime-binding truth?

Verify:
- [ ] The backend execution binding now carries or can access runtime-binding status from the deeper ladder (pack → binding → request → consumer → execution → realization)
- [ ] If the compiler currently stops at `BackendExecutionBindingState` without referencing the deeper adapter ladder, note this as a gap — the binding state should eventually reflect whether the backend is merely contract-shaped or concretely realizable
- [ ] The compiler does NOT silently skip steps when a backend is unavailable — it emits honest unavailable receipts

### 1.2 Physics contracts execution binding completeness

**File**: `src/world_model/sim_synth_physics/physics_contracts.py`

Verify:
- [ ] `PhysicsExecutionContract` is built during compilation and included in `SimSynthPhysicsWorldState`
- [ ] Contains honest backend/fidelity/timestep contracts
- [ ] `route_status` field reflects actual backend availability, not optimistic default
- [ ] Emitted as a canonical state artifact accessible to downstream consumers (shadow execution, calibration, training corpus)
- [ ] `target_hardware_class` field carries honest Unitree/tabletop/etc. classification

### 1.3 Gen2Sim admission explicit receipt emission

**File**: `src/world_model/sim_synth_physics/gen2sim_admission.py`

Verify:
- [ ] `compile_gen2sim_admission_state()` returns a `Gen2SimAdmissionState` that is included in the compiled world state
- [ ] Admission decisions reference canonical asset contracts (not just free-floating metadata)
- [ ] Real vs synthetic evidence provenance is tracked in admission fields
- [ ] `assess_local_branch_corpus_gen2sim()` emits per-stage tallies as canonical receipt metadata (promotion_stage counts, evidence counts)
- [ ] If gen2sim admission currently does NOT emit a separate typed receipt (only returns state), note whether a `Gen2SimAdmissionReceipt` should be added for receipt-chain completeness, or whether the state object is sufficient

### 1.4 Training corpus receipt consumption

**File**: `src/world_model/sim_synth_physics/training_corpus.py`

Verify:
- [ ] Training corpus reads from `SimSynthPhysicsWorldState` canonical receipts — not from bespoke backend-specific JSON parsing
- [ ] Runtime-binding metadata is now threaded into training corpus rows (this was partially touched in the Tier 2 tranche — verify it's complete)
- [ ] Training data carries backend/fidelity provenance from receipts
- [ ] Inferential learnability contracts propagate into training artifacts
- [ ] If training corpus currently extracts binding metadata but not full receipt chain metadata (calibration receipt, adaptation receipt, shadow execution receipt), note the gap

## Tier 3 Verification Checks

### 3.1 Render provider contract chains

**File**: `src/world_model/sim_synth_physics/render_providers.py`

Verify:
- [ ] `compile_branch_render_provider_state()` produces `BranchRenderProviderState` with honest provider status for GGDS, NAG, LSD
- [ ] GGDS optimizer binding posture is complete (even if GPU execution is still blocked)
- [ ] NAG Gaussian renderer fallback chain from GGDS → NAG is specified
- [ ] LSD vector scene fallback logic exists
- [ ] All three provider families carry honest available/unavailable/stub posture — no silent stubs
- [ ] **Note**: render_providers currently does NOT emit typed receipts. Determine whether a `BranchRenderProviderReceipt` should exist alongside the state object for receipt-chain completeness.

### 3.2 Promotion/demotion machinery

**File**: `src/world_model/sim_synth_physics/promotion.py`

Verify:
- [ ] `resolve_helper()` correctly implements `disabled|auto|required` posture
- [ ] Promotion to "promoted" requires `benchmark_gate.ready == True`
- [ ] Weight assignment: 0.7 for promoted, 0.25 for shadow candidate, 0.0 for disabled/missing
- [ ] Demotion path: if a promoted helper fails benchmark evidence, does the machinery return to shadow_candidate? (Check whether this path exists or is only implicit)
- [ ] Promotion traces are emitted as part of compiler receipts (check that helper_status dicts flow into the compiled world state metadata)

### 3.3 Branch planner routing completeness

**Files**: `src/world_model/sim_synth_physics/branch_planner.py`, `branch_planner_runtime.py`

Verify:
- [ ] Heuristic path: produces `SyntheticBranchPlan` with `generation_mode` set correctly when benchmark_gate is not ready
- [ ] Learned path: routes through `LearnedBranchPlanner` (or equivalent) when promoted
- [ ] Fallback from learned to heuristic emits honest receipt or status trace
- [ ] Branch plans reference the correct backend fidelity and adaptation policy from the compiled physics context
- [ ] Branch planner does NOT silently skip when no planner is available — it falls back with receipt

### 3.4 Inferential yield scoring

**File**: `src/world_model/sim_synth_physics/inferential.py`

Verify:
- [ ] `build_simulation_job_inferential_contract()` produces contracts with frontier_gain, epiplexity_delta, transfer_score
- [ ] `agenda_score_with_inferential_prior()` correctly combines base ranking score with inferential contract
- [ ] Coverage gap rank from `rank_gaps_for_agenda()` feeds into frontier_gain
- [ ] Economic priority and trust priority are properly weighted in epiplexity_delta and transfer_score
- [ ] Adjusted yield scores react to backend fidelity selection (check whether `benchmark_signals` or physics context quality affects epiplexity_confidence)
- [ ] `benchmark_provenance_quality()` checks real backend provenance flags (semantic_grounding_non_heuristic, scene_tracks_backend_real, vision_backbone_real)

### 3.5 Randomization/calibration reaction to evidence

**Files**: `src/world_model/sim_synth_physics/randomization.py`, `calibration.py`

Verify:
- [ ] `compile_physics_adaptation_policy()` produces `PhysicsAdaptationPolicyState` from physics context
- [ ] Domain randomization regimes are derived from fidelity tier and hardware class
- [ ] Calibration receipts (`build_physics_calibration_receipt()`, `build_physics_adaptation_receipt()`) react to `PhysicsExecutionContract` route_status
- [ ] Calibration quality score reacts to benchmark gates and helper status
- [ ] Randomization axes for humanoid-target hardware classes (`unitree_g1`, `unitree_r1`, or equivalent) are specified — or the gap is honestly noted
- [ ] The schema/interface supports future reaction to `BackendRuntimeExecutionReceipt` outcomes, even if concrete evidence is still pending

### 3.6 Shadow execution receipt completeness

**File**: `src/world_model/sim_synth_physics/shadow_execution.py`

Verify:
- [ ] `materialize_backend_shadow_execution()` emits `BackendShadowExecutionReceipt` for supported backends
- [ ] Isaac and Holosoma shadow paths both exist and emit receipts
- [ ] Episode IDs and artifact refs are collected in the receipt
- [ ] Receipt distinguishes `shadow_only_preview` vs `shadow_with_data_harvest` modes (or equivalent distinction)
- [ ] Shadow execution consumes the Tier 2 runtime truth: does it use the deeper adapter ladder (runtime binding, adapter execution, adapter realization) or does it still jump directly from `PhysicsExecutionContract` to shadow materialization? If the latter, note whether the shadow path should be updated to thread through the same ladder.

## Tier 4 Test Requirements (After Tier 1 + 3 Verification)

### 4.2 Compiler round-trip test

Add a test that:
- Compiles a `SimSynthPhysicsWorldState` from mock inputs
- Verifies all expected state sub-objects are populated (physics context, adaptation policy, execution binding, asset contract, runtime bridge, synthetic branches, gen2sim admission, diffusion conditioning)
- Verifies all expected receipts are listed or producible
- Verifies `to_dict()` serialization round-trips without data loss
- Verifies runtime-binding metadata is accessible from the compiled state

### 4.3 Backend adapter smoke tests (remaining)

- [ ] PyBullet: correct metadata, no unitree claims, correct tabletop envelope
- [ ] All three backends emit `BackendAdapterDescriptor` with honest metadata fields

### 4.4 Work-order backlog integration test

- [ ] `runtime_work_orders.py` loads `NON_TRAINING_GPU_RUN_BACKLOG.json` without crash
- [ ] Environment variable substitution works
- [ ] Missing runtime targets produce "pending" (not "failed") work-orders
- [ ] `blocked_by_runtime_binding` status correctly triggers when binding_status is "binding_blocked"

## What Codex Should NOT Do

- Do not add new rungs to the runtime ladder
- Do not refactor existing compiler structure — verify and complete it
- Do not start Perception WM implementation
- Do not modify frozen Phase B baseline
- Do not add new WM-level abstractions

## Verification

After each tier:
```bash
python3 -m compileall src/ && pytest tests/ -v
pytest tests/world_model/ -v
```

## Required Handoff Artifact

After completing this verification tranche, Codex must emit an updated `docs/economic_world_model/claude_to_comment_on.md` with:

- Tranche Spec Coverage table mapping to this verification spec's items
- What Was Not Changed section for any untouched files
- Per-item verification result: passed / gap found / blocked by external
- Whether the Phase 1 boundary should now be considered structurally closed
- Any gaps that are honestly external vs gaps that are internal structural wiring issues

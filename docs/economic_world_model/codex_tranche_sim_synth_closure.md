# Codex Tranche: Sim / Synth / Physics WM Structural Closure

## Classification

- **Type**: implementation tranche (structural wiring completion)
- **Branch**: `codex/multi-wm-architecture-plan`
- **Priority**: highest — this is the current implementation center of gravity
- **Rationale**: Phase 1 architectural pieces are largely in place. The remaining work is verification, completion, and wiring of existing structural surfaces so the remaining Phase 1 blockers become honestly external (provider maturity, GPU, assets, calibration, benchmark density).

## Sequencing Context

The Perception / Grounding WM canonical state schema exists as a prepared spec (`codex_tranche_perception_wm_schema.md`). That spec is deliberately held as adjacent doctrine/schema work until this Sim / Synth / Physics closure tranche is far enough along.

Do NOT shift implementation priority to Perception WM while this tranche has implementable gaps.

## What Codex Should Implement

This tranche is organized into three priority tiers. Work top-down.

### Tier 1: Compiler and Receipt Chain Completeness (highest priority)

These ensure the core compilation pipeline assembles all canonical state and emits all required receipts.

#### 1.1. Compiler full-artifact assembly audit

**File**: `src/world_model/sim_synth_physics/compiler.py`

Verify and complete `compile_sim_synth_physics_world_state()` so it:
- Compiles all state objects in the documented order: physics context → adaptation policy → execution binding → robot asset contract → runtime bridge → synthetic branches → gen2sim admission → diffusion conditioning
- Emits all associated receipts: `PhysicsCalibrationReceipt`, `PhysicsAdaptationReceipt`, `BackendExecutionBindingReceipt`, `BackendRuntimeBridgeReceipt`, `BackendRuntimeWorkOrderReceipt`, branch render-provider receipts, simulation outcome receipts
- Includes inferential learnability contracts on branch plans
- Does not silently skip steps when a backend is unavailable — instead emits an honest unavailable receipt

#### 1.2. Physics contracts execution binding completeness

**File**: `src/world_model/sim_synth_physics/physics_contracts.py`

Verify `PhysicsExecutionContract` is:
- Built during compilation and included in `SimSynthPhysicsWorldState`
- Contains honest backend/fidelity/timestep contracts
- Emitted as a canonical state artifact (not just internal intermediate)

#### 1.3. Gen2Sim admission explicit receipt emission

**File**: `src/world_model/sim_synth_physics/gen2sim_admission.py`

Verify:
- Admission decisions are emitted as explicit `Gen2SimAdmissionState` receipts
- Decisions reference canonical asset contracts
- Real vs synthetic evidence provenance is tracked
- The per-branch admission path in `assess_local_branch_corpus_gen2sim()` emits per-stage tallies as canonical receipt metadata

#### 1.4. Training corpus receipt consumption

**File**: `src/world_model/sim_synth_physics/training_corpus.py`

Verify:
- Training corpus reads from `SimSynthPhysicsWorldState` canonical receipts
- No bespoke backend-specific JSON parsing outside the receipt chain
- Training data is tagged with backend/fidelity provenance from receipts
- Inferential learnability contracts propagate into training artifacts

### Tier 2: Backend Adapter and Runtime Wiring (medium priority)

These ensure each backend lane has honest, complete adapter metadata and runtime integration.

#### 2.1. Isaac backend adapter completeness

**Files**: `src/world_model/sim_synth_physics/adapters/backend_isaac.py`, `src/world_model/sim_synth_physics/adapters/isaac_unitree_deployment.py`

Verify:
- `describe_isaac_adapter()` emits full `BackendAdapterDescriptor` with all metadata fields (supports_receipt_harvest, domain_randomization, system_identification, etc.)
- `build_isaac_unitree_deployment_contract()` validates all required assets against normalized manifest
- Mode readiness logic (sim_eval, teleop_bridge, lerobot_eval, real_deployment) gates correctly on asset availability
- Integration with runtime_targets and runtime_layouts is complete

#### 2.2. Holosoma backend adapter wiring

**Files**: `src/world_model/sim_synth_physics/adapters/backend_holosoma.py`, `src/world_model/sim_synth_physics/backend_bindings.py`

Verify:
- Holosoma binding is properly routed in backend_bindings compilation
- Motion source counting is accurate
- Task preset list stays in sync with available presets
- Shadow execution path emits `BackendShadowExecutionReceipt` for holosoma shadow runs

#### 2.3. PyBullet adapter coverage

**File**: `src/world_model/sim_synth_physics/adapters/backend_pybullet.py`

Verify:
- Emits all required binding metadata
- Properly identifies fixed-base tabletop envelope
- Does not claim unitree_assets support
- Supports required domain-randomization axes for tabletop regime

#### 2.4. Runtime outcome parsers for all backends

**File**: `src/world_model/sim_synth_physics/runtime_outcome_parsers.py`

Verify parsers exist for:
- Isaac / Isaac Lab outcome artifacts
- Holosoma motion clip outcomes
- PyBullet sim outcomes
- All emit canonical `BackendRuntimeOutcomeReceipt`

#### 2.5. Runtime evidence harvesting

**File**: `src/world_model/sim_synth_physics/runtime_evidence.py`

Verify:
- Evidence is harvested from `BackendRuntimeLaunchReceipt` outputs
- Calibration truth is extracted from concrete runtime results
- Materialization artifacts are properly referenced in outcome receipts

### Tier 3: Render Provider, Promotion, and Inferential Completeness (important but lower urgency)

#### 3.1. Render provider contract chains

**File**: `src/world_model/sim_synth_physics/render_providers.py`

Verify:
- GGDS optimizer binding is complete
- NAG Gaussian renderer fallback chain works
- LSD vector scene fallback logic exists
- All three provider families emit `BranchRenderProviderState` with honest provider status

#### 3.2. Promotion/demotion machinery

**File**: `src/world_model/sim_synth_physics/promotion.py`

Verify:
- Learned backend selector promoted only when benchmark_gate ready
- Learned branch planner promoted only when benchmark_gate ready
- Demotion back to shadow triggered on evidence failure
- Helper weight assignment follows documented rules (0.7 promoted, 0.25 shadow candidate)

#### 3.3. Branch planner routing completeness

**Files**: `src/world_model/sim_synth_physics/branch_planner.py`, `branch_planner_runtime.py`

Verify:
- Routes through heuristic path when benchmark_gate not ready
- Routes through LearnedBranchPlanner when promoted
- Emits `SyntheticBranchPlan` with correct generation_mode in both paths
- Fallback from learned to heuristic emits honest receipt

#### 3.4. Inferential yield scoring

**File**: `src/world_model/sim_synth_physics/inferential.py`

Verify:
- Branch yield scores incorporate coverage gap rank from `rank_gaps_for_agenda()`
- Economic priority and trust priority properly weighted
- Adjusted yield scores react to backend fidelity selection
- Inferential learnability contracts carry frontier gain, epiplexity delta, transfer score

#### 3.5. Randomization/calibration reaction to evidence

**Files**: `src/world_model/sim_synth_physics/randomization.py`, `calibration.py`

Verify:
- Domain randomization regimes are derived from `PhysicsAdaptationPolicyState`
- Calibration profiles can react to `BackendRuntimeExecutionReceipt` outcomes (at least the schema/interface supports this even if concrete evidence is still pending)
- Randomization axes for humanoid-target hardware classes are specified

#### 3.6. Shadow execution receipt completeness

**File**: `src/world_model/sim_synth_physics/shadow_execution.py`

Verify:
- Shadow work-order paths emit `BackendShadowExecutionReceipt` for all supported backends
- Episode IDs and artifact refs are collected correctly
- "shadow_only_preview" vs "shadow_with_data_harvest" modes are distinguished

### Tier 4: Test Coverage (complete after Tiers 1-3)

#### 4.1. Isaac/Unitree deployment contract tests

Add tests for:
- `build_isaac_unitree_deployment_contract()` with missing assets
- Mode readiness logic (sim_eval, teleop_bridge, lerobot_eval, physical_deploy)
- Preferred profile selection

#### 4.2. Compiler round-trip test

Add a test that:
- Compiles a `SimSynthPhysicsWorldState` from mock inputs
- Verifies all expected receipts are present
- Verifies all state sub-objects are populated
- Verifies `to_dict()` serialization round-trips

#### 4.3. Backend adapter smoke tests

Add/verify tests for each backend adapter:
- pybullet: correct metadata, no unitree claims
- holosoma: correct metadata, correct shadow posture
- isaac: correct metadata, honest unavailable when missing

#### 4.4. Work-order backlog integration test

Verify:
- `runtime_work_orders.py` loads NON_TRAINING_GPU_RUN_BACKLOG.json without crash
- Environment variable substitution works
- Missing runtime targets produce pending (not failed) work-orders

## What Codex Should NOT Do

- Do not refactor existing compiler structure — verify and complete it
- Do not implement concrete Isaac/Holosoma/GGDS execution (blocked by GPU/provider)
- Do not modify frozen Phase B baseline
- Do not start Perception WM implementation — that is the next phase
- Do not add new WM-level abstractions; this tranche is about closing the existing Phase 1 boundary

## Verification

After each tier:
```bash
python3 -m compileall src/ && pytest tests/ -v
pytest tests/world_model/ -v
```

## Definition of Done

Phase 1 Sim / Synth / Physics WM counts as structurally closed when:
- All Tier 1-3 verifications pass
- All Tier 4 tests pass
- The remaining gap list is limited to items blocked by external providers, GPU, real data, calibration, or benchmark evidence
- No implementable structural wiring gaps remain

## Required Handoff Artifact

After completing this tranche (or after completing a meaningful sub-tranche), Codex must emit:

**`docs/economic_world_model/claude_to_comment_on.md`**

containing:
- which tiers/items were completed
- which verification checks passed or failed
- which remaining gaps are honestly blocked by external factors
- any structural issues discovered during verification
- whether the Phase 1 boundary should be considered structurally closed
- any doctrinal questions about the sim/synth/physics WM boundary

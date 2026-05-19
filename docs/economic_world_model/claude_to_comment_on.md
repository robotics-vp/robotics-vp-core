# Claude Commentary Artifact

## Current Status

- **Date**: 2026-04-03
- **Branch**: `codex/multi-wm-architecture-plan`
- **Primary implementation center**: Phase 2 Perception / Grounding WM
- **Phase 1 posture**: structurally closed on audited internal surfaces; remaining blockers are external GPU/runtime/asset items tracked in `docs/economic_world_model/phase1_external_gpu_runtime_backlog.md`; Habitat-derived Sim/Synth/Physics adoption track remains an explicit reopenable Phase 1.x item
- **Latest pass**: Claude implementation pass — first bounded neural seam + receipt emission

## Tranche 2.1 Coverage (Codex — prior pass)

Moved Phase 2 from schema/doctrine presence toward the first
loop-facing subsystem behavior.

Implemented:

- `src/world_model/perception_grounding/compiler.py`
  - new `compile_perception_grounding_world_state(...)`
  - compiles canonical Perception / Grounding state from real upstream inputs
- `src/world_model/perception_grounding/__init__.py`
  - exports the compiler
- `src/world_model/sim_synth_physics/adapters/semantic_inputs.py`
  - consumes compiled Perception state and bridge summaries into live sim-synth semantic context
- `src/world_model/sim_synth_physics/compiler.py`
  - accepts `perception_grounding_state=` and threads it into the canonical sim-synth input context
- `src/vla/rollout_labeler.py`
  - compiles Perception / Grounding state from real episode scene tracks + semantic evidence
  - consumes annotation-bridge outputs into rollout labeling tags and metadata
- `src/vision/backbone_stub.py`
  - now exposes typed provider/advisory posture through `VisionBackboneProviderContract`
- `src/policies/vision_encoder.py`
  - exposes the same provider-contract posture

## Tranche 2.2a Coverage (Claude — this pass)

This pass landed the **first bounded neural seam** and **receipt emission**,
satisfying the anti-heuristic-without-neuralization standard.

Implemented:

- `src/world_model/perception_grounding/neural_seams.py` **(NEW)**
  - `EvidenceFusionSeam(torch.nn.Module)` — real set-attention module
    - 2-head multi-head self-attention over provider evidence tokens
    - input projection (d=12 → d_model=32), self-attention, FFN, layer norm
    - per-provider weight head (softmax-normalized)
    - pooled confidence head (sigmoid)
    - ~10-50K trainable params (deliberately tiny — smallest useful seam)
    - `heuristic_init()` classmethod for conservative initialization
    - `describe()` for receipt/logging metadata
    - `param_count()` introspection
  - `encode_provider_features()` — encodes provider kind, availability,
    truth class, and belief-state signals into a typed feature tensor
  - `PROVIDER_KIND_VOCAB`, `TRUTH_CLASS_SCORES` — typed vocabularies

- `src/world_model/perception_grounding/compiler.py` **(MODIFIED)**
  - `_evidence_routing()` now **branches on promotion stage**:
    - `"heuristic_fallback"`: existing hardcoded weighted fusion (unchanged)
    - `"promoted"` + seam provided: neural seam forward pass produces weights + confidence
    - fallback on any neural seam error (graceful degradation)
  - `_evidence_routing()` now **emits `EvidenceFusionReceipt`** on every call:
    - records fusion method, provider weights, confidence, disagreement
    - records whether neural seam was used or heuristic fallback
    - receipt stored in state metadata as `evidence_fusion_receipt`
  - `compile_perception_grounding_world_state()` accepts optional `evidence_fusion_seam=`
  - New `PerceptionCompilationResult` dataclass: `(state, receipts)`
  - New `compile_perception_grounding_with_receipts()` function

- `src/world_model/perception_grounding/__init__.py` **(MODIFIED)**
  - exports `EvidenceFusionSeam`, `encode_provider_features`,
    `PerceptionCompilationResult`, `compile_perception_grounding_with_receipts`

- `tests/test_perception_grounding_compiler.py` **(MODIFIED — 9 new tests)**
  - `test_evidence_fusion_seam_forward_pass` — seam produces valid weights + confidence
  - `test_evidence_fusion_seam_batched` — batched input works
  - `test_evidence_fusion_seam_with_mask` — masked providers get zero weight
  - `test_compiler_backward_compat_without_seam` — existing API unchanged
  - `test_compiler_with_neural_seam_promoted` — promoted path uses seam
  - `test_compiler_neural_seam_fallback_without_benchmark` — no benchmark = heuristic
  - `test_compile_with_receipts_returns_typed_result` — receipt structure correct
  - `test_compile_with_receipts_neural_seam` — receipt records neural seam use
  - `test_evidence_fusion_seam_describe` — introspection metadata correct

## What Topologically Became More Real (cumulative)

- Evidence fusion is no longer permanently heuristic. A real `torch.nn.Module`
  sits behind the promotion gate and executes when `benchmark_signals` promote.
- The promotion machinery has its first real consumer — the neural seam forward
  pass only runs at `"promoted"` stage, controlled by the existing
  `resolve_evidence_fusion_helper` posture.
- Receipt emission is live. The compiler now emits a typed
  `EvidenceFusionReceipt` on every compilation, recording which path was taken,
  what weights were produced, and whether the neural or heuristic path ran.
- The anti-heuristic-without-neuralization standard is now satisfied at the
  evidence fusion surface — the hardcoded 0.55/0.25/0.15/0.05 is explicitly
  transitional with a real neural successor codepath.

## What Internal Incompleteness Was Fixed (this pass)

1. Missing neural seam codepath behind promotion posture — **FIXED**: `EvidenceFusionSeam`
2. Missing evidence fusion receipt emission — **FIXED**: `EvidenceFusionReceipt` emitted on every compilation
3. Missing promotion-stage branching in compiler — **FIXED**: `_evidence_routing` branches on stage
4. Missing `compile_with_receipts` API — **FIXED**: `PerceptionCompilationResult` returned

## What Was Not Changed

- No Phase 1 Sim / Synth / Physics work was reopened.
- No new top-level WM was introduced.
- No GPU/provider bring-up was faked.
- No state.py schema changes.
- No semantic_bridges.py changes.
- No provider_contracts.py changes.
- No monolithic semantic model or mother-latent was introduced.
- Backward compatibility fully preserved — all existing callers unaffected.

## SemanticVLA Treatment

Unchanged from prior pass. `SemanticVLA` remains explicitly transitional,
scaffolding-only, backward-compatible.

## Phase 2 Closure Assessment

Current authoritative read: see
`docs/economic_world_model/phase2_closure_assessment.md`.

- Category A: `0`
- Category B: real provider / GPU / calibration / real-data / held-out-evidence
  blockers only
- Category C: `0`
- maturity: `shadow_runtime`

The final local structural seam closed on 2026-05-18 was live
`SemanticBridgeReceipt` emission for the active WM-native bridge family.

## Robust-Subsystem Read

The Perception / Grounding WM now:

- compiles canonical state from real inputs
- owns typed heuristic-fallback plus bounded learned seam posture
- has multiple real neural seam codepaths behind promotion posture
- emits the audited live receipt family, including semantic bridge receipts
- produces bridge outputs with named downstream preconditions
- changes downstream behavior in three existing loops

It is at `shadow_runtime` with bounded neural codepaths available behind
benchmark gating. The earlier embodiment-facing usefulness proof is now landed
through `embodiment_shadow_consumer.py`.

## Recommendation

- Treat Phase 2 as structurally closure-ready, not provider-ready.
- The last cheap local hardening pass is now landed: LeRobot projection-adapter
  parity for `vision_backbone_projection`.
- Per the roadmap, the current implementation center now returns to the queued
  Phase 1.x Sim / Synth / Physics leg, not an immediate jump to Phase 3.

## Next Best Tranche

Current re-entry status after the Phase 2 pocket:

1. returned to the Phase 1.x Sim / Synth / Physics leg
2. landed the first CPU-local tranche:
   - shared surface family:
     `TaskMeasurementSurface`, `SceneHierarchyState`,
     `DifferentiablePhysicsProviderState`, `SurrogatePhysicsProviderState`
   - paired receipt family:
     `TaskMeasurementReceipt`, `SimRealGapReceipt`, `BackendMismatchReceipt`,
     `SurrogatePhysicsReceipt`, `SurrogateCalibrationReceipt`
   - simulator/task contract pair:
     `SimulatorBackendContractState`, `TaskDefinitionContractState`
   - local `camera_geometry.py` utilities
   - local `VectorizedSimRunner` batch facade
3. next best local follow-on should deepen use of these surfaces rather than
   inventing another ontology: scene-hierarchy, transfer-evidence,
   branch-validity / reject-filter, geometry-backed sensor-alignment,
   replay-validity / task-consistency, runtime receipt-manifest, and manifest
   validation consumers are now live in runtime artifacts and training-row
   harvest while external GPU/provider work remains blocked

### Habitat-derived Sim/Synth/Physics adoption track reminder

Unchanged from prior pass. The biggest remaining opportunity sits in
Sim / Synth / Physics WM. See `roadmap.md` and `multi_wm_architecture_plan.md`.

## Doctrine Updates Landed (prior pass — unchanged)

### Future Economic WM

See `doctrine_economic_wm_future_architecture.md`.

### Meta-Regal-Node Superposition WM

See `doctrine_meta_regal_node_wm.md`.

### Anti-heuristic-without-neuralization + embodiment-facing usefulness

- Structural preparation (receipts, promotion gates) is necessary but not
  sufficient — bounded neural seams must follow
- bounded neural seams now exist across the active Phase 2 successor path
- embodiment-facing consumption is now landed
- Habitat extraction is not exhausted — biggest remaining opportunity is
  Sim/Synth/Physics

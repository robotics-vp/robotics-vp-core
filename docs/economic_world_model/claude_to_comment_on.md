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

### Category A: still internal

Phase 2 is not closed yet. Remaining internal items include:

1. ~~learned/helper seams exist behind typed posture, but the current compiler path is still heuristic-only~~ **RESOLVED**: evidence fusion seam is now a real neural codepath
2. provider invocation / provider-availability / deployment-resource receipts are typed but only `EvidenceFusionReceipt` is live — remaining receipt types need emission
3. provider registry / install/runtime scan path is not yet compiled into Perception WM truth
4. replay/training export for Perception WM state and bridge outputs is not yet its own dedicated path
5. downstream consumption is present, but still narrow:
   - one Sim / Synth semantic-context consumer
   - one annotation/rollout-labeling consumer
   - no embodiment-facing or economic-facing shadow consumer yet
6. annotation bridge projection heads not yet neural (next-priority neural seam)
7. dimensional regime markers not yet added to state/bridge metadata

### Category B: external

- real SAM 3 / 3.1, DINOv2/SigLIP, V-JEPA 2, and depth runtime on GPU hosts
- real provider weights/checkpoints and multi-provider concurrent execution
- real robot/egocentric perception streams, calibration, and long-horizon humanoid corpora

### Category C

- none newly unresolved

## Robust-Subsystem Read

The Perception / Grounding WM now:

- compiles canonical state from real inputs
- owns real heuristic fusion/evidence-routing posture
- **has its first real neural seam behind promotion posture** (evidence fusion)
- emits typed receipts from compilation (evidence fusion receipt)
- produces bridge outputs with named downstream preconditions
- changes downstream behavior in two existing loops

It is at `shadow_runtime` with the first `bounded_runtime_authority` codepath
(evidence fusion seam) behind benchmark gating.

**Critical remaining proof of subsystem usefulness**: embodiment-facing
affordance / action-relevance shadow consumption. Without this, Perception
risks remaining a well-instrumented semantic shell that is structurally
complete but not actually useful for robot control.

## Recommendation

- Keep Phase 2 as the implementation center.
- Do not reopen Phase 1 unless new external runtime/assets arrive or a direct contradiction appears.
- The neural seam landing shifts the next-priority work toward:
  1. Additional receipt types (provider availability, deployment resource, bridge receipts)
  2. Embodiment-facing shadow consumer skeleton
  3. Annotation bridge projection heads (second neural seam)
  4. Provider contract → compiler connection
  5. Dimensional regime markers

## Next Best Tranche (Tranche 2.2b — for Codex)

### Priority 1: Additional receipt emission

Extend `compile_perception_grounding_with_receipts` to emit:

- `ProviderAvailabilityReceipt` per provider
- `PerceptionContributionReceipt` per compilation
- `SemanticBridgeReceipt` per active bridge

### Priority 2: Embodiment-facing shadow consumer skeleton

A minimal consumer that reads
`perception_grounding_state.semantic_bridge_registry.embodiment_bridge` and
produces typed output. This validates the bridge output shape and prepares the
Phase 3 interface contract.

### Priority 3: Annotation bridge projection heads (second neural seam)

Tiny learned MLPs for the annotation bridge's object→label/affordance/risk
heads. The annotation bridge is already functionally load-bearing for training
dataset formation. This is the second-highest-priority neural seam.

### Priority 4: Provider contract → compiler connection

The compiler should accept an optional `PerceptionProviderRegistry` and build
`ProviderSurfaceState` from the real typed contracts instead of inferring
providers from argument presence.

### Priority 5: Dimensional regime + bridge input source markers

- Add `feature_dim_regime` to `SceneGraphState` and bridge state metadata.
  Values: `"heuristic_d8"` (current), `"provider_d128"` (target).
- Add `bridge_input_source` to `SemanticBridgeRegistry` metadata.
  Current: `"semantic_world_model_heuristic"`. Target: `"canonical_scene_graph_substrate"`.

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
- **First bounded neural seam now landed**: `EvidenceFusionSeam`
- Embodiment-facing consumption is the next critical proof of Perception
  subsystem usefulness
- Habitat extraction is not exhausted — biggest remaining opportunity is
  Sim/Synth/Physics

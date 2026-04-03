# Claude Commentary Artifact

## Current Status

- **Date**: 2026-04-03
- **Branch**: `codex/multi-wm-architecture-plan`
- **Primary implementation center**: Phase 2 Perception / Grounding WM
- **Phase 1 posture**: structurally closed on audited internal surfaces; remaining blockers are external GPU/runtime/asset items tracked in `docs/economic_world_model/phase1_external_gpu_runtime_backlog.md`; Habitat-derived Sim/Synth/Physics adoption track remains an explicit reopenable Phase 1.x item

## Tranche Coverage

This pass moved Phase 2 from schema/doctrine presence toward the first
loop-facing subsystem behavior.

Implemented:

- `src/world_model/perception_grounding/compiler.py`
  - new `compile_perception_grounding_world_state(...)`
  - compiles canonical Perception / Grounding state from real upstream inputs:
    - scene tracks
    - belief state
    - VLA semantic evidence
    - existing semantic-world-model heuristics
- `src/world_model/perception_grounding/__init__.py`
  - exports the compiler
- `src/world_model/sim_synth_physics/adapters/semantic_inputs.py`
  - consumes compiled Perception state and bridge summaries into live sim-synth semantic context
  - now emits perception-backed inferential summary values instead of only raw semantic passthrough
- `src/world_model/sim_synth_physics/compiler.py`
  - accepts `perception_grounding_state=` and threads it into the canonical sim-synth input context
- `src/vla/rollout_labeler.py`
  - compiles Perception / Grounding state from real episode scene tracks + semantic evidence
  - consumes annotation-bridge outputs into rollout labeling tags and metadata
- `src/vision/backbone_stub.py`
  - now exposes typed provider/advisory posture through `VisionBackboneProviderContract`
  - latent metadata now carries explicit stub/advisory truth
- `src/policies/vision_encoder.py`
  - exposes the same provider-contract posture

## What Topologically Became More Real

- Perception / Grounding WM is no longer only a schema package. It now owns a
  real compiler path that produces canonical scene graph, temporal grounding,
  evidence routing, provider/dataset/task/resource surfaces, and a heuristic
  semantic-bridge registry from real upstream inputs.
- The semantic successor family is no longer merely declared. The first bridge
  outputs are compiled and downstream-consumed:
  - Sim / Synth semantic bridge now affects sim-synth semantic context
  - Annotation / evidence semantic bridge now affects rollout-labeling tags and row metadata
- `VisionBackboneStub` is no longer ambient placeholder functionality. It now
  declares explicit `stub_smoke_only` provider truth and advisory posture.

## What Internal Incompleteness Was Fixed

Fixed in this pass:

1. Missing Perception compiler/runtime path
2. Missing first downstream Sim / Synth shadow consumer
3. Missing first downstream annotation/evidence shadow consumer
4. Missing typed provider/advisory posture for `backbone_stub.py`
5. Missing first functional semantic bridge preconditions in live compiled outputs

## What Was Not Changed

- No Phase 1 Sim / Synth / Physics work was reopened.
- No new top-level WM was introduced.
- No GPU/provider bring-up was faked.
- No bounded runtime authority was given to Perception helpers.
- No monolithic semantic model or mother-latent was introduced.

## SemanticVLA Treatment

`SemanticVLA` remains:

- explicitly transitional
- scaffolding-only
- backward-compatible

It is **not** the semantic owner. The current semantic owner/successor posture is:

1. canonical Perception / Grounding semantic substrate
2. WM-native semantic bridge family
3. provider-backed / fusion-backed evidence entering that substrate
4. downstream WM-specific semantic consumption

## Phase 2 Closure Assessment

### Category A: still internal

Phase 2 is not closed yet. Remaining internal items include:

1. provider invocation / provider-availability / deployment-resource receipts are typed but not yet emitted by the live compiler/runtime path
2. provider registry / install/runtime scan path is not yet compiled into Perception WM truth the way late Phase 1 did for sim-synth
3. learned/helper seams exist behind typed posture, but the current compiler path is still heuristic-only shadow runtime
4. replay/training export for Perception WM state and bridge outputs is not yet its own dedicated path
5. downstream consumption is present, but still narrow:
   - one Sim / Synth semantic-context consumer
   - one annotation/rollout-labeling consumer
   - no embodiment-facing or economic-facing shadow consumer yet

### Category B: external

- real SAM 3 / 3.1, DINOv2/SigLIP, V-JEPA 2, and depth runtime on GPU hosts
- real provider weights/checkpoints and multi-provider concurrent execution
- real robot/egocentric perception streams, calibration, and long-horizon humanoid corpora

### Category C

- none newly unresolved on the audited compiler-and-consumer tranche

## Robust-Subsystem Read

The Perception / Grounding WM is now beginning to satisfy the
subsystem-within-WM bar:

- it compiles canonical state from real inputs
- it owns real heuristic fusion/evidence-routing posture
- it produces bridge outputs with named downstream preconditions
- it changes downstream behavior in two existing loops

It is still only at early `shadow_runtime`, not `bounded_runtime_authority`.

**Critical remaining proof of subsystem usefulness**: embodiment-facing
affordance / action-relevance shadow consumption. Without this, Perception
risks remaining a well-instrumented semantic shell that is structurally
complete but not actually useful for robot control. The embodiment bridge is
compiled but has no downstream consumer. Wiring that consumer is how
Perception starts becoming obviously relevant to actual G1-operable loop
behavior.

## Why The Remaining Gaps Are Honest

The gaps above are no longer “missing schema” or “missing doctrine” gaps.
They are now the correct next-stage gaps:

- provider/runtime truth emission
- richer replay/export surfaces
- more downstream consumers
- later GPU/provider bring-up

That is the right posture. The branch should not regress to treating
Perception as a beautiful contract shell.

## Recommendation

- Keep Phase 2 as the implementation center.
- Do not reopen Phase 1 unless new external runtime/assets arrive or a direct contradiction appears.
- Parallel Phase 3 prep is acceptable, but Phase 2 should keep primary implementation priority until:
  - Perception receipts are live
  - provider/runtime truth is compiled
  - at least one more downstream WM consumes the bridge family in shadow mode

## Next Best Tranche (Tranche 2.2)

### Priority 1: Receipt emission + promotion-gate wiring

The compiler must return receipts alongside state. Each compilation should emit:

- `EvidenceFusionReceipt` (fusion method, provider weights, confidence)
- `ProviderAvailabilityReceipt` per provider (availability, truth class)
- `PerceptionContributionReceipt` (grounding quality, semantic yield)
- `SemanticBridgeReceipt` per active bridge (quality, downstream usefulness)

The evidence fusion path in `_evidence_routing()` must branch on
`evidence_helper["promotion_stage"]` and record which path was taken. At
`heuristic_fallback`, it uses the current weighted fusion. At `promoted`, it
should call through a learned fusion path (which can initially raise
NotImplementedError or fall back with receipt). The branch must exist.

### Priority 2: Provider contract → compiler connection

The compiler should accept an optional `PerceptionProviderRegistry` and build
`ProviderSurfaceState` from the real typed contracts instead of inferring
providers from argument presence. Current inference is the fallback, not the
only path.

### Priority 3: Embodiment-facing shadow consumer skeleton

A minimal consumer that reads
`perception_grounding_state.semantic_bridge_registry.embodiment_bridge` and
produces typed output. This validates the bridge output shape and prepares the
Phase 3 interface contract.

### Priority 4: Dimensional regime + bridge input source markers

- Add `feature_dim_regime` to `SceneGraphState` and bridge state metadata.
  Values: `"heuristic_d8"` (current), `"provider_d128"` (target).
- Add `bridge_input_source` to `SemanticBridgeRegistry` metadata.
  Current: `"semantic_world_model_heuristic"`. Target: `"canonical_scene_graph_substrate"`.

### Priority 5: First bounded neural seam implementation

**Critical**: Do not let receipt/gating/provider truth work become a reason to
postpone bounded neural implementation indefinitely. Once Priorities 1-2 are
landed, the very next step should be to start implementing the first bounded
neural seam. Candidates, in priority order:

1. **Evidence fusion seam**: a tiny learned set-attention module (100K-500K
   params, 2-4 attention heads) that can be swapped in at the `promoted`
   promotion stage. Initially heuristic-initialized, disabled by default,
   benchmark-gated. Even if it starts as a parameterized version of the
   current weighted fusion, it should be a real `torch.nn.Module` behind the
   promotion posture.
2. **Annotation bridge projection heads**: the annotation bridge's
   object→label/affordance/risk heads as tiny learned MLPs. These are the
   smallest useful neural seam and the annotation bridge is already
   functionally load-bearing for training dataset formation.
3. **Provider calibration/projection head**: if DINOv2 weights become locally
   available, a learned projection head (d=1024→d=128) behind the
   `VisionBackboneProviderContract` projection_head_posture.

The rule is: heuristic fusion/bridge/graph paths are transitional priors. The
compiler/runtime path should increasingly prepare for and then execute immediate
bounded neural substitution.

### Habitat-derived Sim/Synth/Physics adoption track reminder

The Habitat extraction is not exhausted. The biggest remaining opportunity
sits in Sim / Synth / Physics WM for:

- simulator/task separation discipline
- articulated embodiment + sensor config
- scene/measurement harness patterns
- vectorized runtime/eval patterns
- camera geometry / view-warp utilities

This should be named as a specific open Phase 1.x adoption item, not forgotten.
See updated `roadmap.md` and `multi_wm_architecture_plan.md`.

## Doctrine Updates Landed This Pass

### Future Economic WM

The Economic WM is now framed as a neuralizable, scalable, typed
allocator-governor — the canonical world model of productive flow,
dissipation, and allocative opportunity. Key properties:

- multi-timescale (fast/meso/slow-adiabatic variable split)
- asymmetric upward/downward transport
- four-component decomposition (state estimator → dynamics → allocator →
  governance/reciprocity)
- staged neuralization (typed ontology first → neural estimation → neural
  dynamics → neural allocator → local shaping compilers)
- quant-inspired imports as algorithmic patterns: coherent risk, distributional
  Pareto, regime switching, risk budgeting, stress testing, execution-cost
  awareness

See `doctrine_economic_wm_future_architecture.md`.

### Meta-Regal-Node Superposition WM

The Economic WM is explicitly **not the sovereign governor** of the stack.
Above it sits the meta-regal-node WM that composes multiple
domain-governance nodes (economics, anti-reward-hacking, plausibility, safety,
deployment truth, data value, later coordination) under regime-sensitive
Pareto, veto, and admissibility logic.

Two fundamentally different Pareto problems:

- **intra-domain** (within Economic WM): throughput vs energy vs wear etc.
- **inter-domain** (within meta-regal-node): economics vs safety vs
  plausibility etc. More fundamental: governs whether intra-domain
  optimization can be trusted.

The architecture preserves governance pluralism: no single domain ontology
(including economics) can silently redefine the others.

See `doctrine_meta_regal_node_wm.md`.

### Anti-heuristic-without-neuralization + embodiment-facing usefulness

- Structural preparation (receipts, promotion gates) is necessary but not
  sufficient — bounded neural seams must follow
- Embodiment-facing consumption is the next critical proof of Perception
  subsystem usefulness — prevents Perception from being a semantic shell
- Habitat extraction is not exhausted — biggest remaining opportunity is
  Sim/Synth/Physics, now named as a 3-tier adoption track

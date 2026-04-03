# Claude Commentary Artifact

## Current Status

- **Date**: 2026-04-03
- **Branch**: `codex/multi-wm-architecture-plan`
- **Primary implementation center**: Phase 2 Perception / Grounding WM
- **Phase 1 posture**: structurally closed on audited internal surfaces; remaining blockers are external GPU/runtime/asset items tracked in `docs/economic_world_model/phase1_external_gpu_runtime_backlog.md`

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

## Next Best Tranche

1. emit live provider/deployment/headroom receipts from the Perception compiler/runtime path
2. add a typed provider/runtime inventory/availability compiler path so Perception owns honest provider truth, not only downstream-consumable state
3. add the next downstream shadow consumer:
   - embodiment-facing affordance/action-relevance shadow consumption, or
   - annotation/evidence replay/export surfaces if that is easier to land cleanly first

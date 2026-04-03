# Codex Tranche: Perception / Grounding WM Closure Tranche 2.1

## Classification

- **Type**: implementation continuation after schema/doctrine reconciliation
- **Branch**: `codex/multi-wm-architecture-plan`
- **Priority**: ACTIVE
- **Sequencing**: follows audited late-Phase-1 closure; Phase 1 remains reopened only by new external runtime/assets or a direct contradiction

## Current Branch Truth

Already landed on this branch:

- canonical Phase 2 Perception / Grounding WM package in `src/world_model/perception_grounding/`
- canonical semantic substrate state:
  - `ObjectTrackState`
  - `SceneEdge`
  - `SceneGraphState`
  - `TemporalGroundingState`
  - `EvidenceRoutingState`
  - `PerceptionGroundingWorldState`
- semantic successor stack:
  - `SemanticBridgeRegistry`
  - Sim / Synth bridge state
  - Embodiment bridge state
  - Annotation / evidence bridge state
  - Economic bridge state
- lower-WM provider/dataset/task/deployment-resource surfaces:
  - `ProviderSurfaceState`
  - `DatasetSurfaceState`
  - `TaskMeasurementSurface`
  - `DeploymentResourceSurface`
  - `ComputeEnvelopeState`
  - `InferenceCapacityState`
  - `BatteryState`
  - `ThermalState`
- typed receipts:
  - `ProviderAvailabilityReceipt`
  - `ProviderInvocationReceipt`
  - `GroundingCalibrationReceipt`
  - `InferenceHeadroomReceipt`
  - `DeploymentResourceReceipt`
  - `EvidenceFusionReceipt`
  - `TemporalGroundingReceipt`
  - `PerceptionContributionReceipt`
- helper promotion/demotion:
  - graph transformer
  - temporal grounding
  - evidence fusion
  - semantic bridges
- `SemanticVLA` explicitly transitional/scaffolding-only with successor metadata

## What This Tranche Should Do

This is no longer a pure schema-only tranche. The next highest-leverage work is
to make the Phase 2 package compiler- and consumer-shaped.

### 1. Compiler / runtime skeleton

Build `src/world_model/perception_grounding/compiler.py` with:

- `compile_perception_grounding_world_state(...)`

Inputs should include, where available:

- scene tracks / scene-track artifacts
- belief/evidence state
- provider registry or provider statuses
- optional semantic-evidence / teacher hints
- optional deployment/resource posture

Outputs should include:

- `PerceptionGroundingWorldState`
- typed receipts for provider availability, calibration, and runtime posture

This should stay honest:

- real inputs when available
- reduced-quality but explicit state when providers are unavailable
- no silent provider masquerade

### 2. Provider-contract consumer wiring

Put the current placeholder consumers behind typed Phase 2 provider posture:

- `src/vision/backbone_stub.py`
- any remaining `SemanticVLA` callers that still treat it as a semantic owner

The goal is not to “improve” the stub. The goal is to make placeholder use
structurally visible through provider/advisory posture.

### 3. First downstream shadow consumers

Add the first Perception-WM shadow consumption hooks:

- Sim / Synth / Physics shadow consumer
- annotation / semantic-evidence shadow consumer

This should be typed and additive:

- do not rewrite the downstream WM
- do not give Phase 2 bounded authority yet
- do make the canonical Perception state consumable in shadow mode

## Bridge Preconditions To Preserve

The next tranche must keep the stronger semantic-successor posture explicit.

### Sim / Synth bridge preconditions

- object preservation
- synthetic-vs-real semantic alignment
- branch evaluation relevance
- branch-outcome semantic receipts

### Embodiment bridge preconditions

- affordance
- action relevance
- bodily-feasibility relevance
- object-task relation
- later resource-conditioned action semantics

### Annotation / semantic-evidence bridge preconditions

- object-linked primitive/event crosswalk
- failure/recovery labeling
- teacher/runtime semantic alignment
- training-dataset formation

### Economic bridge preconditions

- grounding quality
- semantic contribution
- action-relevant structural yield
- later allocation-facing fixed-dimensional summaries

These preconditions should stay named now even where the consuming WM comes
later. Phase 2 lays the semantic foundation for them.

## Habitat-Inspired Constraints To Preserve

Use Habitat-style patterns only as design-pattern donors:

- dataset/world inventory layer
- provider/runtime layer
- task/measurement layer
- deployment/resource layer
- vectorized runtime/eval discipline
- explicit sensor/config posture

Do not:

- flatten the stack into one environment object
- let providers become truth owners
- dissolve WM ownership boundaries

## What This Tranche Should Not Do

- no new top-level WM
- no mother-latent / monolithic semantic blob
- no reopening Phase 1 because of already-external blockers
- no direct RL-first bridge implementation
- no GPU/provider bring-up masquerading as local closure work

## Verification

Minimum:

```bash
python3 -m compileall src -q
python3 -m ruff check src/world_model/perception_grounding src/vla/semantic_vla.py tests/test_perception_grounding_world_model.py
python3 -m pytest -q tests/test_perception_grounding_world_model.py
```

When the first downstream consumer lands, extend with the affected shadow tests.

## Required Handoff Artifact

Keep `docs/economic_world_model/claude_to_comment_on.md` as a single clean
current-state artifact and make it answer:

- what internal incompleteness was fixed
- whether the semantic successor stack is fully verified on audited surfaces
- what remains internal vs external
- what the next Phase 2 consumer/compiler tranche should be

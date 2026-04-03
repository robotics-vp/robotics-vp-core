# Doctrine: Provider / Dataset / Resource Surfaces For Phase 2

## Purpose

Phase 2 Perception / Grounding WM should not stop at canonical object and
relation state. It also needs explicit lower-WM surfaces for:

- dataset/world inventory
- provider/runtime availability
- task/measurement state
- deployment/resource posture

This borrows the useful separation patterns from Habitat-style stacks without
letting Habitat become the ontology or truth owner in this repo.

Relevant design-pattern donors:

- Habitat-Sim README
- Habitat-Lab docs
- Habitat-Lab `Measure` pattern
- Habitat-Lab `VectorEnv` pattern

What we are borrowing from them:

- explicit dataset/world inventory separation
- explicit simulator/provider separation
- explicit task/measurement surfaces
- vectorized runtime/eval discipline
- explicit sensor/config posture

What we are not borrowing:

- a single environment object as the master ontology
- provider-owned semantic truth
- flattening WM boundaries into one runtime container

## Layering Rule

Adopt the pattern:

1. **Dataset/world inventory layer**
2. **Provider/runtime layer**
3. **Task/measurement layer**
4. **Deployment/resource layer**
5. **WM-owned canonical state and semantic bridges above those layers**

Do not collapse these into one environment object. In this repo:

- the Perception / Grounding WM owns the typed state
- external providers remain providers
- receipts remain replayable and benchmarkable
- later Economic WM allocation consumes these surfaces instead of inventing them

## Contract Families

Phase 2 should explicitly name and preserve:

- `ProviderSurfaceState`
- `DatasetSurfaceState`
- `TaskMeasurementSurface`
- `DeploymentResourceSurface`
- `ComputeEnvelopeState`
- `InferenceCapacityState`
- `BatteryState`
- `ThermalState`

Typed receipts that should accompany them:

- `ProviderAvailabilityReceipt`
- `InferenceHeadroomReceipt`
- `DeploymentResourceReceipt`

## Why These Belong In Lower WMs First

These surfaces matter before the Economic WM because they already change
runtime truth in lower WMs:

- Perception WM needs provider availability, batch capacity, and headroom to
  decide what evidence can be gathered honestly.
- Sim / Synth / Physics WM later needs the same family for backend/fidelity/
  materialization decisions.
- Embodiment / Actuation WM later needs them for action feasibility, latency,
  and on-device vs companion placement.

Only after those surfaces exist as typed lower-WM state should the Economic WM
elevate them into allocatable budget objects.

## Cross-WM Resource Surface Scope

Resource surfaces are not a Perception-only pattern. Each lower WM should
independently own its version of these typed surfaces:

- **Perception / Grounding WM** (Phase 2, currently live):
  - provider availability, batch capacity, inference headroom
  - sensor inventory, calibration assets
  - perception measurement quality
  - deployment/companion compute posture

- **Sim / Synth / Physics WM** (Phase 1.x, to be added):
  - backend availability (Isaac, Holosoma, LDM, GGDS)
  - GPU headroom, sim fidelity budget
  - materialization latency, branch capacity
  - sim-real consistency measurements

- **Embodiment / Actuation WM** (Phase 3, to be added):
  - action-feasibility latency, joint-limit posture
  - on-device vs companion compute placement
  - control-rate feasibility, safety watchdog headroom
  - battery/thermal impact on action feasibility

- **Economic WM** (Phase 5):
  - consumes lower-WM resource surfaces as allocatable budget objects
  - does not originate resource truth

The receipt families (`ProviderAvailabilityReceipt`, `InferenceHeadroomReceipt`,
`DeploymentResourceReceipt`) should also become cross-WM patterns. Each lower
WM emits its own receipts; the Economic WM later consumes them for allocation.

## Semantic Bridge Preconditions

The semantic bridge family should be shaped with these surfaces in mind now.

The Perception canonical semantic substrate later feeds:

- Sim / Synth / Physics bridge:
  - object preservation
  - synthetic-vs-real semantic alignment
  - branch evaluation
  - branch outcome semantics
- Embodiment bridge:
  - affordance
  - action relevance
  - bodily feasibility relevance
  - object-task relation
- Annotation / semantic-evidence bridge:
  - object-linked primitive/event crosswalk
  - failure / recovery labeling
- Economic bridge:
  - grounding quality
  - semantic contribution
  - action-relevant structural yield

These are bridge preconditions now, even if some consuming WMs are later.

## Resource Doctrine

On-board compute availability and inference headroom should first exist as
first-class provider/deployment-resource state within the lower WMs, with typed
receipts and honest availability posture. Only later should they be elevated
into allocatable Economic-WM objects.

The same rule applies to:

- battery
- thermal headroom
- bandwidth
- companion-compute posture

## Phase 2 Implementation Bias

Before GPU/provider bring-up, Phase 2 should already have:

- typed lower-WM provider/dataset/task/resource surfaces
- typed semantic bridge registry and receipts
- explicit provider truth and unavailable-mode posture
- explicit deployment-resource blockers rather than generic “not ready”

After GPU/provider bring-up, those surfaces should gain:

- real provider execution receipts
- real inference headroom measurements
- real calibration and runtime traces
- benchmark-gated promotion signals

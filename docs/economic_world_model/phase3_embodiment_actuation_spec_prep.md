# Phase 3 Embodiment / Actuation Spec Prep — 2026-05-19

## Entry posture

Phase 3 can begin as a **spec and canonical-state preparation lane** once the
Phase 1.x closure assessment is accepted. The recommended starting point is not
provider bring-up or hardware execution; it is the same pattern that worked for
Phase 2 and Phase 1.x:

1. typed canonical state contracts
2. shadow compiler from existing artifacts
3. downstream consumers
4. receipt emission
5. bounded learned seams behind promotion posture
6. provider/runtime contracts

Phase 3 should start from `docs/actuation_embodiment_world_model.md`, not from a
GR00T/Isaac ontology.

## Current substrate available before Phase 3

| Existing substrate | How Phase 3 should use it |
|--------------------|---------------------------|
| `src/embodiment/core.py` advisory artifacts | Use as an input/source-of-truth candidate for early shadow compilation, not as the final canonical WM schema |
| `src/embodiment/registry.py` capability registry | Promote into explicit capability and embodiment-profile state contracts |
| `src/runtime/action_adapter_v2.py` | Use as the runtime-facing action adapter contract surface |
| `src/runtime/observation_adapter_v2.py` | Use as the deploy observation/proprioception/timing contract surface |
| `src/world_model/perception_grounding/embodiment_shadow_consumer.py` | Use as the first Perception→Embodiment bridge input |
| `src/world_model/sim_synth_physics/adapters/embodiment_inputs.py` | Use as the first Sim→Embodiment transfer input |
| Phase 1.x runtime scan and manifests | Use provider/runtime truth as external evidence, not as Embodiment-owned truth |
| Phase 2 provider/evidence receipts | Consume observation and grounding confidence without collapsing Perception truth into Embodiment truth |

## Proposed Phase 3.1 contracts

Create a new additive package, likely:

```text
src/world_model/embodiment_actuation/
  __init__.py
  state.py
  receipts.py
  compiler.py
  provider_contracts.py
  promotion.py
```

Initial canonical state surface:

- `EmbodimentActuationWorldState`
- `CapabilityState`
- `EmbodimentProfileState`
- `ActuatorConfigurationState`
- `JointStateVector`
- `ContactStateVector`
- `SafetyEnvelopeState`
- `ActionSpaceState`
- `ObservationInterfaceState`
- `ContactAffordanceGraphState`
- `LocalDynamicsForecastState`
- `InverseRetargetTraceState`
- `ActionProposalBundleState`
- `EmbodimentDriftSummaryState`
- `EmbodimentCostVectorState`
- `CalibrationTargetState`

The six-subsystem mapping from `docs/actuation_embodiment_world_model.md` should
own these surfaces:

| Subsystem | State surfaces |
|-----------|----------------|
| 1. Capability / Embodiment State Surface | capability, embodiment profile, actuator config, action space, observation interface |
| 2. Contact / Affordance Graph Builder | contact state, contact-affordance graph, feasible interaction candidates |
| 3. Local Contact Dynamics Model | short-horizon local dynamics forecasts, contact transition confidence |
| 4. Inverse-Dynamics / Retargeting Lane | inverse-retarget traces, joint remap candidates, kinematic feasibility |
| 5. Joint Skill / Action Proposal Head | action proposal bundles, action chunks, skill candidates |
| 6. Drift / Calibration / Cost Evaluator | drift summary, calibration targets, cost vectors, safety/latency/energy receipts |

## Proposed receipt family

Phase 3 should emit receipts from the beginning. Minimum receipt contracts:

- `EmbodimentCompilationReceipt`
- `CapabilityProfileReceipt`
- `ActionSpaceValidationReceipt`
- `ObservationInterfaceReceipt`
- `ContactAffordanceReceipt`
- `LocalDynamicsReceipt`
- `InverseRetargetReceipt`
- `ActionProposalReceipt`
- `SafetyEnvelopeReceipt`
- `EmbodimentDriftReceipt`
- `CalibrationTargetReceipt`
- `EmbodimentCostReceipt`
- `SimEmbodimentTransferReceipt`

Receipts should explicitly record source refs, truth class, promotion posture,
missing evidence, degraded-mode flags, and downstream preconditions.

## Phase 3.1 acceptance gates

The first implementation tranche should be considered complete only when:

1. The canonical state dataclasses exist and are exported.
2. A shadow compiler can produce `EmbodimentActuationWorldState` from existing
   advisory embodiment artifacts plus optional Perception/Sim inputs.
3. Compilation emits `EmbodimentCompilationReceipt` and at least capability,
   action-space, observation-interface, and safety receipts.
4. No output is treated as runtime authority by default.
5. Existing Phase B baseline math, trust-net, `w_econ`, and lambda-controller
   paths remain untouched.
6. Tests prove backward compatibility for existing `src/embodiment/core.py`
   behavior.

Suggested focused verification:

```bash
python3 -m ruff check src/world_model/embodiment_actuation tests/test_embodiment_actuation_world_model.py
python3 -m compileall src/world_model/embodiment_actuation tests/test_embodiment_actuation_world_model.py -q
python3 -m pytest tests/test_embodiment_actuation_world_model.py -q
```

## Phase 3.2 shadow compiler

After the state contracts land, compile from existing local inputs:

- current `compute_embodiment(...)` artifacts
- `EmbodimentRegistry` entries
- `ActionAdapterV2` and `ObservationAdapterV2` references
- Perception embodiment shadow-consumer outputs
- Sim / Synth / Physics embodiment input adapter outputs
- optional runtime-layout scan refs for provider truth

The compiler should be permissive but explicit: missing provider/hardware data
is allowed, but missing fields must appear as structured `unavailable` or
`external_blocked` posture, not silent defaults.

## Phase 3.3 downstream consumers

Initial consumers should be shadow-only:

- Sim / Synth / Physics transfer boundary consumes action feasibility,
  retargeting readiness, and drift posture.
- Perception / Grounding consumes contact/affordance feedback as advisory
  downstream evidence.
- Runtime adapters consume action/observation contracts as validation context,
  not direct control authority.
- Economic consumers ingest cost/safety/drift receipts only after receipts exist.

## Phase 3.4 bounded learned seams

Do not start with model training. Reserve seams and promotion posture first:

| Seam | Scope | Initial posture |
|------|-------|-----------------|
| Local contact dynamics | forecast contact/state transitions | `disabled` or `auto` with heuristic fallback |
| Inverse retargeting | remap task/body actions into joint/control-space traces | `disabled` or `auto` |
| Action proposal / skill chunking | propose action chunks or skill candidates | `disabled` until receipts and benchmarks exist |
| Drift / calibration evaluator | estimate calibration decay and transfer mismatch | `auto` only after evidence exists |

Every seam must emit receipts and have a demotion path. No seam should bypass
safety envelopes or runtime adapter validation.

## Phase 3.5 provider/runtime contracts

Provider contracts should represent, but not require, external availability:

- Unitree G1 / R1-class morphology and joint profile refs
- Isaac / MuJoCo / Holosoma runtime refs
- action/control-rate constraints
- actuator latency and watchdog refs
- onboard/companion compute placement
- battery/thermal/energy surfaces
- safety envelope and emergency-stop posture

The current Phase 1.x assessment shows why these should remain typed external
refs until provider/hardware evidence exists.

## Non-goals for the first Phase 3 tranche

- No hardware deployment.
- No native GR00T import.
- No policy promotion.
- No rewrite of `src/embodiment/core.py`.
- No changes to frozen Phase B baseline math.
- No economic reward-path changes.
- No fabricated actuator latency, watchdog, battery, or thermal profiles.

## Recommended next implementation step

After owner/Claude acceptance of Phase 1.x closure, implement Phase 3.1 as a
small additive tranche:

1. add `src/world_model/embodiment_actuation/state.py` and `receipts.py`
2. add a shadow `compiler.py` that consumes existing advisory artifacts
3. add tests for missing-data posture, receipt emission, and no-runtime-authority
4. update roadmap/progress docs with exact verification evidence

## Implementation status — 2026-05-20

Phase 3.1 through 3.3 are now implemented as an additive shadow substrate:

- `state.py`: canonical `EmbodimentActuationWorldState` and all first-pass
  subsystem state surfaces
- `receipts.py`: compilation, capability, action-space, observation-interface,
  contact/affordance, local-dynamics, inverse-retarget, action-proposal,
  safety, drift, calibration, cost, and Sim↔Embodiment transfer receipts
- `provider_contracts.py`: Unitree G1, Holosoma, Isaac, generic provider, and
  runtime-resource contract surfaces that keep external evidence explicit
- `promotion.py`: bounded seam promotion posture for local dynamics, inverse
  retargeting, action proposal, and drift/calibration seams
- `compiler.py`: shadow compiler from existing advisory embodiment artifacts,
  registry/adapters, Perception shadow surfaces, provider contracts, and
  optional joint state
- `consumers.py`: first shadow consumers for Sim/Synth transfer, Perception
  feedback, Runtime adapter validation, and Economic receipt bundles
- `tests/test_embodiment_actuation_world_model.py`: focused regression coverage

Still deferred: provider execution, GPU-backed training, hardware calibration,
native GR00T loops, policy promotion, and any runtime authority.

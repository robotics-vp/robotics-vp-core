# GR00T-Inspired Functionality Status — 2026-05-19

## Doctrine boundary

GR00T / Isaac / VisualSim2Real are **pattern sources**, not this repo's ontology.
The native topology remains the multi-WM stack: Sim / Synth / Physics,
Perception / Grounding, Embodiment / Actuation, Transport, Runtime, Governance,
and Economic layers with typed state, receipts, promotion gates, and external
provider truth kept explicit.

The useful borrow is operational discipline: teacher/student lanes,
sim-to-real receipts, deploy-shaped observations, randomization/reset curricula,
checkpoint/export gates, and safety/eval promotion posture.

## End-state functionality

In the end state, the GR00T-inspired lane should look like this:

```text
Sim / Synth / Physics WM
  - privileged sim teacher traces
  - randomized/reset curricula
  - backend-mismatch + sim-real-gap receipts
        |
        v
Perception / Grounding WM
  - deploy-shaped observation bundles
  - camera/depth/temporal degradation surfaces
  - provider/evidence receipts
        |
        v
Embodiment / Actuation WM
  - action-space and joint-limit contracts
  - inverse-retarget traces
  - local dynamics forecasts
  - student/action-proposal bundles
  - drift, latency, safety, and cost receipts
        |
        v
Runtime / Evaluation / Economic Consumers
  - shadow execution first
  - benchmark/promotion gates
  - economic value attribution after evidence exists
```

Concrete end-state capabilities:

1. **Teacher/student sim-to-real lane**
   - Sim / Synth / Physics can emit privileged teacher traces from provider
     runtimes.
   - Embodiment / Actuation can train or evaluate deployable student/action
     proposal heads against degraded, deploy-shaped observation bundles.
   - The student never becomes native truth until promotion receipts and
     benchmark gates pass.

2. **Deploy-shaped observation discipline**
   - Camera bundles, proprioception, timing, delay, dropout, occlusion, and
     degraded-mode profiles are explicit inputs.
   - Perception owns observation evidence; Embodiment owns actuator/body-side
     feasibility and control consequences.

3. **Action-space and retargeting hygiene**
   - Joint maps, limits, action dimensions, control rates, torque/velocity
     envelopes, and safety envelopes are typed.
   - GR00T-style G1 action hygiene is borrowed as a discipline, not as a
     replacement action ontology.

4. **Randomization, reset, and curriculum receipts**
   - Provider runs carry explicit domain-randomization axes, reset conditions,
     task definitions, and branch/replay validity.
   - Failed or rejected branches remain useful as negative supervision and
     diagnostic receipts.

5. **Export and promotion gates**
   - ONNX/checkpoints are candidate artifacts, not proof.
   - Promotion requires manifest validation, real provider outcomes,
     benchmark density, safety posture, and economic downstream evidence.

6. **Transfer-boundary accountability**
   - Sim owns simulation assumptions and backend mismatch.
   - Embodiment owns remap, retarget, feasibility, realized drift, latency
     divergence, local recovery posture, and deployment-side degradation.

## What is wired now

| Surface | Current status |
|---------|----------------|
| Phase 2 Perception typed evidence | Wired as canonical Perception / Grounding state, provider contracts, semantic bridge receipts, deployment-resource surfaces, and bounded neural seams behind promotion posture |
| Phase 1.x runtime truth ladder | Wired for runtime packs, bindings, preflight scans, selected policy refs, receipt manifests, manifest validation, training admissibility, reject sidecars, and bounded reject heads |
| Holosoma deploy smoke | Wired as a local no-heavy-install ONNX action smoke: actor observation `[1, 100]` to finite action `[1, 29]`; native WM runtime routing remains gated |
| Isaac / Unitree local scan | Wired enough to identify local policy/assets/upstream roots and the real missing latency/watchdog profiles; native runtime execution remains external |
| Embodiment registry substrate | Wired as `CapabilityProfile`, `EmbodimentRegistryEntry`, and `EmbodimentRegistry` |
| Runtime adapter contracts | Wired as `ActionAdapterV2` and `ObservationAdapterV2` with schema, timing, translator, and embodiment references |
| Embodiment advisory artifacts | Wired through current `compute_embodiment(...)` outputs: profile, affordance graph, skill segments, cost, value attribution, drift, and calibration targets |
| Cross-WM shadow inputs | Wired through Perception's embodiment shadow consumer and Sim/Synth's embodiment input adapter |
| Embodiment / Actuation canonical WM substrate | Wired as `src/world_model/embodiment_actuation/`: canonical state contracts, receipts, provider contracts, promotion posture, shadow compiler, and first shadow consumers |
| GR00T-shaped native loop sockets | Structurally wired as teacher/student-adjacent state and receipt surfaces: deploy observation, action-space validation, inverse retarget trace, action proposal bundle, transfer receipt, and promotion gate posture |

## What is not wired yet

- No native GR00T training loop is imported or owned.
- No privileged-teacher to deploy-student training loop has been run.
- No full native Holosoma simulated episode has been promoted.
- No Isaac / Unitree runtime execution has been validated.
- No G1 hardware calibration, latency, watchdog, or drift evidence exists.
- No policy checkpoint or ONNX artifact is benchmark-promoted.
- No Embodiment / Actuation output has runtime authority; the compiler and
  consumers are shadow/advisory only.

## Borrow / adapt / ignore

| Upstream pattern | Treatment here |
|------------------|----------------|
| Teacher/student sim-to-real training | Borrow as typed run/eval/training discipline across Sim and Embodiment WMs |
| Isaac/GR00T action-space hygiene | Adapt into Embodiment action-space, joint-limit, and safety-envelope contracts |
| Randomization/reset curricula | Borrow as Sim-owned provider receipts and training-row metadata |
| Export gates for deployable artifacts | Borrow as candidate-artifact plus promotion-gate discipline |
| GR00T task primitives or policy architecture | Ignore as ontology; only borrow if a bounded seam earns promotion |
| Isaac Lab as sovereign runtime truth | Ignore; it is one provider family under repo-native contracts |
| PPO/DAgger/ResNet/ONNX as required primitives | Ignore; these are optional implementations or artifact formats, not architectural truth |

# UE5 / Unreal Provider-Family Posture

## Purpose

This note defines how UE5 / Unreal should be understood inside Ixion's
existing multi-WM topology.

The short version:

- UE5 is highly relevant to Ixion.
- UE5 is a major provider-family for the Sim / Synth / Physics WM.
- UE5 is **not** the ontology of the stack.
- UE5 does **not** become the owner of scene truth, embodiment truth,
  controller truth, economic truth, or governance truth.

This note is deliberately constitutional. It reserves the right typed surfaces
for later implementation without letting an engine article redefine the repo.

## Why UE5 Matters to Ixion

UE5 matters because it is unusually strong in the provider families that are
already strategically important to Ixion:

- photorealistic rendering for camera-first sim-real gap closure
- large-scene materialization for industrial and warehouse environments
- digital-twin ingestion from real deployment surfaces
- synthetic data generation with render-time labels
- programmable domain randomization and PCG-based layout variation
- broad multi-modal sensor simulation
- middleware-connected real-time simulation
- later hybrid full-loop autonomy testing, headless cloud generation, and
  industrial twin validation

That is real leverage for the Sim / Synth / Physics WM. It is not a license to
collapse the stack into "UE + RL" or "UE as world truth."

## Governing Rules

### Provider-not-truth-owner rule

UE5 is a provider family. It may generate scenes, render outputs, sensor
streams, digital twins, and runtime bridges, but it does not own:

- canonical scene truth
- branch truth
- transfer truth
- deployment truth
- embodiment truth
- economic valuation truth

Those remain WM-owned typed surfaces and receipts.

### Hybrid backend rule

UE5 should often be paired with other backend families rather than treated as
the sole physics authority.

Expected posture:

- UE5 for world realism, photoreal rendering, sensors, digital twins, and
  large-scene variation
- MuJoCo / Bullet / Newton / AGX-like lanes where other contact, control, or
  fidelity regimes are more appropriate

Hybrid is normal, not a fallback embarrassment.

### Anti-overfit rule

Do not let UE5 define the topology.

Ixion keeps:

- the existing WM ordering
- the Sim / Synth / Physics WM as the provider-governing owner
- Perception / Grounding as the owner of grounded scene/state truth
- Embodiment / Actuation as the owner of body-local control truth
- Economic WM as the later consumer of receipts rather than the owner of
  simulation mechanics

### Randomization-policy rule

UE5 can execute randomization, PCG, and asset variation, but the policy for
what gets randomized, why, and under which calibration regime remains
WM-owned.

The engine executes. The WM decides.

## Capability Placement by Family

### 1. Photorealistic rendering

Relevant UE5 capabilities:

- Nanite
- Lumen
- hardware ray tracing

Why it matters:

- stronger camera-first sim-real alignment
- better visual realism for branch screening
- better perception-training corpora and edge-case generation

Primary topology:

- Sim / Synth / Physics WM
  - Scene / Asset / Materialization Layer
  - Render / Diffusion / Materialization Lane
  - Sim-Real Gap / Realism Evaluator

Typed surfaces to reserve:

- `UESceneMaterializationState`
- `UEPhotorealRenderReceipt`
- `UESimRealVisualAlignmentReceipt`
- `UEAssetContentContract`

What to defer:

- full cloud-scale headless photoreal branch generation at volume
- benchmark-grade visual calibration against real site captures

### 2. Physics simulation

Relevant UE5 capabilities:

- Chaos Physics
- pairing with AGX-style high-fidelity physics

Why it matters:

- useful physics in visually realistic environments
- industrial machine and terrain regimes where engineered physics matters
- integrated interactive simulation for autonomy testing

Primary topology:

- Sim / Synth / Physics WM
  - Backend / Runtime / Provider Surface
  - Fidelity / Randomization / Calibration Allocator
  - Sim-Real Gap / Realism Evaluator

Typed surfaces to reserve:

- `UEHybridBackendBindingState`
- `UEPhysicsExecutionContract`
- `UEPhysicsCalibrationReceipt`

Doctrine constraint:

UE physics is valuable, but UE should often sit in a hybrid backend posture
with MuJoCo / Bullet / Newton / AGX-like lanes rather than becoming "the one
physics truth."

### 3. Digital twins / reconstruction

Relevant UE5-adjacent capabilities:

- RealityScan
- photogrammetry
- SLAM point clouds
- LiDAR-supported reconstruction
- pose / registration export
- Colmap-compatible outputs
- Linux server, CLI, and remote-command workflows

Why it matters:

- deployment-matched regression environments
- real-site to twin to synthetic-branch loops
- faster warehouse / workcell / facility reconstruction

Primary topology:

- Sim / Synth / Physics WM
  - Scene / Asset / Materialization Layer
  - Task / Measurement / Episode Layer
  - Sim-Real Gap / Realism Evaluator

Typed surfaces to reserve:

- `UEDigitalTwinIngestReceipt`
- `UEDigitalTwinRegistrationContract`
- `UEDigitalTwinRegressionEnvironmentState`

What stays outside Unreal ownership:

- site-grounding truth
- branch admission
- transfer valuation

### 4. Synthetic data

Relevant UE5 capabilities:

- render-time labels
- segmentation / depth / pose / motion / material truth

Why it matters:

- perception training corpora
- branch evaluation
- annotation/export bridges
- simulation-side evidence production

Primary topology:

- Sim / Synth / Physics WM
  - Task / Measurement / Episode Layer
  - Branch Planner / Branch Evaluator
  - Render / Diffusion / Materialization Lane

Downstream consumers:

- Perception / Grounding WM training/eval inputs
- later Economic WM yield and cost accounting

Typed surfaces to reserve:

- `UESyntheticDataReceipt`
- `UEAnnotationExportReceipt`
- `UESensorSimulationReceipt`

Doctrine constraint:

UE may produce synthetic labels and synthetic corpora, but synthetic-data
economics and valuation remain later Economic-WM concerns, not engine-owned
truth.

### 5. Domain randomization / PCG / Fab / Quixel

Relevant UE5 capabilities:

- programmable materials / lighting / weather / geometry variation
- camera parameter variation
- PCG layout generation
- asset ingestion from Fab / Quixel

Why it matters:

- rare-event and clutter generation
- occlusion-heavy scenes
- warehouse / office / workcell layout variation
- broader transfer coverage without a separate ontology

Primary topology:

- Sim / Synth / Physics WM
  - Fidelity / Randomization / Calibration Allocator
  - Branch Planner / Branch Evaluator
  - Scene / Asset / Materialization Layer

Typed surfaces to reserve:

- `UERandomizationPolicyState`
- `UEPCGLayoutGenerationReceipt`
- `UEAssetContentContract`

Doctrine constraint:

Randomization policy remains WM-owned. UE executes the variation plan; it does
not decide what coverage policy is correct.

### 6. Sensor simulation

Relevant UE5 capabilities:

- RGB / HDR
- depth
- LiDAR
- radar / SAR
- thermal / FLIR
- IMU / GPS
- underwater sonar as proof of breadth
- noise / distortion / latency / synchronization modeling

Why it matters:

- stronger perception training and calibration
- sensor-fusion preparation
- deployment-side timing and degraded-mode preparation
- humanoid/mobile readiness later on

Primary topology:

- Sim / Synth / Physics WM for provider/runtime ownership
- Perception / Grounding WM for consumption of simulated sensor corpora
- Embodiment / Actuation WM for sensor-timing and execution-side consequences

Typed surfaces to reserve:

- `UESensorSimulationContract`
- `UESensorSimulationReceipt`
- `UESensorTimingProfile`
- `UESensorSynchronizationReceipt`

### 7. ROS / ROS2 / gRPC / middleware integration

Relevant UE5 capabilities:

- ROSIntegration
- rclUE
- UE ROS2 sensor plugins
- CARLA-style ecosystems
- Tempo-style gRPC / Protobuf posture

Why it matters:

- external-runtime bridging
- companion-compute and middleware-connected simulation
- later hardware-facing integration and HIL testing

Primary topology:

- Sim / Synth / Physics WM
  - Backend / Runtime / Provider Surface
- later Phase 4A / 4B / 4E deployment-enabler phases

Typed surfaces to reserve:

- `UEMiddlewareBridgeContract`
- `UERuntimeTransportProfile`
- `UESensorStreamBridgeReceipt`

Doctrine constraint:

This is strategically important, but it should not displace current lower-WM
structural priorities before the relevant phases.

### 8. Full simulation loop / hybrid architectures / cloud scale

Relevant UE5 capabilities:

- RL-connected or autonomy-connected loops
- hybrid render + MuJoCo / Bullet contact posture
- Linux containers / Docker / headless GPU rendering
- cloud-scale synthetic generation and validation

Why it matters:

- branch generation at scale
- autonomy validation runs
- later transfer-evaluation and deployment-testing lanes
- recurring weekly GPU / Runpod execution

Primary topology:

- Sim / Synth / Physics WM
  - Branch Planner / Branch Evaluator
  - Render / Diffusion / Materialization Lane
  - Backend / Runtime / Provider Surface

Typed surfaces to reserve:

- `UEHeadlessExecutionContract`
- `UEHybridBackendBindingState`
- `UEAutonomyValidationReceipt`

What to defer:

- recurring cloud/headless generation until provider/runtime lanes and receipts
  are structurally real

### 9. Industry 4.0 / digital twin operations

Relevant UE5 capabilities:

- warehouse / factory / AGV / workcell / facility twins
- operator training
- HIL
- teleop
- XR co-presence

Why it matters:

- future industrial digital-twin deployment surfaces
- operator / teleop / recovery fallback preparation
- future industrial cybernetics posture

Primary topology:

- Sim / Synth / Physics WM for twin generation and regression environments
- Embodiment / Actuation WM for operator / teleop / recovery implications
- later Economic WM for cost/yield/value consumption

Typed surfaces to reserve:

- `UEIndustrialTwinScenarioState`
- `UEOperatorTrainingReceipt`
- `UETeleopFallbackReadinessReceipt`

## Reserve Now vs Defer Later

### Reserve now

- provider-family placement in the Sim / Synth / Physics WM
- typed contract and receipt families
- hybrid backend doctrine
- anti-overfit and provider-not-truth-owner rules
- Sim↔Embodiment boundary implications for sensor timing, latency, and
  deployment mismatch
- staged roadmap placement

### Defer later

- real UE runtime bring-up
- plugin/runtime packaging
- headless cloud generation loops
- ROS2 / gRPC bridge execution
- digital-twin ingest pipelines at scale
- HIL / teleop / industrial twin workflows

These are execution and GPU/runtime problems, not current topology problems.

## What Unreal Should Never Become in This Repo

UE5 should never become:

- the master ontology of the stack
- the canonical owner of scene truth
- the canonical owner of embodiment or controller truth
- the canonical owner of economic valuation or governance
- a justification for reordering the WMs
- a shortcut to "mother-latent" collapse through engine-native blobs
- a blanket replacement for other backend families where hybrid posture is more
  honest

## Bottom Line

UE5 is a major provider-family for the Sim / Synth / Physics WM.

That is a strong claim, but it is still a bounded claim:

- yes to UE5 for realism, sensors, digital twins, synthetic data, and
  middleware-connected simulation
- yes to explicit typed surfaces that make those capabilities legible
- no to Unreal becoming the ontology, body owner, controller owner, or
  economic owner of the stack

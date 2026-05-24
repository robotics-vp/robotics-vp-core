# Humanoid Target Readiness

## Purpose

This document turns the repo's long-term G1/R1-class hardware target into an explicit readiness checklist.

It is not a commitment that the repo is close to humanoid deployment today.

It is a planning and audit artifact that answers:

- what must be true before the stack can honestly claim Unitree G1/R1-class readiness
- which current modules are useful substrate versus insufficient proxy
- what benchmark classes and contracts are missing
- which model-capacity and environment assumptions must be revisited

It also assumes the stronger subsystem rule from the multi-WM plan:

- each lower WM and each deployment-enabler phase should be pushed until the main remaining blockers are honest G1/R1-class data, GPU, calibration, middleware, and benchmark prerequisites
- missing neural scaffolds, missing package/runtime lanes, or missing production-loop integration should not remain as the reason a subsystem is "not ready"

## Target Assumption

The target considered here is:

- Unitree G1/R1-class embodied systems
- bipedal whole-body control as the primary design center
- 21+ DoF whole-body control
- locomotion plus manipulation
- dexterous or semi-dexterous hand use
- human-proximate safety expectations
- real onboard/companion compute constraints

This target is materially different from:

- fixed-base tabletop manipulation
- workcell-only operation
- gripper-only action spaces
- purely offline or episode-timescale control

## Embodiment Posture Hierarchy

The repo's humanoid target posture is now explicit:

| Posture | Architectural role | Promotion meaning | What it must not become |
| --- | --- | --- | --- |
| `bipedal_whole_body` | Primary standard and default design center for Unitree G1/R1-class readiness | The stack can reason over floating-base balance, support/contact, gait, loco-manipulation, bimanual/dexterous action, whole-body safety, and real compute/battery limits | A decorative future target that leaves fixed-base assumptions as the real default |
| `stable_base_mobile_manipulator` | Safety fallback / degraded-mode posture for humanoid operation when bipedal authority is unsafe, unavailable, or not yet promoted | The stack can preserve task progress or recover safely using a stable-base/mobile-manipulation envelope with explicit fallback receipts | A replacement for bipedal whole-body readiness, or a quiet way to promote tabletop manipulation as humanoid readiness |
| `fixed_base_tabletop` | Legacy curriculum, smoke-test, replay, and manipulation-skill island | Useful for pretraining, regression, semantic/economic plumbing, and narrow manipulation gates | Final embodiment-readiness evidence for G1/R1-class deployment |

Every lower WM, environment, sim adapter, training row, and benchmark should be able to state which posture it serves. If that posture is not declared, the artifact should be treated as `fixed_base_tabletop` or `unknown`, never as bipedal evidence by default.

The stable-base mobile-manipulator posture is a safety fallback and bring-up/degraded-mode lane. It is valuable because it provides a conservative control envelope for recovery, operator handoff, and partial task continuity, but it cannot close humanoid readiness gates on its own.

## Program Window Assumption

Assume:

- the first serious multi-WM training runs start on September 1, 2026
- the current multi-WM architecture should have its plumbing laid by August 31, 2026
- the next major external milestone is a Unitree G1 purchase window in July 2027
- the strict program target is sustainably autonomous G1 operation by September 30, 2027, with the control loop running, collecting data, and improving without recurring architecture surgery

That implies a stricter interpretation of readiness:

- before September 1, 2026, the missing work should be mainly structural plumbing, canonical contracts, runtime-package seams, receipt emission, and provider truth
- after September 1, 2026, the missing work should increasingly be training data, GPU time, calibration truth, benchmark evidence, Unitree assets, and whole-body integration
- by July 2027, it is acceptable to still be blocked on real hardware, calibration, or benchmark evidence
- by July 2027, it is not acceptable to still be blocked on missing lower-WM canonical state, missing replay/training exports, or missing runtime/provider contract plumbing that should have been laid in 2026
- by September 30, 2027, it is acceptable to still be improving benchmarks and capacity, but it is not acceptable to still be blocked on missing autonomous replay capture, missing recovery/teleop trace discipline, missing sensor/comms/control-loop separation, or missing on-robot improvement plumbing
- a lower WM or deployment layer does not count as ready if it only logs or summarizes; for the humanoid target, it must be driving all relevant downstream modules in the future hardware-ready loop, not merely one consumer or one demo path

## Critical Path

For a July 2027 Unitree G1 step and a September 2027 sustainable-autonomy target, the expected program shape is:

1. By August 31, 2026: lower-WM and economic-WM plumbing is structurally real.
2. From September 1, 2026 through December 31, 2026: run a weekly A100 program, sub-module by sub-module by WM, with loop runs, then training, then fine-tuning where the receipts justify it.
3. From January 1, 2027 through March 31, 2027: accumulate benchmark and calibration evidence for perception, sim/backend truth, embodiment contracts, and promotion posture.
4. From April 1, 2027 through June 30, 2027: harden Unitree-facing adapters, safety-adjacent middleware, whole-body replay/telemetry, and hardware integration discipline.
5. In July 2027: purchase and integration should expose hardware and calibration limits, not reveal that the architecture was still missing canonical subsystem plumbing.
6. From July 1, 2027 through August 31, 2027: convert first hardware bring-up into a stable recurring control-loop program with replay capture, safety/recovery traces, degraded-mode handling, and repeatable data export.
7. By September 30, 2027: the G1 control loop should be running sustainably enough to:
   - operate repeatedly without manual architectural intervention
   - collect replay, telemetry, and governance receipts continuously
   - feed those artifacts into recurring training/fine-tuning cycles
   - produce bounded ongoing improvement from real robot data rather than one-off bring-up demos
8. After that autonomy target is reached, transition into a production-loop runtime discipline:
   - weekly GPU / Runpod execution
   - external dataset aggregation interleaved with robot-origin loop runs
   - recurring training, fine-tuning, benchmarking, and redeployment
   - backlog exhaustion until important external/provider/training lanes are no longer sitting idle
   - then a stronger focus on latency and inference efficiency

The intended order of attack in that weekly A100 program is:

- sim / synth / physics sub-modules first
- perception / grounding sub-modules second
- embodiment / actuation sub-modules third
- economic-WM consolidation after the lower-WM outputs are producing real receipts
- local meta-node neuralization and later meta-node superposition / control only after the lower-WM and economic-WM surfaces are stable enough to justify higher-layer compute

That sequencing is important for a G1 target because the expensive weekly compute should first make the lower embodied/perceptual/sim surfaces real before asking the higher economic/control layers to optimize over them, and because a September 2027 autonomy target depends more on robust lower-loop recurrence than on one impressive purchase-window demo.

## Current Status

Current status is:

- **not humanoid-ready**

That is not a criticism of the current stack. It is an honest statement about scope.

As of 2026-05-24, the local structural readiness surface is stronger than the
earlier docs-only posture:

- Phase 3.5 typed capacity/schema/env/benchmark artifacts exist under
  `src/world_model/humanoid_readiness/phase35.py` and
  `artifacts/economic_world_model/phase35_humanoid_capacity_env_refit/`;
- Phase 3.5 now also has a canonical bipedal chassis scaffold under
  `src/world_model/embodiment_actuation/bipedal_chassis.py` and
  `artifacts/economic_world_model/phase35_bipedal_chassis_scaffold/`, with a
  29-DoF local chassis profile, limb frame tree, joint-limit envelopes,
  whole-body observation/action schemas, bipedal support states, and balance
  receipts;
- Phase 3.5 also has a no-GPU/no-hardware bipedal readiness audit under
  `src/world_model/embodiment_actuation/bipedal_readiness.py` and
  `artifacts/economic_world_model/phase35_bipedal_readiness_audit/`, with robot
  asset intake contracts, parse receipts, kinematic consistency reports,
  joint-vector validation receipts, balance geometry reports, and whole-body
  replay row slots;
- Phase 4 local non-hardware control-loop, companion-compute/comms, and
  operator/recovery contract artifacts exist under
  `src/world_model/humanoid_readiness/phase4.py` and
  `artifacts/economic_world_model/phase4_deployment_enabler_sweep/`;
- Phase 4 now also has a dry-run downstream-controller scaffold under
  `src/world_model/humanoid_readiness/downstream_controller.py` and
  `artifacts/economic_world_model/phase4_downstream_controller_scaffold/`,
  with Unitree ROS2 / SDK2-shaped bridge targets, G1Pilot-style fallback bridge
  targets, controller modes, command frames, safety receipts, dispatch-denied
  invocations, and replay-ready controller receipts;
- Phase 4 now has a Unitree/G1 bring-up readiness pack under
  `src/world_model/humanoid_readiness/unitree_bringup_readiness.py` and
  `artifacts/economic_world_model/phase4_unitree_bringup_readiness/`, with
  dependency inventory receipts, G1Pilot/fallback review receipts, robot asset
  joint-conformance receipts, stream and command contracts, a local-only timing
  probe, physical safety preflight receipts, operator e-stop/recovery runbooks,
  and a sim/hardware evidence ledger;
- Phase 4 now also has executable local Unitree/G1 harnesses under
  `src/world_model/humanoid_readiness/unitree_local_harness.py` and
  `artifacts/economic_world_model/phase4_unitree_local_harnesses/`, with
  synthetic low-state / IMU / wireless-e-stop / contact traces, JSONL replay
  receipts, mock receivers, stale-data validators, Unitree ROS2 message parses,
  no-publish command-shape receipts, mock timing/watchdog receipts,
  safety/recovery state transitions, and Unitree ROS2 / MuJoCo / G1Pilot
  preflight receipts;
- Phase 4 now has a runtime-evidence bridge under
  `src/world_model/humanoid_readiness/unitree_runtime_bridge.py` and
  `artifacts/economic_world_model/phase4_unitree_runtime_evidence_bridge/`,
  with ROS2/colcon readiness receipts, a guarded Unitree MuJoCo no-policy
  headless step receipt, JSONL/rosbag2/MCAP trace adapter receipts, expanded
  safety-envelope receipts, and scripted operator-recovery drill receipts;
- Phase 6.5 local meta-node state/receipt/target/robustness/gate artifacts
  exist under `src/world_model/humanoid_readiness/phase65.py` and
  `artifacts/economic_world_model/phase65_meta_node_neuralization/`.

These artifacts close local scaffold gaps only. The bipedal chassis scaffold and
readiness audit are still not a hardware-calibrated body model: their joint
envelopes are local planning envelopes, their asset parser is contract-level,
and their balance/replay receipts are schema/evidence slots. They do not change the
not-humanoid-ready deployment status because Unitree assets/runtime,
calibration, live streams, control interfaces, timing/jitter traces,
hardware or honest sim evidence, trained weights, and promotion-grade
benchmarks are still missing. The bring-up readiness pack narrows that list by
turning local dependency discovery, public G1 asset parsing, dry-run command
conformance, timing-probe slots, safety preflight slots, and operator recovery
runbooks into receipts; it does not remove the need for runtime build
verification, live DDS/SDK streams, hardware calibration, honest sim launch
evidence, or on-robot safety drills. The local harness pack goes one level
deeper for preflight by executing synthetic trace, command-shape,
timing/watchdog, and safety/recovery checks, but it is still not live stream,
command echo, physical calibration, or robot evidence. The runtime-evidence
bridge now adds minimal local MuJoCo no-policy stepping evidence, but still does
not prove ROS2 bridge integration, policy-controlled sim, command echo,
physical calibration, teleop runtime, hardware, or deployment-grade readiness.

The repo already has strong substrate worth preserving:

- typed runtime contracts in `src/runtime/packets.py`
- event and decision traces in `src/runtime/event_spine.py`
- governance traces in `src/governance/trace.py`
- evidence and belief-state layers in `src/evidence/bus.py` and `src/evidence/belief_state.py`
- semantic world-model state in `src/world_model/semantic_world_model.py`
- embodiment scaffolding in `src/embodiment/core.py` and `src/embodiment/registry.py`
- observation/action normalization scaffolding in `src/runtime/observation_adapter_v2.py` and `src/runtime/action_adapter_v2.py`
- sensor-bundle and grounding substrate in `src/motor_backend/sensor_bundle.py`, `src/vision/scene_ir_tracker/io/scene_tracks_runner.py`, and `src/ingestion/x_humanoid_adapter.py`

What is still missing is not “more docs.” It is:

- whole-body embodiment state
- humanoid simulation environments
- real-time control separation
- physical safety substrate
- sensor-fusion and spatial-state ownership
- robot asset/calibration discipline
- companion-compute middleware
- humanoid-specific benchmark gates

## Environment And Simulation Layout Standard

The environment and simulation layout should follow the posture hierarchy rather than treating all robot envs as interchangeable. The target layout is:

```text
envs/
  tabletop_curriculum/          # fixed-base skill islands and regression tests
  stable_base_fallback/         # mobile/stable-base safety fallback and recovery posture
  bipedal_whole_body/           # primary G1/R1 humanoid-readiness env families

sim_backends/
  typed_adapter_contracts/      # Isaac / MuJoCo / Unitree / other backend truth
  robot_assets/                 # robot-description, joint-map, limits, calibration refs
  replay_exports/               # posture-tagged receipts and training rows
```

This is a conceptual layout standard, not a claim that those directories already exist. The key requirement is that every env/sim artifact emits a posture tag, backend truth, robot-asset refs, observation/action schema refs, and promotion posture.

### What Current Envs Are Good For

Current envs such as:

- `workcell`
- `dishwashing`
- `drawer_vase`

should currently be treated as:

- `fixed_base_tabletop` curriculum lanes
- manipulation skill islands
- replay and control-plane substrate testbeds
- early pretraining or curriculum domains
- semantic/economic/governance infrastructure test cases

They should **not** be treated as:

- bipedal whole-body humanoid proxies
- locomotion/manipulation proxies
- balance or recovery benchmarks
- mobile-navigation benchmarks
- final embodiment-readiness validation domains

### Required future env/sim families

The bipedal whole-body standard requires named future env/sim families for:

- `bipedal_whole_body_balance_reach`: balance-constrained reaching and manipulation
- `bipedal_whole_body_locomotion_manipulation`: walking while carrying, reaching, placing, opening, or tool-using
- `bipedal_whole_body_bimanual`: dual-arm coordination under whole-body constraints
- `bipedal_whole_body_dexterous_contact`: hand/contact-rich manipulation under balance and safety constraints
- `bipedal_whole_body_disturbance_recovery`: push, slip, stumble, contact disturbance, and failed-step recovery
- `stable_base_mobile_manipulator_fallback`: conservative mobile/stable-base recovery and degraded-mode task continuity
- `fixed_base_tabletop_curriculum`: restricted manipulation pretraining and regression only

Promotion gates must not let a `fixed_base_tabletop` or `stable_base_mobile_manipulator` success satisfy a `bipedal_whole_body` benchmark unless the benchmark explicitly names transfer evidence and remaining gaps.

## Readiness Checklist

Status key:

- `present`: substrate exists in meaningful form
- `partial`: useful scaffold exists but not enough for humanoid readiness
- `missing`: no honest readiness path yet

| Area | What must exist | Current repo anchors | Status | Main gap |
| --- | --- | --- | --- | --- |
| Canonical runtime substrate | packets, events, governance, replayable receipts | `src/runtime/packets.py`, `src/runtime/event_spine.py`, `src/governance/trace.py` | `present` | needs broader embodied deployment consumers later |
| Semantic state substrate | typed semantic state and meta-node context | `src/world_model/semantic_world_model.py` | `present` | still not a humanoid embodiment model |
| Embodiment normalization | capability profiles, action/observation schema refs | `src/embodiment/registry.py`, `src/runtime/observation_adapter_v2.py`, `src/runtime/action_adapter_v2.py` | `partial` | fixed-base assumptions still dominate |
| Whole-body embodiment state | torso, limbs, balance, contact, gait, dexterity | `src/world_model/embodiment_actuation/bipedal_chassis.py` | `partial` | local chassis/frame/schema/balance slots exist; no calibrated sim/hardware stream yet |
| Compute envelope / placement budgeting | onboard/companion compute headroom, reserve, placement class, QoS | `src/world_model/humanoid_readiness/phase35.py`, `src/world_model/humanoid_readiness/phase4.py` | `partial` | local planning contracts exist; no measured runtime telemetry yet |
| Battery / power resource state | state of charge, reserve, discharge ceiling, allocatable spend, thermal coupling | `src/world_model/humanoid_readiness/phase35.py`, `src/world_model/humanoid_readiness/phase4.py` | `partial` | local planning contracts exist; no real battery/thermal stream yet |
| Humanoid sim env integration | Unitree-class sim lane under typed backend contract | `src/envs/physics/isaac_backend.py`, `src/motor_backend/*` | `missing` | no real G1/R1 sim integration |
| Perception / grounding for humanoids | egocentric + depth + 3D grounding + body-aware scene state | `src/vision/scene_ir_tracker/io/scene_tracks_runner.py`, `src/vision/reconstruction/four_d_reconstruction.py` | `partial` | GPU/SAM3D and canonical perception WM still pending |
| Sensor fusion | IMU, proprio, camera, depth, force/torque fusion | `src/ingestion/x_humanoid_adapter.py`, `src/runtime/observation_adapter_v2.py` | `partial` | no real fusion stack yet |
| Real-time control split | reflex/servo loop separated from governance loop | `src/world_model/humanoid_readiness/phase4.py` | `partial` | contract surfaces exist; no measured control interface or jitter traces yet |
| Downstream controller primitive | bounded low-level command frame, safety receipt, dispatch receipt | `src/world_model/humanoid_readiness/downstream_controller.py` | `partial` | dry-run command frames exist; no ROS2/SDK2 publish, hardware, or sim dispatch yet |
| Physical safety layer | joint limits, self-collision, e-stop, fall protection | local `JointLimitEnvelope` planning receipts | `partial` | joint-limit envelopes are not hardware-calibrated safety limits |
| Spatial state / SLAM | localization, mapping, navigation state | none canonical yet | `missing` | current envs are mostly fixed-workcell |
| Companion compute / comms | onboard/offboard split, QoS, degraded-link handling | `src/world_model/humanoid_readiness/phase4.py` | `partial` | local middleware contracts exist; no ROS2/DDS/Unitree runtime evidence yet |
| Robot asset + calibration | URDF/Xacro/SRDF, extrinsics, calibration sidecars | `src/world_model/embodiment_actuation/bipedal_readiness.py` | `partial` | asset intake/parser receipts exist; no real calibrated transforms yet |
| Teleop / recovery fallback | operator override and recovery trace path | `src/world_model/humanoid_readiness/phase4.py` | `partial` | local operator/recovery contracts exist; no live recovery traces yet |
| Humanoid benchmark gates | benchmark taxonomy and promotion rules | workcell/grounding benchmark gates only | `missing` | no humanoid benchmark layer |
| Model-capacity audit | explicit sizing review by subsystem | `src/world_model/humanoid_readiness/phase35.py` | `partial` | local capacity bands exist; no trained capacity evidence yet |

## Benchmark Matrix

The repo eventually needs benchmark classes beyond current workcell/manipulation gates.

| Benchmark class | Why it matters for G1/R1 | Required environment or hardware lane | Required artifacts | Current status |
| --- | --- | --- | --- | --- |
| Balance stability | a humanoid cannot be “task-capable” if it cannot stay upright robustly | humanoid sim + later hardware | runtime packet, contact state, balance metrics, event spine | `missing` |
| Locomotion plus manipulation | walking and acting must compose | humanoid sim | whole-body action schema, task receipts, safety traces | `missing` |
| Bimanual coordination | many humanoid tasks are naturally two-handed | humanoid sim | dual-arm action contracts, contact receipts | `missing` |
| Dexterous hand task completion | gripper-only assumptions break here | hand-capable sim or hardware | hand state, contact traces, failure taxonomy | `missing` |
| Push / slip / stumble recovery | recovery is core to real embodied robustness | humanoid sim + later hardware | disturbance events, recovery actions, governance traces | `missing` |
| Self-collision / joint-limit compliance | physical safety baseline | sim + later hardware | low-level safety receipts, veto traces | `missing` |
| Human-proximate safety | robot must remain safe around people | sim with human model + later hardware | safety envelope refs, override traces | `missing` |
| Sensor-dropout robustness | real sensing is imperfect | sim and later hardware | degraded-sensing flags, recovery traces | `missing` |
| Companion-link degradation | onboard/offboard split must fail safely | hardware-in-loop or middleware emulation | comms QoS receipts, watchdog events | `missing` |
| Compute-pressure / placement degradation | inferential and perception load must degrade safely under on-device limits | middleware emulation, sim, or later hardware | compute-envelope refs, placement receipts, degraded-mode traces | `missing` |
| Battery / thermal degraded mode | long-horizon field behavior depends on resources | later hardware or hardware emulation | compute/battery telemetry refs, planning reactions | `missing` |
| Workcell manipulation continuity | still useful as lower-tier manipulation check | current workcell envs | current replay + benchmark artifacts | `partial` |

## Model-Capacity Review Targets

Not every model needs to be large. The question is where scale is structurally required.

### Modules likely to need materially more capacity

- future embodiment / actuation WM encoders
- compute-envelope / battery-forecast / placement models that must reason about real on-device resource pressure rather than abstract scalar budgets
- whole-body control-conditioned policy heads
- perception / grounding WM modules that fuse:
  - egocentric vision
  - depth
  - proprioception
  - body state
  - spatial state
- sim / synth / physics WM components that model:
  - contact-rich whole-body behavior
  - locomotion/manipulation transitions
  - disturbance recovery
- cross-WM transport bridges carrying richer body + scene topology

### Modules that can likely stay relatively compact

- economic WM allocation and governance layers if they operate over canonical lower-WM state
- many orchestration/meta-choice helpers
- some queueing and scheduling layers
- later meta-node superposition layers, if they govern rather than re-encode raw embodiment state

### Capacity review questions

Every lower-WM/submodule review should answer:

1. Is this model representing raw embodied complexity or consuming typed summaries?
2. Is the action space still implicitly gripper-scale?
3. Does this model need to represent contact, balance, locomotion, and dexterity directly?
4. Would increasing capacity here reduce real bottlenecks, or merely compensate for a missing lower-WM contract?
5. Does this module assume onboard, companion, or offline/GPU compute that will not actually be available on the target robot?

## Neural Scaffolding Consequences

The posture hierarchy changes the neural scaffolding target. Neural seams should be shaped around bipedal whole-body control first, with stable-base/mobile fallback as an explicit safety/degraded-mode classifier, not as the hidden default.

Minimum neural scaffold families for the Phase 3.5 / Phase 4 return:

| Scaffold family | Primary posture | Function | Local non-GPU work now | Future evidence required |
| --- | --- | --- | --- | --- |
| Whole-body state encoder | `bipedal_whole_body` | Encode floating base, limbs, hands, IMU, proprioception, contact, support phase, and body-relative scene state | Define input/output contract, topology, manifest, rows, and CPU shape checks | GPU training, sim/hardware trajectories, calibration receipts |
| Contact/support/balance predictor | `bipedal_whole_body` | Predict support polygon, slip/fall risk, contact feasibility, and balance margin | Define losses and receipt schema; emit synthetic/local placeholder rows without promotion | humanoid sim disturbance corpora and hardware validation |
| Loco-manipulation action proposal head | `bipedal_whole_body` | Propose action chunks that couple gait, posture, arms, hands, and task constraints | Define action schema, chunk horizons, policy-head manifest, denied-promotion gates | trained policy heads, benchmark pass, safety gates |
| Inverse-dynamics / retargeting lane | `bipedal_whole_body` | Convert demonstrations, teleop, sim, and video-derived traces into Unitree-native action/state rows | Define retargeting receipts, robot-asset refs, feasibility filters, row schemas | real assets, datasets, retargeting quality evals |
| Stable-base fallback selector | `stable_base_mobile_manipulator` | Detect when bipedal authority should degrade to stable-base/mobile-manipulator mode | Define classifier inputs, intervention receipts, veto/defer semantics | safety benchmark evidence and recovery traces |
| Latency/watchdog/resource predictor | all postures, but bipedal-primary | Forecast control-rate feasibility, stale-data risk, compute/battery/thermal pressure | Define resource-state rows and Phase 4A/4E contracts | measured middleware/hardware telemetry |

Training rows should carry at least:

- `embodiment_posture`: `bipedal_whole_body`, `stable_base_mobile_manipulator`, `fixed_base_tabletop`, or `unknown`
- `promotion_scope`: `curriculum`, `fallback`, `shadow`, `advisory`, `benchmark_candidate`, or `promoted`
- robot-description / joint-map / joint-limit / calibration refs
- support/contact/balance labels where available
- compute, battery, thermal, latency, and placement refs
- transfer annotations when a tabletop or fallback lane is used as pretraining rather than direct promotion evidence

The stable-base fallback selector may be learned, but it must remain below the bipedal whole-body target: it can veto, defer, recover, or request operator handoff; it cannot silently redefine the primary embodiment standard.

## Environment Refit Requirements

The future environment roadmap should include named lanes for:

- Unitree G1/R1 simulation integration
- locomotion plus manipulation
- balance-constrained reaching
- bimanual skill execution
- dexterous hand contact tasks
- mobile navigation plus task completion
- push / slip / stumble recovery
- degraded sensing and degraded comms scenarios

### Unitree sim integration requirements

The eventual Unitree sim lane should define:

- backend choice
- robot asset source
- observation contract deltas
- action contract deltas
- replay compatibility
- benchmark compatibility
- safety and calibration receipt surfaces

This should not be an ad hoc env fork. It should sit behind typed backend and schema contracts.

The concrete Phase 3.5 return artifact is
`docs/economic_world_model/phase35_humanoid_capacity_env_refit.md`. It records
the local capacity bands, onboard/companion/battery assumptions, humanoid
observation/action schema deltas, posture-tagged environment taxonomy, Unitree
sim integration target, benchmark taxonomy, canonical bipedal chassis scaffold,
and no-GPU/no-hardware readiness audit. It remains planning-only until sim
assets, runtime evidence, hardware evidence, training, and promotion-grade
benchmarks exist.

## Contract Deltas Required For Humanoid Targeting

Current contracts will need to grow to include:

- richer proprio vectors
- IMU state
- force/torque state
- whole-body kinematic state
- contact and support-phase state
- foot/contact event summaries
- latency and control-rate metadata
- safety envelope and watchdog refs
- robot-description version refs
- calibration refs
- spatial-state refs
- comms/degradation status refs
- battery/thermal/compute-pressure refs
- compute-envelope refs
- allocatable compute headroom
- placement-class refs
- battery reserve and discharge-budget refs
- thermal-headroom refs

## Robot Asset And Calibration Checklist

Before claiming serious humanoid-target readiness, the repo should have a canonical story for:

- URDF / Xacro / SRDF storage and versioning
- joint-name normalization
- actuator-index normalization
- sensor extrinsics and intrinsics
- hand/end-effector definition
- self-collision geometry
- calibration receipts
- controller-gain and firmware identity refs

Without this, runtime packets may be typed but still not anchored to a real robot identity.

## Companion Compute And Communication Checklist

Humanoid readiness requires explicit answers to:

- what runs on-robot versus on companion compute
- what inferential compute remains allocatable after servo, perception, and safety reservations
- what latency budget each layer assumes
- how ROS2 / DDS / Unitree SDK2 messages become canonical WM state
- what happens on packet loss or stale perception
- how watchdog events are emitted
- how degraded-link mode enters replay and governance traces
- how battery reserve and compute availability constrain placement and runtime policy

## Teleop And Recovery Checklist

The stack should eventually support a bounded operator path for:

- manual takeover
- recovery-mode activation
- calibration bring-up
- safety stop and resume
- post-event replay labeling

This should become part of the typed event/governance/replay substrate, not a manual side channel.

## Repo-Grounded Gap Map

Concrete current code truths:

- `src/embodiment/core.py` is valuable, but still advisory and not a humanoid embodiment WM
- `src/embodiment/registry.py` and the runtime adapters are useful contract substrate, but not whole-body semantics
- `src/envs/physics/isaac_backend.py` remains a stub, so “Isaac-ready” is not equivalent to humanoid-sim-ready
- `src/ingestion/x_humanoid_adapter.py` exists, which is useful, but it is not a full sensor-fusion or deployment middleware layer
- `src/vision/scene_ir_tracker/io/scene_tracks_runner.py` is a strong grounding seam, but not a complete perception WM
- `src/motor_backend/workcell_env_backend.py` and `src/motor_backend/sensor_bundle.py` provide good rollout/sensor substrate, but still in workcell-centric form
- current benchmark gates are largely grounding/workcell-centric rather than humanoid-centric

## Honest Promotion Rule

The repo should not claim G1/R1-class readiness until all of the following are true:

- a canonical embodiment / actuation WM exists
- a Unitree-class sim lane exists behind typed backend contracts
- sensor fusion, physical safety, and spatial state each have canonical ownership
- real-time servo vs governance split is explicit
- robot asset and calibration handling is canonical
- communication / degraded-link behavior is part of replay and governance truth
- humanoid benchmark gates exist and are passed at the appropriate level

The repo should not claim it is on-track for the September 2027 target until all of the following are also true:

- the G1-facing control loop can run repeatedly without architecture edits between runs
- replay, telemetry, calibration, safety, and degraded-mode traces are emitted from those runs as canonical artifacts
- the training/export path can consume robot-origin receipts on a recurring schedule
- bounded autonomous data collection and bounded autonomous improvement are both real loop properties rather than operator-only workflows

The repo should not claim it has entered the post-September-2027 production-loop runtime phase until:

- weekly GPU / Runpod operations are actually recurring rather than ad hoc
- the important external dataset, loop-run, trainer, fine-tune, and benchmark backlogs are being actively exhausted
- latency/inference work has become the next-order optimization problem rather than a distraction from missing training/runtime coverage

## Recommended Near-Term Use

This document should be used as:

- the checklist backing Phase 3.5 and later humanoid-target work
- the acceptance reference for future Unitree sim integration planning
- the benchmark taxonomy seed for later promotion gates
- the model-capacity audit reference for lower-WM sizing decisions

It should **not** be used to justify immediate humanoid deployment claims from the current stack.

# Humanoid Target Readiness

## Purpose

This document turns the repo's long-term G1/R1-class hardware target into an explicit readiness checklist.

It is not a commitment that the repo is close to humanoid deployment today.

It is a planning and audit artifact that answers:

- what must be true before the stack can honestly claim Unitree G1/R1-class readiness
- which current modules are useful substrate versus insufficient proxy
- what benchmark classes and contracts are missing
- which model-capacity and environment assumptions must be revisited

## Target Assumption

The target considered here is:

- Unitree G1/R1-class embodied systems
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

## Current Status

Current status is:

- **not humanoid-ready**

That is not a criticism of the current stack. It is an honest statement about scope.

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

## What Current Envs Are Good For

Current envs such as:

- `workcell`
- `dishwashing`
- `drawer_vase`

should currently be treated as:

- manipulation skill islands
- replay and control-plane substrate testbeds
- early pretraining or curriculum domains
- semantic/economic/governance infrastructure test cases

They should **not** be treated as:

- whole-body humanoid proxies
- locomotion/manipulation proxies
- balance or recovery benchmarks
- mobile-navigation benchmarks
- final embodiment-readiness validation domains

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
| Whole-body embodiment state | torso, limbs, balance, contact, gait, dexterity | none canonical yet | `missing` | no G1/R1-class body state model |
| Humanoid sim env integration | Unitree-class sim lane under typed backend contract | `src/envs/physics/isaac_backend.py`, `src/motor_backend/*` | `missing` | no real G1/R1 sim integration |
| Perception / grounding for humanoids | egocentric + depth + 3D grounding + body-aware scene state | `src/vision/scene_ir_tracker/io/scene_tracks_runner.py`, `src/vision/reconstruction/four_d_reconstruction.py` | `partial` | GPU/SAM3D and canonical perception WM still pending |
| Sensor fusion | IMU, proprio, camera, depth, force/torque fusion | `src/ingestion/x_humanoid_adapter.py`, `src/runtime/observation_adapter_v2.py` | `partial` | no real fusion stack yet |
| Real-time control split | reflex/servo loop separated from governance loop | none canonical yet | `missing` | no 200-1000 Hz layer split |
| Physical safety layer | joint limits, self-collision, e-stop, fall protection | governance/econ safety only today | `missing` | safety is not yet physical-control-grade |
| Spatial state / SLAM | localization, mapping, navigation state | none canonical yet | `missing` | current envs are mostly fixed-workcell |
| Companion compute / comms | onboard/offboard split, QoS, degraded-link handling | none canonical yet | `missing` | no ROS2/DDS/Unitree middleware contract |
| Robot asset + calibration | URDF/Xacro/SRDF, extrinsics, calibration sidecars | none canonical yet | `missing` | robot identity/calibration not managed canonically |
| Teleop / recovery fallback | operator override and recovery trace path | none canonical yet | `missing` | no bounded human-recovery lane |
| Humanoid benchmark gates | benchmark taxonomy and promotion rules | workcell/grounding benchmark gates only | `missing` | no humanoid benchmark layer |
| Model-capacity audit | explicit sizing review by subsystem | none yet | `missing` | no formal capacity audit for 21+ DoF target |

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
| Battery / thermal degraded mode | long-horizon field behavior depends on resources | later hardware or hardware emulation | compute/battery telemetry refs, planning reactions | `missing` |
| Workcell manipulation continuity | still useful as lower-tier manipulation check | current workcell envs | current replay + benchmark artifacts | `partial` |

## Model-Capacity Review Targets

Not every model needs to be large. The question is where scale is structurally required.

### Modules likely to need materially more capacity

- future embodiment / actuation WM encoders
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
- what latency budget each layer assumes
- how ROS2 / DDS / Unitree SDK2 messages become canonical WM state
- what happens on packet loss or stale perception
- how watchdog events are emitted
- how degraded-link mode enters replay and governance traces

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

## Recommended Near-Term Use

This document should be used as:

- the checklist backing Phase 3.5 and later humanoid-target work
- the acceptance reference for future Unitree sim integration planning
- the benchmark taxonomy seed for later promotion gates
- the model-capacity audit reference for lower-WM sizing decisions

It should **not** be used to justify immediate humanoid deployment claims from the current stack.

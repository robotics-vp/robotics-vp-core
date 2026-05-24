# Phase 3.5 Humanoid Capacity and Environment Refit

Date: 2026-05-23

## Purpose

This pass returns from local Phase-6 transport closure to the humanoid target
refit required before later deployment and meta-node work.

It applies the current embodiment posture doctrine:

- `bipedal_whole_body` is the primary Unitree G1/R1-class readiness standard;
- `stable_base_mobile_manipulator` is a safety fallback / degraded-mode lane;
- `fixed_base_tabletop` is curriculum, regression, and manipulation-skill-island
  evidence only.

This is a local planning and contract refit. It is not a claim that Unitree sim,
hardware, provider runtime, or whole-body training has run.

## Capacity Bands

These are planning bands for architecture and schema work, not vendor spec
claims.

| Band | Placement | Intended work | Must not carry |
| --- | --- | --- | --- |
| `onboard_reflex_reserve` | robot onboard | servo/reflex loops, watchdogs, physical safety checks, emergency fallback triggers | large perception, transport training, economic planning, provider calls |
| `onboard_low_rate_state` | robot onboard when available | compact proprio/IMU/contact state encoding, health/state summaries, degraded-mode flags | bipedal promotion claims without measured timing |
| `companion_realtime_assist` | companion compute | perception fusion, local mapping, whole-body proposal scoring, retargeting prechecks, transport proposal evaluation | hard servo authority unless Phase 4A proves timing and watchdog behavior |
| `companion_heavy_inference` | companion compute | larger segmentation/grounding, sim preview, transport critics, benchmark capture support | unbounded latency paths in the control loop |
| `offline_gpu_training` | scheduled GPU/provider plane | bridge/receiver training, perception/sim/embodiment model training, benchmark sweeps | live policy control or on-robot authority |

The central rule is reserve-first: bipedal safety and reflex capacity must be
reserved before perception, transport, economics, or provider work is treated as
executable.

## Onboard, Companion, and Battery Assumptions

Local contracts should assume:

- on-robot compute is scarce and must preserve servo/reflex and safety reserves;
- companion compute can help with perception, mapping, transport evaluation, and
  proposal scoring, but it is not automatically safe for hard real-time control;
- battery state is canonical deployment state, not a cosmetic telemetry field;
- thermal headroom, stale-data risk, communication quality, and compute
  placement must be replayable receipt fields;
- degraded-mode behavior should be explicit rather than hidden in controller
  fallbacks.

Minimum resource fields for Phase 3.5/4 contracts:

- `compute_placement`: `onboard`, `companion`, `cloud`, or `offline_gpu`;
- `control_rate_class`: `servo_reflex`, `whole_body_fast`, `wm_slow`,
  `offline`;
- `battery_reserve_class`: `nominal`, `reserve`, `critical`, or `unknown`;
- `thermal_headroom_class`: `nominal`, `constrained`, `critical`, or `unknown`;
- `comms_qos_class`: `fresh`, `stale`, `degraded`, `lost`, or `unknown`;
- `degraded_mode_allowed`: boolean plus reason receipt.

## Observation Schema Deltas

The bipedal primary target requires observation surfaces beyond fixed-base
manipulation.

| Surface | Required fields | Primary posture | Notes |
| --- | --- | --- | --- |
| Whole-body proprioception | floating-base pose/velocity, joint position/velocity/torque/temp, actuator mode | `bipedal_whole_body` | Must preserve robot asset and joint-map refs |
| IMU and support state | IMU orientation/rates, support foot/contact phase, slip estimate, balance margin | `bipedal_whole_body` | Cannot be inferred from tabletop task state |
| Contact and force state | foot/hand contact flags, force/torque estimates, contact normals, self-collision proximity | `bipedal_whole_body` | Needed for balance and manipulation coupling |
| Egocentric perception | camera/depth refs, calibration refs, timestamp alignment, body-relative scene state | `bipedal_whole_body` | Provider outputs remain sidecar/advisory until proven |
| Resource and timing | compute placement, latency, stale-data age, battery/thermal posture, comms QoS | all postures | Feeds Phase 4A/4E and Economic WM receipts |

## Action Schema Deltas

| Action family | Required contract delta | Promotion posture |
| --- | --- | --- |
| Whole-body action chunk | coupled base/torso/arm/hand/foot action horizon with support-phase constraints | `bipedal_whole_body` only |
| Balance-preserving reach | reach/action proposal plus balance margin and fallback envelope | `bipedal_whole_body` primary |
| Bimanual/dexterous manipulation | dual-arm/hand targets, contact plan, force limits, tool state | `bipedal_whole_body` primary |
| Stable-base fallback action | conservative mobile/stable-base manipulation envelope and recovery mode | fallback/degraded only |
| Tabletop curriculum action | fixed-base narrow manipulation action | curriculum/regression only |
| Operator/recovery action | teleop/handoff/recovery envelope with authority and replay refs | Phase 4F only |

## Environment Taxonomy

Current env families such as workcell, dishwashing, and drawer/vase remain useful
but must be classified as `fixed_base_tabletop` unless they emit stronger posture
truth.

| Env family | Role | Promotion limit |
| --- | --- | --- |
| `fixed_base_tabletop_*` | manipulation curriculum, regression, replay plumbing, semantic/economic scaffolding | cannot close G1/R1 whole-body readiness |
| `stable_base_mobile_manipulator_*` | fallback, recovery, degraded-mode task continuity, operator handoff rehearsal | cannot replace bipedal readiness |
| `bipedal_whole_body_*` | primary humanoid readiness path | required for G1/R1-class promotion |

Every env/sim artifact should emit:

- posture tag;
- backend truth;
- robot asset refs;
- observation/action schema refs;
- replay/export compatibility refs;
- promotion posture.

## Unitree Sim Integration Target

The Unitree sim lane should be a typed integration target, not an ad hoc env
fork.

Minimum target contract:

- `UnitreeSimBackendContract`: backend id, version, timing model, physics
  posture, unsupported modes;
- `UnitreeRobotAssetRef`: robot family, URDF/Xacro/SRDF or equivalent source,
  joint map, limits, collision geometry, calibration refs;
- `HumanoidObservationSchemaRef`: proprio, IMU, contact, egocentric perception,
  resource/timing fields;
- `HumanoidActionSchemaRef`: whole-body chunk, support-phase constraints,
  fallback/recovery envelope;
- `HumanoidReplayExportRef`: event spine, governance trace, recovery trace,
  posture-tagged training rows;
- `PromotionPosture`: curriculum, fallback, shadow, benchmark candidate, or
  promoted.

## Benchmark Taxonomy

| Benchmark class | Primary posture | Current status | Future closure evidence |
| --- | --- | --- | --- |
| Balance stability | `bipedal_whole_body` | missing | sim/hardware balance receipts, disturbance traces |
| Locomotion-manipulation | `bipedal_whole_body` | missing | whole-body task receipts, action/recovery traces |
| Bimanual / dexterous task | `bipedal_whole_body` | missing | hand/contact receipts, success/failure taxonomy |
| Disturbance recovery | `bipedal_whole_body` | missing | push/slip/stumble recovery traces and watchdog receipts |
| Degraded sensing | all postures | missing | stale/dropout flags, recovery and demotion traces |
| Stable-base fallback | `stable_base_mobile_manipulator` | planning only | fallback selection receipts and task-continuity evidence |
| Tabletop curriculum | `fixed_base_tabletop` | partial | regression/curriculum evidence only |

## Phase 3.5 Output State

This refit establishes the local doctrine and contract deltas needed before
Phase 4 local scaffolding:

- G1/R1-class capacity bands are expressed as placement and timing classes;
- onboard/companion/battery assumptions are explicit and receipt-shaped;
- humanoid observation and action schema deltas are named;
- fixed-base/tabletop envs are partial curriculum domains only;
- Unitree sim integration has a typed target contract;
- benchmark taxonomy is posture-aware;
- bipedal whole-body remains primary, stable-base remains fallback, and
  fixed-base remains curriculum/regression.

Remaining blockers:

- real Unitree sim assets and backend runtime evidence;
- live streams and measured control timing;
- hardware or hardware-in-loop evidence;
- trained whole-body models;
- promotion-grade humanoid benchmarks.

## Local Scaffold Implementation

As of 2026-05-24 this phase is backed by typed local artifacts, not only this
planning note.

Code and CLI surfaces:

- `src/world_model/humanoid_readiness/phase35.py`
- `scripts/economic_world_model/prepare_phase35_humanoid_capacity_env_refit.py`
- `src/world_model/embodiment_actuation/bipedal_readiness.py`
- `scripts/economic_world_model/audit_phase35_bipedal_readiness.py`
- `tests/test_humanoid_phase35_4_65_scaffolds.py`
- `tests/test_humanoid_phase35_bipedal_readiness.py`

Current artifact output:

- `artifacts/economic_world_model/phase35_humanoid_capacity_env_refit/humanoid_phase35_refit_report_v1.json`
- `artifacts/economic_world_model/phase35_bipedal_chassis_scaffold/bipedal_chassis_scaffold_report_v1.json`
- `artifacts/economic_world_model/phase35_bipedal_readiness_audit/phase35_bipedal_readiness_audit_v1.json`
- `capacity_band_count=5`
- `schema_delta_count=10`
- `env_taxonomy_count=3`
- `benchmark_target_count=7`
- `bipedal_chassis_joint_count=29`
- `bipedal_chassis_frame_count=22`
- `bipedal_chassis_joint_limit_envelope_count=29`
- `bipedal_balance_receipt_count=3`
- `canonical_bipedal_chassis_present=true`
- `limb_frame_tree_present=true`
- `joint_limit_envelope_present=true`
- `whole_body_observation_schema_present=true`
- `whole_body_action_schema_present=true`
- `balance_envelope_present=true`
- `local_structural_refit_complete=true`
- `ready_for_phase4_local_sweep=true`
- `local_asset_ingestion_contract_present=true`
- `asset_parse_receipt_count=1` in the default no-asset local run
- `real_asset_parsed=false` in the default no-asset local run
- `kinematic_validators_present=true`
- `joint_vector_validation_receipt_count=2`
- `balance_geometry_report_count=3`
- `whole_body_replay_row_count=3`
- `phase35_no_gpu_no_hardware_prepared=true`

Denied gates remain explicit:

- `ready_for_training=false`
- `training_executed=false`
- `weights_written=false`
- `provider_executed=false`
- `hardware_executed=false`
- `unitree_sim_runtime_executed=false`
- `live_policy_control=false`
- `reward_math_mutation=false`
- `promotion_eligible=false`

## Bipedal Chassis Scaffold Boundary

The Phase 3.5 chassis scaffold moves the local surface beyond hand/gripper
models by adding:

- `HumanoidChassisProfile` for a `g1_29dof` bipedal whole-body target;
- `HumanoidFrameTree` plus `LimbCoordinateFrame` rows for pelvis, torso, head,
  IMU/camera, left/right legs, feet, arms, wrists, and hands;
- one `JointLimitEnvelope` per controlled joint;
- `WholeBodyObservationSchema` and `WholeBodyActionSchema`;
- `BipedalSupportState` rows for double support, left single support, and right
  single support;
- `BalanceEnvelopeReceipt` rows that keep balance evidence observational only.

Numeric joint envelopes are local planning envelopes, not hardware-calibrated
safety limits. They exist so replay rows, schemas, and future sim/hardware
checks have a canonical place to land. Promotion still requires URDF/sim asset
parsing, calibrated transforms, measured IMU/contact/balance streams, Unitree
sim or hardware evidence, and balance benchmark receipts.

## No-GPU / No-Hardware Readiness Audit

The additional Phase 3.5 readiness audit closes the local work that can be done
before real Unitree assets, sim runtime, hardware, or GPU training are
available:

- `HumanoidRobotAssetContract` names required URDF/MJCF/SRDF/USD, joint-map,
  limit, collision-geometry, frame-transform, and calibration roles;
- `RobotAssetParseReceipt` emits either an explicit unavailable-asset receipt
  or a local XML parse receipt for URDF/MJCF/SRDF-style files;
- `KinematicConsistencyReport` checks 21+ DoF coverage, action-channel
  alignment, joint-limit coverage, frame-tree health, left/right symmetry, and
  optional asset joint alignment;
- `JointVectorValidationReceipt` validates neutral planning vectors and a
  synthetic limit-violation probe without live policy authority;
- `BalanceGeometryReport` computes support polygon area and schema-level
  COM/ZMP/COP inclusion where local support slots provide enough geometry;
- `WholeBodyReplayRow` creates shadow replay row slots that tie posture,
  support state, balance receipt, schema refs, joint-limit validation, asset
  contract, kinematic report, and resource-timing refs together.

The default local run intentionally emits `real_asset_parsed=false` because no
real robot asset path is supplied. A local synthetic URDF path can exercise the
parser and alignment validator, but that still remains asset-contract evidence,
not calibrated hardware evidence. The audit keeps `ready_for_unitree_runtime`,
`ready_for_training`, `hardware_calibrated_limits`,
`unitree_sim_runtime_executed`, `hardware_executed`, `training_executed`,
`weights_written`, `live_policy_control`, `reward_math_mutation`, and
`promotion_eligible` false.

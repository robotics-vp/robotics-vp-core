# Phase 4 Local Deployment-Enabler Sweep

Date: 2026-05-23

## Purpose

This pass defines the local non-hardware Phase 4 sweep after the Phase 3.5
humanoid capacity and environment refit.

It is contract/runbook/interface scaffolding only. It does not claim live
streams, real-time control interfaces, hardware execution, Unitree runtime
evidence, sim runtime evidence, training, promotion, or live policy authority.

## Scope

Phase 4 local work is allowed to create:

- typed contracts;
- runbook templates;
- receipt schemas;
- replay/training-aware sidecar slots;
- denied-promotion manifests;
- interface stubs that make future evidence collection possible.

Phase 4 local work must not claim closure until live streams, control
interfaces, hardware/sim runtime evidence, and timing evidence exist.

## 4A Real-Time Control-Loop Separation Contracts

Minimum local contracts:

| Contract | Purpose | Evidence needed later |
| --- | --- | --- |
| `ControlLoopRateContract` | Names servo/reflex, whole-body fast, WM-slow, and offline loop rates | measured loop timing and jitter |
| `ServoReflexBoundaryReceipt` | Proves which work must stay on robot and below governance/economic timescales | runtime control interface evidence |
| `SlowLoopCommandEnvelope` | Bounds what slow WM/economic/meta layers may request | integration with actual action interface |
| `WatchdogDegradationReceipt` | Records stale-data, low-battery, thermal, and comms-triggered degradation | live stream and watchdog traces |
| `AuthoritySplitManifest` | Denies slow-loop live authority until timing/safety gates pass | rollback/demotion tests |

Local denied gates:

- no 200-1000 Hz control claim;
- no live actuator authority;
- no reward math mutation;
- no hardware safety claim.

## 4E Companion Compute and Communication Middleware Contracts

Minimum local contracts:

| Contract | Purpose | Evidence needed later |
| --- | --- | --- |
| `ComputePlacementContract` | Names onboard, companion, cloud, and offline placement classes | measured runtime placement traces |
| `CommsQoSReceipt` | Records latency, stale-data age, packet loss, and freshness class | ROS2/DDS/Unitree middleware evidence |
| `CompanionOffloadEnvelope` | Bounds what may be offloaded without hard real-time authority | timing and failure-mode tests |
| `BatteryThermalComputeReceipt` | Joins compute spend to battery reserve and thermal headroom | live telemetry |
| `DegradedLinkRunbook` | Names fallback behavior under stale/lost comms | operator/recovery and watchdog traces |

Local denied gates:

- no companion-control authority;
- no measured QoS claim;
- no live offload claim;
- no provider or hardware execution claim.

## 4F Operator / Teleop / Recovery Contracts

Minimum local contracts:

| Contract | Purpose | Evidence needed later |
| --- | --- | --- |
| `OperatorHandoffContract` | Names when operator intervention is requested or required | teleop runtime trace |
| `TeleopSessionReceipt` | Captures operator commands, timing, authority, and replay refs | real or sim teleop session |
| `RecoveryTraceReceipt` | Records recovery action, cause, degraded posture, and outcome | sim/hardware recovery drills |
| `FallbackAuthorityGate` | Separates stable-base fallback from bipedal promotion | safety benchmark evidence |
| `PostmortemReplayExport` | Makes recovery traces replay/training-aware | replay export validation |

Local denied gates:

- no operator-loop runtime claim;
- no safety recovery closure claim;
- no promoted fallback authority.

## 4B / 4C / 4D Local Stubs

These phases get schema/runbook/interface stubs only during the local sweep.

| Phase | Stub | Purpose | Full closure waits for |
| --- | --- | --- | --- |
| 4B Sensor Fusion | `SensorFusionInputSchemaStub` | Names camera/depth/IMU/proprio/force streams and timestamp expectations | live streams, calibration, sync evidence |
| 4C Physical Safety | `PhysicalSafetyEnvelopeStub` | Names joint limits, self-collision, e-stop, fall protection, veto receipts | actual safety interface and hardware/sim tests |
| 4D Spatial State / SLAM | `SpatialStateInterfaceStub` | Names localization/map/nav state refs and degraded-spatial-state receipts | SLAM backend runtime and mobile sim/hardware evidence |

These stubs must be explicit about being stubs. They should produce
planning-only manifests and denied-promotion gates rather than masquerading as
runtime evidence.

## Replay and Training Awareness

Every Phase 4 local surface should preserve:

- posture tag: `bipedal_whole_body`, `stable_base_mobile_manipulator`,
  `fixed_base_tabletop`, or `unknown`;
- robot asset refs;
- observation/action schema refs;
- timing and placement refs;
- degraded-mode reason;
- event-spine / governance-trace refs;
- replay export posture;
- promotion posture.

## Closure Boundary

Local Phase 4 can be considered scaffolded when contracts and stubs exist. It is
not closed for deployment until:

- live streams exist;
- actual control interfaces exist;
- timing/jitter traces exist;
- companion-compute middleware is measured;
- operator/teleop/recovery traces exist;
- hardware or honest sim runtime evidence exists;
- rollback/demotion gates pass.

After this local sweep, the roadmap can move to Phase 6.5 local meta-node
neuralization and robustness without claiming Phase 4 hardware closure.

## Local Scaffold Implementation

As of 2026-05-24 this phase is backed by typed local artifacts and a repeatable
CLI.

Code and CLI surfaces:

- `src/world_model/humanoid_readiness/phase4.py`
- `src/world_model/humanoid_readiness/downstream_controller.py`
- `scripts/economic_world_model/prepare_phase4_deployment_enabler_sweep.py`
- `scripts/economic_world_model/prepare_phase4_downstream_controller_scaffold.py`
- `tests/test_humanoid_phase35_4_65_scaffolds.py`
- `tests/test_humanoid_phase4_downstream_controller.py`

Current artifact output:

- `artifacts/economic_world_model/phase4_deployment_enabler_sweep/humanoid_phase4_deployment_enabler_sweep_report_v1.json`
- `artifacts/economic_world_model/phase4_downstream_controller_scaffold/phase4_downstream_controller_scaffold_report_v1.json`
- `contract_surface_count=15`
- `stub_surface_count=3`
- phase counts: `4A=5`, `4B=1`, `4C=1`, `4D=1`, `4E=5`, `4F=5`
- `local_non_hardware_scaffold_complete=true`
- `ready_for_phase65_local_meta_nodes=true`
- `bridge_target_count=5`
- `mode_count=6`
- `proposal_count=6`
- `command_frame_count=6`
- `safety_receipt_count=6`
- `invocation_count=6`
- `controller_receipt_count=6`
- `unitree_bridge_contract_present=true`
- `g1pilot_fallback_contract_present=true`
- `dry_run_controller_present=true`
- `local_downstream_controller_scaffold_complete=true`

Denied gates remain explicit:

- `hardware_dispatch_enabled=false`
- `ros2_publish_attempted=false`
- `unitree_sdk2_write_enabled=false`
- `g1pilot_runtime_invoked=false`
- `training_executed=false`
- `weights_written=false`
- `provider_executed=false`
- `hardware_executed=false`
- `unitree_sim_runtime_executed=false`
- `live_policy_control=false`
- `reward_math_mutation=false`
- `promotion_eligible=false`

## Downstream Controller Scaffold

The local downstream controller pass creates a primitive/fallback controller
surface below WM proposals without granting actuator authority. It is inspired
by the Unitree ROS2 / SDK2 DDS command split and G1Pilot-style upper-body
fallback control, but it does not vendor or invoke those projects.

Current local surfaces:

- `ControllerBridgeTarget` rows for Unitree ROS2 low-level command shape,
  Unitree sport-request fallback shape, G1Pilot joint fallback, G1Pilot
  Cartesian fallback, and an offline OCS2/TSID/Crocoddyl whole-body-control
  reference target;
- `ControllerModeSpec` rows for `hold_pose`, `joint_pd_tracking`,
  `cartesian_upper_body_tracking`, `stable_base_fallback`,
  `operator_teleop_pass_through`, and `e_stop_veto`;
- `DownstreamControllerProposal` rows tied to Phase 3.5 whole-body replay rows;
- `LowLevelCommandFrame` rows with dry-run joint-PD / Cartesian / fallback /
  veto payloads;
- `ControllerSafetyReceipt` rows for joint-limit clamp, stale-data watchdog,
  support-phase, operator override, and e-stop gates;
- `ControllerInvocation` and `ControllerReceipt` rows that deny dispatch while
  keeping replay/training-aware evidence.

The scaffold intentionally includes a synthetic joint-limit clamp probe and an
e-stop veto probe so Phase 4 can test the safety receipt path locally. It still
does not publish `/lowcmd`, `/api/sport/request`, invoke G1Pilot, dispatch to
Unitree SDK2, or claim hardware/sim execution.

Key blockers before Phase 4 can move beyond local dry-run controller evidence:

- Unitree ROS2 / SDK2 runtime installed and verified;
- G1Pilot or equivalent fallback runtime vendored, pinned, or replaced by a
  repo-native equivalent after license/dependency review;
- real robot description, joint map, and calibration sidecars;
- live low-state, IMU, command, and operator/e-stop streams;
- validated low-command, sport-request, or upper-body controller interface;
- measured control-loop timing and jitter;
- physical safety calibration and rollback/demotion tests;
- hardware or honest sim runtime evidence.

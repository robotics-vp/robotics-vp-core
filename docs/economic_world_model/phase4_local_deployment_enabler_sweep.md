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
- `scripts/economic_world_model/prepare_phase4_deployment_enabler_sweep.py`
- `tests/test_humanoid_phase35_4_65_scaffolds.py`

Current artifact output:

- `artifacts/economic_world_model/phase4_deployment_enabler_sweep/humanoid_phase4_deployment_enabler_sweep_report_v1.json`
- `contract_surface_count=15`
- `stub_surface_count=3`
- phase counts: `4A=5`, `4B=1`, `4C=1`, `4D=1`, `4E=5`, `4F=5`
- `local_non_hardware_scaffold_complete=true`
- `ready_for_phase65_local_meta_nodes=true`

Denied gates remain explicit:

- `training_executed=false`
- `weights_written=false`
- `provider_executed=false`
- `hardware_executed=false`
- `unitree_sim_runtime_executed=false`
- `live_policy_control=false`
- `reward_math_mutation=false`
- `promotion_eligible=false`

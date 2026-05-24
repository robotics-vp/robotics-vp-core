# Phase 4 Unitree/G1 Bring-Up Readiness

Date: 2026-05-24

## Purpose

This pass tackles the non-GPU and non-hardware Unitree/G1 blockers one by one
and turns each locally addressable part into a typed receipt.

It is pre-purchase preparation only. It does not claim Unitree runtime,
hardware, honest sim, training, promotion, or live actuator authority.

## Local Blocks

The local readiness pack emits one block receipt for each blocker class:

| Block | Local work completed | Still needed literally |
| --- | --- | --- |
| `runtime_dependency_manifest` | Inventory and marker-check Unitree/G1 OSS roots | fetch/pin missing roots, license review, build/import/launch smoke |
| `g1pilot_or_fallback_review` | Keep G1Pilot-shaped fallback contracts and dry-run frames | fetch/pin/review G1Pilot or replace with repo-native fallback |
| `robot_asset_calibration_intake` | Parse available G1 URDF and compare against canonical 29 controlled joints | hardware calibration sidecars, certified safety limits |
| `live_stream_interface_contracts` | Name low-state, IMU, e-stop, low-command, sport-request, and replay streams | live DDS/SDK stream capture and timestamp/QoS evidence |
| `command_interface_conformance` | Group dry-run command frames by command family and clamp path | actual ROS2/SDK2/G1Pilot write-path echo in honest sim or hardware |
| `timing_jitter_probe` | Emit a local-only `perf_counter` timing probe receipt | DDS/hardware loop timing and jitter traces |
| `physical_safety_preflight` | Emit dispatch-veto preflight receipts for clamp, stale data, e-stop, collision/fall, and stable-base demotion | physical safety calibration and rollback/demotion drills |
| `operator_estop_recovery_runbook` | Emit runbooks for e-stop, stale stream, low balance margin, and teleop takeover | operator drills in honest sim or hardware |
| `sim_hardware_evidence_ledger` | Record present/missing local roots and candidate runtime lanes | successful honest sim launch traces and hardware low-state/control traces |

## Current Artifact Result

Primary artifact:

- `artifacts/economic_world_model/phase4_unitree_bringup_readiness/phase4_unitree_bringup_readiness_report_v1.json`

Current result:

- `block_count=9`
- `dependency_target_count=8`
- `dependency_verified_count=5`
- `asset_joint_subset_aligned=true`
- `stream_contract_count=6`
- `command_conformance_receipt_count=4`
- `timing_jitter_probe_count=1`
- `safety_preflight_receipt_count=5`
- `operator_recovery_runbook_count=4`
- `evidence_ledger_count=1`
- `local_pre_purchase_prepared=true`
- `honest_sim_or_hardware_evidence_present=false`

The current host has verified local layouts for Unitree SDK2, Unitree models,
Unitree RL Gym, Unitree IsaacLab-style sim work, and Unitree LeRobot work.
Unitree ROS2, G1Pilot, and Unitree MuJoCo remain missing local roots on this
host.

## Closure Boundary

This closes local pre-purchase readiness only. The key blockers that remain
before buying or bringing up a G1 are:

- Unitree ROS2 / SDK2 build and interface verification.
- G1Pilot or equivalent fallback runtime review, pinning, and smoke tests.
- Hardware calibration sidecars and certified joint/safety limits.
- Live low-state, IMU, contact, wireless/e-stop, low-command, and sport-request
  streams.
- Actual low-command, sport-request, or upper-body write-path validation.
- DDS or on-robot control-loop timing/jitter measurements.
- Physical safety calibration, demotion, rollback, and recovery drills.
- Operator teleop/e-stop/recovery traces.
- Honest sim or hardware runtime evidence.

No ROS2/DDS publish, Unitree SDK2 write, G1Pilot invocation, Unitree MuJoCo /
RL Gym / IsaacLab sim execution, hardware execution, provider execution,
training, weight writes, live policy control, reward mutation, or promotion
occurred.

# Phase 4 Unitree Local Harnesses

Date: 2026-05-24

## Purpose

This pass pushes the locally controllable Unitree/G1 blockers past checklist
status into executable local harnesses. It uses synthetic traces, no-publish
command validation, mock timing loops, a local safety/recovery state machine,
and source-layout preflight receipts against the local Unitree ROS2, Unitree
MuJoCo, and G1Pilot roots.

It is still local preparation only. It does not observe real streams, publish
ROS2/DDS messages, write Unitree SDK2 commands, invoke G1Pilot, launch MuJoCo,
execute hardware, train weights, mutate reward math, or promote authority.

## Harnesses

| Harness | Local work completed | Still needed literally |
| --- | --- | --- |
| Trace / stream harness | `LowStateTrace`, `ImuTrace`, `WirelessEStopTrace`, and `ContactTrace` JSONL export/import; mock receivers; stale-data validation; replay receipts | live low-state, IMU, contact, and wireless/e-stop streams from honest sim or robot |
| Command shape harness | Parses Unitree ROS2 `.msg` definitions from `/Users/amarmurray/code/unitree_ros2`; validates dry-run `LowCmd` and sport `Request` payload shapes without publishing | ROS2/SDK2/G1Pilot command echo in honest sim or hardware |
| Mock timing / watchdog harness | Runs local producer/consumer timing loops, jitter histograms, command-latency receipts, stale stream events, and watchdog demotion receipts | DDS/network/on-robot timing and jitter traces |
| Safety / recovery harness | Executes local e-stop latch, stale-data veto, joint clamp, stable-base demotion, operator recovery state transitions, and synthetic drill receipts | physical safety calibration, stop-distance evidence, teleop/e-stop drills |
| Unitree MuJoCo / ROS2 preflight | Emits dependency, build-tool/import, source-layout, XML-parse, and launch-request receipts without launching runtimes | actual Unitree ROS2 build/launch and MuJoCo or hardware trace output |

## Current Artifact Result

Primary artifact:

- `artifacts/economic_world_model/phase4_unitree_local_harnesses/phase4_unitree_local_harness_report_v1.json`

Current local result:

- `low_state_trace_count=12`
- `imu_trace_count=12`
- `wireless_estop_trace_count=12`
- `contact_trace_count=12`
- `trace_replay_receipt_count=4`
- `mock_receiver_receipt_count=4`
- `stale_validation_receipt_count=4`
- `ros_message_definition_count=7`
- `command_shape_validation_receipt_count=2`
- `mock_timing_run_receipt_count=1`
- `watchdog_demotion_receipt_count=1`
- `safety_transition_count=5`
- `synthetic_safety_drill_receipt_count=1`
- `runtime_preflight_receipt_count=7`
- `trace_stream_harness_complete=true`
- `command_shape_harness_complete=true`
- `mock_timing_watchdog_harness_complete=true`
- `safety_recovery_harness_complete=true`
- `runtime_preflight_harness_complete=true`
- `local_harnesses_complete=true`

Current local preflight notes:

- Unitree ROS2 source layout and message files are present.
- Host build-tool preflight is partial because `colcon` and `ros2` are not
  installed on this host.
- Unitree MuJoCo source layout and G1 XML parse checks pass.
- The Python `mujoco` module is importable on this host, but no simulation was
  launched.
- G1Pilot source layout and launch-request materialization checks pass, but
  G1Pilot was not invoked.

## Closure Boundary

This closes only the local harness roots for the current blocker list. Full
Phase 4 closure still waits for:

- live low-state, IMU, contact, wireless/e-stop, low-command, and sport-request
  stream traces;
- ROS2/SDK2/G1Pilot command echo in honest sim or hardware;
- DDS/network/on-robot timing and jitter evidence;
- calibrated physical safety envelopes and stop-distance measurements;
- operator teleop/e-stop/recovery drill traces;
- honest Unitree MuJoCo, Isaac, RL Gym, or hardware runtime evidence.

Denied gates remain explicit: `ros2_publish_attempted=false`,
`unitree_sdk2_write_enabled=false`, `g1pilot_runtime_invoked=false`,
`mujoco_launch_executed=false`, `ros2_launch_executed=false`,
`hardware_executed=false`, `training_executed=false`,
`weights_written=false`, `reward_math_mutation=false`, and
`promotion_eligible=false`.

## Runtime Evidence Bridge Follow-On

The follow-on runtime-evidence bridge materialized
`artifacts/economic_world_model/phase4_unitree_runtime_evidence_bridge/phase4_unitree_runtime_evidence_bridge_report_v1.json`.
It goes beyond this local harness by attempting a guarded no-policy MuJoCo
headless step, adding ROS2/colcon build readiness receipts, adding rosbag2/MCAP
trace-ingestion adapter receipts, expanding the physical safety envelope slots,
and running scripted local operator-recovery drills.

On the current host, that follow-on emitted `mujoco_trace_row_count=5` and
`minimal_mujoco_headless_step_executed=true`. That is only local no-policy
simulation evidence; it still does not claim ROS2 bridge runtime, command echo,
policy-controlled sim, physical calibration, teleop runtime, hardware, training,
or promotion.

## Blocker Stress-Probe Follow-On

The blocker stress-probe pass now materializes
`artifacts/economic_world_model/phase4_unitree_blocker_stress_probes/phase4_unitree_blocker_stress_probe_report_v1.json`.
It keeps the local harness boundary while pressing harder on the open blockers:
five G1 MuJoCo XMLs step headlessly, CycloneDDS headers compile, and G1Pilot /
Unitree RL Gym / Unitree IsaacLab / Unitree LeRobot static surfaces are visible.
The same artifact records what still does not work locally: ROS2/colcon,
generated-message imports, SDK2 compile/runtime on this macOS host, rosbag2/MCAP
imports and stream files, physical calibration sidecars, G1Pilot runtime deps,
teleop runtime drills, DDS/network timing, command echo, policy control,
hardware, training, and promotion.

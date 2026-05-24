# Phase 4 Unitree Runtime Evidence Bridge

Date: 2026-05-24

## Purpose

This pass moves beyond local harness scaffolding into guarded runtime-evidence
bridges for the Unitree/G1 Phase 4 lane. It covers the five blocker buckets that
can still be advanced before robot hardware:

- ROS2 / colcon runtime readiness;
- minimal no-policy Unitree MuJoCo headless stepping;
- trace ingestion adapters for JSONL, rosbag2, and MCAP;
- expanded physical safety envelope slots;
- scripted operator / e-stop / recovery drills.

The bridge is deliberately narrow. A successful MuJoCo headless step is local
simulation evidence only. It is not ROS2 bridge evidence, command echo,
policy-controlled sim, physical safety evidence, hardware execution, training,
promotion, or live-control authority.

## Current Artifact Result

Primary artifact:

- `artifacts/economic_world_model/phase4_unitree_runtime_evidence_bridge/phase4_unitree_runtime_evidence_bridge_report_v1.json`

Current result:

- `status=ok`
- `local_runtime_evidence_bridge_complete=true`
- `ros2_runtime_preflight_complete=true`
- `mujoco_headless_trace_attempt_complete=true`
- `minimal_mujoco_headless_step_executed=true`
- `mujoco_trace_row_count=5`
- `trace_ingestion_adapters_complete=true`
- `safety_envelope_expansion_complete=true`
- `operator_drill_runner_complete=true`
- `ros2_runtime_readiness_receipt_count=2`
- `trace_import_adapter_receipt_count=3`
- `safety_envelope_expansion_receipt_count=5`
- `operator_recovery_scenario_count=4`
- `operator_recovery_drill_receipt_count=4`

Denied gates remain explicit:

- `ros2_publish_attempted=false`
- `unitree_sdk2_write_enabled=false`
- `g1pilot_runtime_invoked=false`
- `hardware_executed=false`
- `live_policy_control=false`
- `training_executed=false`
- `weights_written=false`
- `reward_math_mutation=false`
- `promotion_eligible=false`

## Bucket Status

| Bucket | Local result | Still needed literally |
| --- | --- | --- |
| ROS2 / colcon readiness | Native and container profile receipts exist with build/import commands, generated-message import checks, package/message inventory, and missing-tool truth | Install/source ROS2 and colcon, run `colcon build`, source generated setup, import generated Unitree messages |
| MuJoCo headless step | Python `mujoco` loaded Unitree G1 `scene_29dof.xml` and emitted 5 no-policy step rows | ROS2 bridge, policy control, command echo, contact/task metrics, longer sim traces |
| Trace ingestion | Existing JSONL traces import through the typed trace bundle; rosbag2 and MCAP adapters are materialized | Real rosbag2 or MCAP files from sim/hardware streams |
| Safety envelope | Joint clamp, self-collision hook, fall/posture guard, stop-distance slot, and calibrated-limit sidecar receipts exist | Calibrated robot limits, collision geometry validation, measured stop distance |
| Operator recovery | Scenario files and local drill receipts cover stale stream, e-stop latch, low balance margin, and teleop takeover | Runtime teleop stack, operator drill traces, sim/hardware recovery outcomes |

## Remaining Evidence Blockers

- Native host preflight currently records missing `cmake`, `colcon`, and
  `ros2`; the container profile currently records missing `docker`.
- `ros2_colcon_build_and_generated_message_import_not_executed`
- `ros2_sdk2_g1pilot_command_echo_missing`
- `rosbag2_or_mcap_real_stream_import_missing`
- `policy_controlled_mujoco_or_hardware_trace_missing`
- `physical_stop_distance_and_calibrated_safety_limits_missing`
- `operator_teleop_runtime_drill_missing`
- `dds_network_or_on_robot_timing_missing`

## Boundary

This bridge runs local checks only. It does not publish ROS2/DDS messages, write
Unitree SDK2 commands, invoke G1Pilot, run hardware, grant live policy control,
train weights, mutate reward math, or promote authority.

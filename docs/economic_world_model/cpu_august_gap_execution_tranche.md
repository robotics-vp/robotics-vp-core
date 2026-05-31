# CPU August-Gap Execution Tranche

Date: 2026-05-30

## Purpose

This pass burns down the CPU/non-GPU August-gap lane after the Phase 7 scaffold
work by returning to lower-WM evidence. It follows the 2026-05-25 CPU tranche
notes [ad-hoc note] and keeps the boundary additive, typed, receipt-emitting,
and non-promotional.

Primary artifact:

- `artifacts/economic_world_model/cpu_august_gap_execution/cpu_august_gap_execution_report_v1.json`

## Current Artifact Result

- `status=ok`
- `cpu_august_gap_tranche_complete=true`
- `ros2_sdk2_build_message_validation_complete=true`
- `trace_import_complete=true`
- `command_dry_run_complete=true`
- `timing_watchdog_complete=true`
- `safety_recovery_complete=true`
- `cpu_mujoco_probe_complete=true`
- `event_spine_replay_joins_complete=true`
- `lower_wm_ingestion_complete=true`
- `validation_receipt_count=5`
- `event_count=11`
- `decision_count=7`
- `replay_step_count=12`
- `event_replay_join_row_count=6`
- `lower_wm_ingestion_row_count=4`

## Runtime Truth

The tranche starts with ROS2 / Unitree SDK2 build and message validation:

- static Unitree ROS2 message definitions parse locally;
- generated Unitree message imports are still blocked because ROS2/colcon
  generated packages are not installed/importable on this host;
- ROS2 colcon build was not attempted because the required ROS2/colcon runtime
  is unavailable;
- Unitree SDK2 header compile fails on this macOS host;
- Unitree SDK2 CMake build was not attempted because it needs a supported Linux
  runtime;
- no ROS2/DDS message publish or SDK2 command write was attempted.

The subsequent CPU surfaces are now joined:

- synthetic low-state, IMU, wireless/e-stop, and contact trace JSONL bundles
  feed replay rows;
- LowCmd and sport-request command shapes stay dry-run/no-publish;
- timing, stale-stream, watchdog-demotion, safety, and recovery receipts join
  into the event spine;
- the current host emits a 5-row no-policy MuJoCo headless trace and 5 G1 model
  stress receipts;
- lower-WM ingestion rows cover Embodiment / Actuation, Sim / Synth / Physics,
  Perception / Grounding absence truth, and Economic WM shadow ingestion.

## Remaining Evidence Blockers

- `ros2_or_colcon_runtime_missing_for_generated_build_import`
- `unitree_sdk2_linux_build_or_header_compile_missing`
- `ros2_sdk2_g1pilot_command_echo_missing`
- `rosbag2_or_mcap_real_stream_import_missing`
- `live_lowstate_imu_contact_wireless_estop_streams_missing`
- `dds_network_or_on_robot_timing_missing`
- `physical_stop_distance_and_calibrated_safety_limits_missing`
- `operator_teleop_runtime_drill_missing`
- `policy_controlled_mujoco_or_hardware_trace_missing`
- `gpu_provider_training_and_promotion_benchmarks_missing`

## Boundary

This is local CPU validation and receipt joining only. It does not publish
ROS2/DDS messages, write Unitree SDK2 commands, invoke G1Pilot, execute
hardware, run policy-controlled sim, grant live policy control, train weights,
mutate reward math, expand Phase 7 authority, or promote any model.

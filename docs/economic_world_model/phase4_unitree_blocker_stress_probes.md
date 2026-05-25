# Phase 4 Unitree Blocker Stress Probes

Date: 2026-05-24

## Purpose

This pass stress-tests the remaining Phase 4 / Embodiment WM blockers after the
runtime-evidence bridge. The rule is the same as the MuJoCo headless step: if a
local probe succeeds, it becomes a typed receipt; if it fails, the blocker stays
explicit and does not become a capability claim.

Primary artifact:

- `artifacts/economic_world_model/phase4_unitree_blocker_stress_probes/phase4_unitree_blocker_stress_probe_report_v1.json`

## Current Artifact Result

- `status=ok`
- `local_phase4_probe_expansion_complete=true`
- `all_local_probe_attempts_complete=true`
- `probe_receipt_count=14`
- `succeeded_probe_count=8`
- `blocked_probe_count=6`
- `mujoco_model_stress_receipt_count=5`
- `mujoco_model_stress_success_count=5`
- `g1_mujoco_model_stress_succeeded=true`
- `g1pilot_static_surface_succeeded=true`
- `cyclonedds_header_compile_succeeded=true`
- `unitree_sdk2_header_compile_succeeded=false`
- `ros2_runtime_available=false`
- `trace_import_modules_available=false`
- `policy_checkpoint_visible=true`
- `isaaclab_task_surface_visible=true`
- `lerobot_adapter_surface_visible=true`

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

## Probe Outcomes

| Probe | Result | Meaning |
| --- | --- | --- |
| ROS2 / colcon host toolchain | blocked | `/opt/ros`, `cmake`, `colcon`, `ros2`, and `docker` are missing on this host. |
| Python runtime imports | partial | Python `mujoco` is available; `rclpy`, `rosbag2_py`, `mcap`, `cyclonedds`, generated Unitree message modules, `unitree_sdk2py`, Pinocchio, and HPP-FCL are missing. |
| Unitree ROS2 static message surface | success | Local source package/message layout is visible and ready for a future generated-message build/import check. |
| G1Pilot static launch surface | success | G1Pilot Python/launch files parse and expose bring-up/teleop surfaces. Runtime deps are still missing. |
| G1Pilot runtime dependencies | blocked | ROS2, Unitree SDK2 Python, UI, IK, message, and teleop dependencies are not installed locally. |
| CycloneDDS header compile | success | SDK2-bundled CycloneDDS headers compile in a local no-network compile-only probe. |
| Unitree SDK2 header compile | blocked | SDK2 headers fail on this macOS host at Linux-only `sys/sysinfo.h`; use a Linux/ROS2 runtime for real SDK2 compile. |
| Unitree MuJoCo G1 model stress | success | Five local G1 XMLs load and step headlessly for 100 no-policy steps each. |
| Unitree RL Gym policy asset visibility | success | G1 policy checkpoint, deploy config, and G1 robot assets are visible; no policy was loaded or executed. |
| Unitree IsaacLab task surface | success | Local G1 task surfaces are visible; Isaac/Omni runtime modules are not installed. |
| Unitree LeRobot adapter surface | success | Local eval/conversion adapters are visible; no real Unitree dataset or stream was imported. |
| rosbag2 / MCAP modules | blocked | `rosbag2_py` and `mcap` are not installed. |
| physical calibration sidecar | blocked | No measured stop-distance or calibrated-limit sidecar exists. |
| operator teleop runtime surface | static success | Teleop launch surface exists, but runtime launch/drill evidence is blocked by missing deps. |

## MuJoCo Model Stress Detail

The current host loaded and stepped these G1 models for 100 no-policy steps
each:

- `scene_29dof.xml`: `nq=36`, `nv=35`, `nu=29`, `nsensor=95`
- `g1_29dof.xml`: `nq=36`, `nv=35`, `nu=29`, `nsensor=95`
- `scene_23dof.xml`: `nq=36`, `nv=35`, `nu=29`, `nsensor=95`
- `g1_23dof.xml`: `nq=36`, `nv=35`, `nu=29`, `nsensor=95`
- `scene.xml`: `nq=36`, `nv=35`, `nu=29`, `nsensor=95`

This is stronger local model/asset evidence than the prior single XML step, but
it is still no-policy, no-bridge, no-command, and no-hardware evidence.

## Remaining Evidence Blockers

- `ros2_colcon_build_and_generated_message_import_not_executed`
- `ros2_sdk2_g1pilot_command_echo_missing`
- `rosbag2_or_mcap_real_stream_import_missing`
- `policy_controlled_mujoco_or_hardware_trace_missing`
- `physical_stop_distance_and_calibrated_safety_limits_missing`
- `operator_teleop_runtime_drill_missing`
- `dds_network_or_on_robot_timing_missing`

## Boundary

These probes are static, import-only, compile-only, or no-policy MuJoCo checks.
They do not publish ROS2/DDS messages, write Unitree SDK2 commands, invoke
G1Pilot, run hardware, execute policy control, train, write weights, mutate
reward math, or promote authority.

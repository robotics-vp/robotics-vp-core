# Motor Backends

## Overview

Motor backends provide a thin abstraction for training, evaluating, and deploying low-level motor policies from the economics stack.

Interface (see `src/motor_backend/base.py`):

- `train_policy(task_id, objective, datapack_ids, num_envs, max_steps, seed)`
- `evaluate_policy(policy_id, task_id, objective, num_episodes, seed)`
- `deploy_policy_handle(policy_id)`

Each backend returns `MotorTrainingResult` / `MotorEvalResult` with raw metrics from the motor stack plus econ metrics normalized by the economics layer.

## Holosoma Backend

Holosoma is optional and isolated behind `HolosomaBackend`.

Install it with:

```
pip install -r requirements-holosoma.txt
```

The backend maps task IDs to Holosoma presets in `src/motor_backend/holosoma_backend.py`.

## Isaac Lab Backend (Optional)

Isaac Lab is optional. To validate the backend locally (1 tiny episode, datapack export, KPIs),
run:

```
python scripts/local_isaac_smoke.py --tasks peg_in_hole bin_picking
```

If Isaac Lab isn't installed, the script exits with a clean message.
It exits with code 2 to indicate a skipped optional backend.

## OpenVLA / VLA Labeling

Rollout labeling can optionally call OpenVLA when enabled. It is safe to run with OpenVLA disabled; the labeler falls back to stub tags and never raises.

Enablement env vars:

- `OPENVLA_ENABLE=1` (or `VLA_ENABLE=1`)
- `OPENVLA_MODEL_NAME` (alias: `OPENVLA_MODEL`)
- `OPENVLA_DEVICE` (e.g. `cuda:0`, `cpu`)
- `OPENVLA_DTYPE` (e.g. `bfloat16`, `float16`)

Expected RGB inputs:

- `EpisodeRollout.rgb_video_path` should point to an RGB video or image file.
- Frames are treated as HxW with channels-last in RGB order (uint8 0–255).
- The labeler uses the first frame as the OpenVLA visual input.

Minimal frame-to-OpenVLA flow:

```
from PIL import Image
import imageio.v2 as imageio

episode = rollouts.episodes[0]
reader = imageio.get_reader(str(episode.rgb_video_path))
frame = reader.get_data(0)
reader.close()
image = Image.fromarray(frame)
action = controller.predict_action(image, instruction)
```

If OpenVLA is enabled but unavailable or misconfigured, the labeler logs a warning, emits a `vla_error` tag, and continues with stub labels.

## Objective Overlay

Economic objectives are declared via `EconomicObjectiveSpec` and compiled into reward overlays:

- `mpl_per_hour` uses `mpl_weight`
- `energy_kwh` uses `-abs(energy_weight)`
- `error_rate` uses `-abs(error_weight)`
- `novelty_score` uses `novelty_weight`
- `risk_score` uses `-abs(risk_weight)`

Example objective config (YAML):

```
mpl_weight: 1.0
energy_weight: 0.2
error_weight: 0.5
novelty_weight: 0.0
risk_weight: 0.3
extra_weights:
  stability_margin: -0.1
```

Load configs with `src/objectives/loader.py::load_objective_spec`.

## Datapack YAML

Minimal datapack YAML schema (see `src/motor_backend/datapacks.py`):

```
id: logging_v1_base
description: "Baseline logging humanoid task"
motion_clips:
  - path: data/mocap/logging_clip_01.npz
    weight: 1.0
domain_randomization:
  terrain: "flat"
  friction_range: [0.5, 1.0]
curriculum:
  initial_difficulty: 0.1
  max_difficulty: 1.0
```

## SensorBundle v1

Rollout datapacks may include a canonical sensor bundle layout (SensorBundle v1):

```
episode_000/
├── rgb/<camera>.npz         # uint8 (T, H, W, 3)
├── depth/<camera>.npz       # float32 meters (T, H, W)
├── seg/<camera>.npz         # int32 instance IDs (T, H, W)
├── intrinsics/<camera>.json # fx, fy, cx, cy, width, height
├── extrinsics/<camera>.npy  # world_from_cam (T, 4, 4)
└── timestamps_s.npy         # float64 timestamps per frame
```

`metadata.json` includes a `sensor_bundle` entry with the version, camera list,
depth unit, and any applied noise config.

## CLI Example

Run a pricing report with Holosoma training:

```
python scripts/report_task_pricing_and_performance.py \
  --task-id humanoid_locomotion_g1 \
  --motor-backend holosoma \
  --objective-config configs/objectives/example_holosoma_objective.yaml \
  --datapacks configs/datapacks/example_holosoma_datapack.yaml \
  --objective-name "baseline_logging_error_min" \
  --notes "G1/T1 warehouse logging baseline with high error penalty" \
  --num-envs 2048 \
  --max-steps 50000
```

Use `--eval-episodes` to add an evaluation run after training.

## Synthetic Backend

`synthetic_backend` is a deterministic backend for tests and smoke runs. It generates stable metrics without any simulator.

Pricing report example:

```
python scripts/report_task_pricing_and_performance.py \
  --task-id humanoid_locomotion_g1 \
  --motor-backend synthetic \
  --objective-config configs/objectives/example_holosoma_objective.yaml \
  --datapacks configs/datapacks/example_holosoma_datapack.yaml
```

Semantic runner example:

```
python scripts/semantic_run.py \
  --intent "synthetic sanity check" \
  --tags warehouse logging \
  --robot-family G1 \
  --task-id humanoid_locomotion_g1 \
  --motor-backend synthetic
```

## Semantic Simulation API

Programmatic entry point for orchestrator-style runs:

```
from src.orchestrator.semantic_simulation import run_semantic_simulation
from src.ontology.store import OntologyStore

store = OntologyStore(root_dir="data/ontology")
result = run_semantic_simulation(
    store=store,
    tags=["humanoid", "logging"],
    robot_family="G1",
    objective_hint="baseline_logging_error_min",
    notes="semantic run for warehouse logging",
    task_id="humanoid_locomotion_g1",
    rollout_base_dir="artifacts/rollouts",
)
```

This will:
- Resolve datapacks by semantic tags and robot family.
- Run Holosoma training/eval.
- Persist a Scenario node into the ontology.
- Capture rollout artifacts (stubbed for now) and label them via the VLA hook.

CLI wrapper (recommended for quick runs):

```
python scripts/semantic_run.py \
  --intent "reduce logging defects" \
  --tags warehouse logging \
  --robot-family G1 \
  --objective-hint baseline_logging_error_min \
  --task-id humanoid_locomotion_g1 \
  --rollout-dir artifacts/rollouts
```

The legacy YAML + direct backend CLI remains supported, but the semantic entrypoint is the recommended interface for orchestration and feedback loops.

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

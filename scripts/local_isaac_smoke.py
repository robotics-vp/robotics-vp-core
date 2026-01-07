#!/usr/bin/env python3
"""
Local Isaac Lab smoke test (optional, non-CI).

Runs a tiny episode on one or more workcell tasks, exports a datapack, and
prints KPIs. Exits with a clean message when Isaac Lab is not installed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.economics.econ_meter import EconomicMeter
from src.motor_backend.factory import make_motor_backend
from src.objectives.economic_objective import EconomicObjectiveSpec
from src.ontology.models import Robot, Task
from src.ontology.store import OntologyStore


def _isaac_missing_message() -> str:
    return (
        "Isaac Lab backend is optional and not installed. "
        "Install Isaac Lab and provide "
        "`src/motor_backend/workcell_isaaclab_backend.py` with "
        "`WorkcellIsaacLabBackend`."
    )


def _has_isaac_backend_module() -> bool:
    return importlib.util.find_spec("src.motor_backend.workcell_isaaclab_backend") is not None


def _build_backend(task_id: str, out_dir: Path, backend_config: dict[str, object]):
    store = OntologyStore(root_dir=str(out_dir / "ontology"))
    task = Task(task_id=task_id, name=f"Isaac smoke: {task_id}", environment_id="workcell")
    robot = Robot(robot_id="isaac_smoke_bot", name="isaac_smoke_bot")
    store.upsert_task(task)
    store.upsert_robot(robot)
    econ_meter = EconomicMeter(task=task, robot=robot)
    try:
        return make_motor_backend(
            "workcell_isaaclab",
            econ_meter=econ_meter,
            store=store,
            backend_config=backend_config,
        )
    except RuntimeError as exc:
        msg = str(exc).strip()
        if "Isaac Lab backend" in msg or "WorkcellIsaacLabBackend" in msg:
            print(msg, file=sys.stderr)
            return None
        print(
            f"Isaac Lab backend failed to initialize: {msg} "
            "(check CUDA drivers / GPU availability / Isaac install).",
            file=sys.stderr,
        )
        return None
    except ImportError:
        print(_isaac_missing_message(), file=sys.stderr)
        return None


def _print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    ordered = json.dumps(metrics, indent=2, sort_keys=True)
    print(f"{prefix} {ordered}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny Isaac Lab workcell smoke episode.")
    parser.add_argument("--tasks", nargs="*", default=["peg_in_hole", "bin_picking"])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--time-step", type=float, default=0.05)
    parser.add_argument("--out-dir", type=str, default="artifacts/isaac_smoke")
    parser.add_argument("--cameras", nargs="*", default=["front"])
    parser.add_argument("--no-sensor-bundle", action="store_true")
    parser.add_argument("--no-rgb", action="store_true")
    args = parser.parse_args()

    if not _has_isaac_backend_module():
        print(_isaac_missing_message(), file=sys.stderr)
        return 2

    if sys.platform == "darwin":
        print("Isaac Lab requires Linux + NVIDIA GPU; macOS is not supported.", file=sys.stderr)
        return 2

    try:
        import torch

        if not torch.cuda.is_available():
            print(
                "Isaac Lab requires an NVIDIA GPU with CUDA drivers; torch.cuda.is_available()=False.",
                file=sys.stderr,
            )
            return 2
    except Exception:
        pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend_config = {
        "physics_mode": "ISAAC",
        "max_steps": int(args.max_steps),
        "time_step_s": float(args.time_step),
        "capture_sensor_bundle": not args.no_sensor_bundle,
        "sensor_cameras": list(args.cameras),
        "capture_rgb_frames": not args.no_rgb,
        "render_max_frames": min(int(args.max_steps), 50),
    }

    objective = EconomicObjectiveSpec()

    for task_id in args.tasks:
        backend = _build_backend(task_id, out_dir, backend_config)
        if backend is None:
            return 2
        scenario_id = f"isaac_smoke_{task_id}"
        result = backend.evaluate_policy(
            policy_id="isaac_smoke_policy",
            task_id=task_id,
            objective=objective,
            num_episodes=args.episodes,
            scenario_id=scenario_id,
            rollout_base_dir=out_dir,
            seed=args.seed,
        )

        print(f"[isaac_smoke] task={task_id} scenario_id={scenario_id}")
        if result.rollout_bundle:
            scenario_dir = out_dir / scenario_id
            count = len(result.rollout_bundle.episodes)
            print(f"[isaac_smoke] rollout_dir={scenario_dir} episodes={count}")
        _print_metrics("[isaac_smoke] raw_metrics=", dict(result.raw_metrics))
        _print_metrics("[isaac_smoke] econ_metrics=", dict(result.econ_metrics))

    return 0


if __name__ == "__main__":
    sys.exit(main())

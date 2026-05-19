#!/usr/bin/env python3
"""
Local Holosoma runtime smoke test (optional, non-CI).

Evaluates one existing policy on a Unitree-target task and prints raw/econ KPIs.
Exits with code 2 when Holosoma or the required policy root is unavailable.
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
from src.world_model.sim_synth_physics.runtime_layouts import describe_holosoma_policy_contract


def _holosoma_missing_message() -> str:
    return (
        "Holosoma backend is optional and not installed. "
        "Install with `pip install -r requirements-holosoma.txt`."
    )


def _has_holosoma_module() -> bool:
    return importlib.util.find_spec("holosoma") is not None


def _auto_policy_ref() -> tuple[str, str, dict]:
    policy_contract = describe_holosoma_policy_contract({})
    for key in ("policy_ref", "primary_checkpoint_ref"):
        candidate = str(policy_contract.get(key, "") or "")
        if candidate:
            return candidate, key, policy_contract
    return "", "", policy_contract


def _resolve_policy_ref(policy_id: str | None) -> tuple[str, str, dict]:
    if policy_id:
        return str(policy_id), "cli", {}
    return _auto_policy_ref()


def _build_preflight(*, task_id: str, policy_ref: str, policy_source: str) -> dict:
    policy_path = Path(policy_ref) if policy_ref else None
    holosoma_available = _has_holosoma_module()
    policy_exists = bool(policy_path is not None and policy_path.exists())
    missing_preconditions: list[str] = []
    if not holosoma_available:
        missing_preconditions.append("holosoma_python_module")
    if not policy_ref:
        missing_preconditions.append("holosoma_policy_ref")
    elif not policy_exists:
        missing_preconditions.append("holosoma_policy_checkpoint")
    return {
        "version": "holosoma_local_smoke_preflight_v1",
        "task_id": task_id,
        "holosoma_available": holosoma_available,
        "policy_ref": policy_ref,
        "policy_ref_source": policy_source,
        "policy_exists": policy_exists,
        "ready": not missing_preconditions,
        "missing_preconditions": missing_preconditions,
    }


def _write_preflight(out_dir: Path, preflight: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "holosoma_smoke_preflight.json"
    path.write_text(json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_backend(task_id: str, out_dir: Path):
    store = OntologyStore(root_dir=str(out_dir / "ontology"))
    task = Task(task_id=task_id, name=f"Holosoma smoke: {task_id}", environment_id="humanoid")
    robot = Robot(robot_id="holosoma_smoke_bot", name="holosoma_smoke_bot")
    store.upsert_task(task)
    store.upsert_robot(robot)
    econ_meter = EconomicMeter(task=task, robot=robot)
    try:
        return make_motor_backend("holosoma", econ_meter=econ_meter, store=store)
    except RuntimeError as exc:
        print(str(exc).strip(), file=sys.stderr)
        return None


def _print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    ordered = json.dumps(metrics, indent=2, sort_keys=True)
    print(f"{prefix} {ordered}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny Holosoma evaluation smoke episode.")
    parser.add_argument("--task-id", default="humanoid_wbt_g1")
    parser.add_argument(
        "--policy-id",
        default=None,
        help="Policy checkpoint path. If omitted, use the local Holosoma policy contract's selected checkpoint.",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=str, default="artifacts/holosoma_smoke")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Write preflight JSON and exit without attempting runtime execution.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    policy_ref, policy_source, _policy_contract = _resolve_policy_ref(args.policy_id)
    preflight = _build_preflight(
        task_id=str(args.task_id),
        policy_ref=policy_ref,
        policy_source=policy_source,
    )
    preflight_path = _write_preflight(out_dir, preflight)
    if args.preflight_only:
        print(json.dumps({**preflight, "preflight_path": str(preflight_path)}, indent=2, sort_keys=True))
        return 0

    if not preflight["holosoma_available"]:
        print(_holosoma_missing_message(), file=sys.stderr)
        print(f"Preflight written to {preflight_path}", file=sys.stderr)
        return 2

    policy_path = Path(policy_ref)
    if not preflight["policy_exists"]:
        print(
            f"Holosoma smoke requires an existing policy checkpoint; missing: {policy_path}",
            file=sys.stderr,
        )
        print(f"Preflight written to {preflight_path}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    backend = _build_backend(args.task_id, out_dir)
    if backend is None:
        return 2

    scenario_id = f"holosoma_smoke_{args.task_id}"
    result = backend.evaluate_policy(
        policy_id=str(policy_path),
        task_id=args.task_id,
        objective=EconomicObjectiveSpec(),
        num_episodes=args.episodes,
        scenario_id=scenario_id,
        rollout_base_dir=out_dir,
        seed=args.seed,
    )
    print(f"[holosoma_smoke] task={args.task_id} scenario_id={scenario_id}")
    if result.rollout_bundle:
        scenario_dir = out_dir / scenario_id
        count = len(result.rollout_bundle.episodes)
        print(f"[holosoma_smoke] rollout_dir={scenario_dir} episodes={count}")
    _print_metrics("[holosoma_smoke] raw_metrics=", dict(result.raw_metrics))
    _print_metrics("[holosoma_smoke] econ_metrics=", dict(result.econ_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

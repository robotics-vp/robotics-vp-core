#!/usr/bin/env python3
"""
Local Holosoma runtime smoke test (optional, non-CI).

Runs the cheapest local proof available for one existing policy. ONNX policies
use a deploy/inference smoke; serialized Holosoma checkpoints use the native
Holosoma evaluation path. Exits with code 2 when required local preconditions
are unavailable.
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


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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
    policy_kind = "onnx_deploy" if policy_path is not None and policy_path.suffix == ".onnx" else "holosoma_eval"
    onnxruntime_available = _has_module("onnxruntime") if policy_kind == "onnx_deploy" else None
    missing_preconditions: list[str] = []
    if not holosoma_available:
        missing_preconditions.append("holosoma_python_module")
    if not policy_ref:
        missing_preconditions.append("holosoma_policy_ref")
    elif not policy_exists:
        missing_preconditions.append("holosoma_policy_checkpoint")
    if policy_kind == "onnx_deploy" and not onnxruntime_available:
        missing_preconditions.append("onnxruntime_python_module")
    return {
        "version": "holosoma_local_smoke_preflight_v1",
        "task_id": task_id,
        "holosoma_available": holosoma_available,
        "policy_ref": policy_ref,
        "policy_ref_source": policy_source,
        "policy_exists": policy_exists,
        "policy_kind": policy_kind,
        "onnxruntime_available": onnxruntime_available,
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


def _onnx_shape(shape: list[object]) -> list[int]:
    normalized: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            normalized.append(dim)
        else:
            normalized.append(1)
    return normalized


def _onnx_dtype(type_name: str):
    import numpy as np

    mapping = {
        "tensor(double)": np.float64,
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
    }
    return mapping.get(type_name, np.float32)


def _summarize_array(value) -> dict:
    import numpy as np

    arr = np.asarray(value)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(np.isfinite(arr).all()) if np.issubdtype(arr.dtype, np.number) else None,
    }


def _run_onnx_deploy_smoke(backend, *, policy_path: Path, task_id: str, out_dir: Path) -> tuple[dict, Path]:
    import numpy as np

    handle = backend.deploy_policy_handle(str(policy_path))
    session = getattr(handle, "_session", None)
    if session is None:
        raise RuntimeError("ONNX policy handle is not initialized; install onnxruntime and verify the policy file.")

    input_feed = {}
    input_specs = []
    for inp in session.get_inputs():
        shape = _onnx_shape(list(inp.shape))
        dtype = _onnx_dtype(str(inp.type))
        input_feed[inp.name] = np.zeros(shape, dtype=dtype)
        input_specs.append({"name": inp.name, "shape": shape, "type": str(inp.type)})

    action = handle.act(input_feed)
    outputs = action if isinstance(action, list) else [action]
    report = {
        "version": "holosoma_onnx_deploy_smoke_v1",
        "task_id": task_id,
        "policy_id": str(policy_path),
        "input_specs": input_specs,
        "output_summaries": [_summarize_array(output) for output in outputs],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "holosoma_onnx_deploy_smoke.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report, path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny local Holosoma policy smoke.")
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

    if preflight["policy_kind"] == "onnx_deploy":
        report, report_path = _run_onnx_deploy_smoke(
            backend,
            policy_path=policy_path,
            task_id=args.task_id,
            out_dir=out_dir,
        )
        print(f"[holosoma_smoke] task={args.task_id} mode=onnx_deploy")
        print(f"[holosoma_smoke] report_path={report_path}")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

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

#!/usr/bin/env python3
"""Execute a checked-in full-stack training bundle inside the current workspace or a Runpod pod."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORTS))

from scripts.runpod.full_stack_training import (
    DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    discover_workspace_state,
    evaluate_bundles,
    load_bundle_config,
    render_bundle_commands,
    select_bundle,
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=str, default="auto", help="Bundle id or auto")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--receipt-dir", type=str, default="")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when readiness gates are not satisfied",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the execution plan but do not run commands",
    )
    parser.add_argument(
        "--teardown",
        choices=["auto", "stop", "remove", "none"],
        default="auto",
        help="When running inside Runpod and --self-teardown is enabled, choose how the pod should tear down itself.",
    )
    parser.add_argument(
        "--self-teardown",
        action="store_true",
        help="Attempt pod teardown at the end when inside Runpod",
    )
    return parser.parse_args()


def _run_command(command: str, log_path: Path) -> Dict[str, Any]:
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            shell=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return {
        "command": command,
        "log_path": _display_path(log_path),
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
    }


def _teardown_mode(requested: str) -> str:
    if requested != "auto":
        return requested
    if str(os.environ.get("RUNPOD_TEARDOWN", "")).strip():
        return str(os.environ["RUNPOD_TEARDOWN"]).strip()
    if str(os.environ.get("RUNPOD_NETWORK_VOLUME_ID", "")).strip():
        return "remove"
    return "stop"


def _maybe_teardown(requested: str, receipt_dir: Path) -> Dict[str, Any]:
    pod_id = str(os.environ.get("RUNPOD_POD_ID", "")).strip()
    mode = _teardown_mode(requested)
    result: Dict[str, Any] = {"attempted": False, "mode": mode, "pod_id": pod_id}
    if not pod_id or mode == "none":
        return result
    if shutil.which("runpodctl") is None:
        result["error"] = "runpodctl not available inside container"
        return result
    cmd = (
        ["runpodctl", "stop", "pod", pod_id]
        if mode == "stop"
        else ["runpodctl", "remove", "pod", pod_id]
    )
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    teardown_log = receipt_dir / "teardown.log"
    teardown_log.write_text(
        (proc.stdout or "")
        + ("\n" if proc.stdout and proc.stderr else "")
        + (proc.stderr or ""),
        encoding="utf-8",
    )
    result.update(
        {
            "attempted": True,
            "command": " ".join(cmd),
            "returncode": int(proc.returncode),
            "passed": proc.returncode == 0,
            "teardown_log": _display_path(teardown_log),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    config = load_bundle_config(Path(args.config))
    state = discover_workspace_state()
    assessments = evaluate_bundles(config, state)
    selected = select_bundle(assessments, bundle_id=args.bundle)
    if selected is None:
        raise SystemExit(f"No bundle found for selector: {args.bundle}")
    if not selected["manually_runnable"] and not args.force:
        raise SystemExit(
            f"Bundle {selected['bundle_id']} is not runnable: "
            + "; ".join(selected["blockers"])
        )

    started_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt_dir = (
        Path(args.receipt_dir)
        if args.receipt_dir
        else REPO_ROOT / "artifacts" / "runpod_training" / run_id
    )
    if not receipt_dir.is_absolute():
        receipt_dir = REPO_ROOT / receipt_dir
    receipt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = receipt_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    commands = render_bundle_commands(config, state, selected["bundle_id"], run_id)
    plan = {
        "run_id": run_id,
        "bundle": selected,
        "workspace_state": state,
        "commands": commands,
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
    }
    (receipt_dir / "execution_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8"
    )

    command_results: List[Dict[str, Any]] = []
    overall_passed = True
    if not args.dry_run:
        for index, command in enumerate(commands, start=1):
            log_path = logs_dir / f"{index:02d}.log"
            result = _run_command(command, log_path)
            command_results.append(result)
            if not result["passed"]:
                overall_passed = False
                break

    receipt = {
        "run_id": run_id,
        "bundle_id": selected["bundle_id"],
        "bundle_title": selected["title"],
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "commands": command_results if not args.dry_run else [],
        "planned_commands": commands,
        "passed": overall_passed,
        "receipt_dir": _display_path(receipt_dir),
    }
    teardown_result: Dict[str, Any] = {"attempted": False}
    if args.self_teardown:
        teardown_result = _maybe_teardown(args.teardown, receipt_dir)
    receipt["teardown"] = teardown_result
    (receipt_dir / "execution_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

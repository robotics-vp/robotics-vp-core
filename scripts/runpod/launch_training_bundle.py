#!/usr/bin/env python3
"""Launch a checked-in full-stack training bundle on Runpod via runpodctl."""

from __future__ import annotations

import argparse
import json
import shlex
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
    parser.add_argument(
        "--image-name",
        type=str,
        default="",
        help="Container image already containing the desired repo state",
    )
    parser.add_argument(
        "--template-id",
        type=str,
        default="",
        help="Runpod template id to launch instead of an explicit image",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="",
        help="Exact Runpod GPU type string visible to runpodctl",
    )
    parser.add_argument("--gpu-count", type=int, default=0)
    parser.add_argument("--cloud", choices=["secure", "community"], default="secure")
    parser.add_argument("--container-disk-gb", type=int, default=0)
    parser.add_argument("--volume-gb", type=int, default=0)
    parser.add_argument("--volume-path", type=str, default="/workspace")
    parser.add_argument("--network-volume-id", type=str, default="")
    parser.add_argument(
        "--cost",
        type=float,
        default=0.0,
        help="Optional Runpod create-pod cost ceiling in USD/hour",
    )
    parser.add_argument("--mem", type=int, default=0)
    parser.add_argument("--vcpu", type=int, default=0)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra environment variables KEY=VALUE",
    )
    parser.add_argument(
        "--port", action="append", default=[], help="Ports to expose, e.g. 8888/http"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Launch even when the readiness gates are not satisfied",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the launch plan but do not call runpodctl",
    )
    parser.add_argument(
        "--receipt-dir",
        type=str,
        default="",
        help="Local directory for launch receipts. Defaults to artifacts/runpod_training/<timestamp>_<bundle>.",
    )
    return parser.parse_args()


def _resolve_defaults(
    args: argparse.Namespace, selected: Dict[str, object]
) -> Dict[str, object]:
    raw_runpod_defaults = selected.get("recommended_runpod", {})
    runpod_defaults: Dict[str, object] = (
        dict(raw_runpod_defaults) if isinstance(raw_runpod_defaults, dict) else {}
    )
    return {
        "gpu_type": args.gpu_type or str(runpod_defaults.get("gpu_label", "")),
        "gpu_count": args.gpu_count
        or _int_default(runpod_defaults.get("gpu_count"), 1),
        "container_disk_gb": args.container_disk_gb
        or _int_default(runpod_defaults.get("container_disk_gb"), 50),
        "volume_gb": args.volume_gb
        or _int_default(runpod_defaults.get("volume_gb"), 100),
        "hourly_price_usd": _float_default(
            runpod_defaults.get("hourly_price_usd"), 0.0
        ),
    }


def _int_default(value: Any, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _float_default(value: Any, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _build_remote_command(
    bundle_id: str, run_id: str, force: bool, network_volume_id: str
) -> str:
    receipt_dir = f"artifacts/runpod_training/{run_id}"
    pieces = [
        "cd /workspace/robotics-vp-core",
        f"python3 scripts/runpod/execute_training_bundle.py --bundle {bundle_id} --receipt-dir {receipt_dir} --self-teardown",
    ]
    if force:
        pieces[-1] += " --force"
    if network_volume_id:
        pieces.insert(1, f"export RUNPOD_NETWORK_VOLUME_ID={network_volume_id}")
        pieces.insert(2, "export RUNPOD_TEARDOWN=remove")
    else:
        pieces.insert(1, "export RUNPOD_TEARDOWN=stop")
    return " && ".join(pieces)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.dry_run and shutil.which("runpodctl") is None:
        raise SystemExit("runpodctl is required but was not found in PATH")
    if not args.image_name and not args.template_id:
        raise SystemExit("Pass either --image-name or --template-id")
    if args.image_name and args.template_id:
        raise SystemExit("Pass only one of --image-name or --template-id")

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

    defaults = _resolve_defaults(args, selected)
    if not defaults["gpu_type"]:
        raise SystemExit(
            "GPU type is empty. Pass --gpu-type with the exact Runpod GPU type string."
        )

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{selected['bundle_id']}"
    )
    local_receipt_dir = (
        Path(args.receipt_dir)
        if args.receipt_dir
        else REPO_ROOT / "artifacts" / "runpod_training" / run_id
    )
    if not local_receipt_dir.is_absolute():
        local_receipt_dir = REPO_ROOT / local_receipt_dir
    local_receipt_dir.mkdir(parents=True, exist_ok=True)

    remote_command = _build_remote_command(
        selected["bundle_id"],
        run_id,
        force=args.force,
        network_volume_id=args.network_volume_id,
    )
    pod_name = run_id.replace("_", "-")

    cmd: List[str] = [
        "runpodctl",
        "create",
        "pod",
        "--name",
        pod_name,
        "--gpuType",
        str(defaults["gpu_type"]),
        "--gpuCount",
        str(defaults["gpu_count"]),
        "--containerDiskSize",
        str(defaults["container_disk_gb"]),
        "--volumePath",
        args.volume_path,
    ]
    if args.cloud == "secure":
        cmd.append("--secureCloud")
    else:
        cmd.append("--communityCloud")
    if args.image_name:
        cmd.extend(["--imageName", args.image_name])
    if args.template_id:
        cmd.extend(["--templateId", args.template_id])
    if args.network_volume_id:
        cmd.extend(["--networkVolumeId", args.network_volume_id])
    else:
        cmd.extend(["--volumeSize", str(defaults["volume_gb"])])
    if args.cost > 0:
        cmd.extend(["--cost", str(args.cost)])
    if args.mem > 0:
        cmd.extend(["--mem", str(args.mem)])
    if args.vcpu > 0:
        cmd.extend(["--vcpu", str(args.vcpu)])
    env_values = list(args.env)
    env_values.append("PYTHONUNBUFFERED=1")
    env_values.append(f"RUNPOD_LAUNCHED_BUNDLE={selected['bundle_id']}")
    for env_value in env_values:
        cmd.extend(["--env", env_value])
    for port in args.port:
        cmd.extend(["--ports", port])
    cmd.extend(["--args", f"bash -lc '{remote_command}'"])

    launch_plan = {
        "bundle": selected,
        "workspace_state": state,
        "resolved_defaults": defaults,
        "pod_name": pod_name,
        "remote_command": remote_command,
        "runpod_command": cmd,
        "image_name": args.image_name,
        "template_id": args.template_id,
        "network_volume_id": args.network_volume_id,
        "dry_run": bool(args.dry_run),
    }
    _write_text(
        local_receipt_dir / "launch_plan.json",
        json.dumps(launch_plan, indent=2, sort_keys=True),
    )
    _write_text(local_receipt_dir / "launch_command.sh", shlex.join(cmd) + "\n")

    if args.dry_run:
        print(json.dumps(launch_plan, indent=2, sort_keys=True))
        return

    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    _write_text(local_receipt_dir / "create_pod.stdout", proc.stdout or "")
    _write_text(local_receipt_dir / "create_pod.stderr", proc.stderr or "")
    receipt = {
        "run_id": run_id,
        "bundle_id": selected["bundle_id"],
        "pod_name": pod_name,
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "local_receipt_dir": _display_path(local_receipt_dir),
        "estimated_cost_usd": selected.get("estimated_cost_usd", {}),
        "remote_receipt_dir": f"artifacts/runpod_training/{run_id}",
        "launched_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_text(
        local_receipt_dir / "launch_receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()

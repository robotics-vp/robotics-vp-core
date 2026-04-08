#!/usr/bin/env python3
"""Evaluate and optionally execute ready non-training GPU run backlog items."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.orchestrator.non_training_gpu_run_backlog import (  # noqa: E402
    DEFAULT_NON_TRAINING_GPU_RUN_BACKLOG_PATH,
    collect_host_capabilities,
    evaluate_non_training_gpu_run_backlog,
)


def _execute(command: str, cwd: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    stderr_lines = [line for line in proc.stderr.splitlines() if line.strip()]
    return {
        "command": command,
        "cwd": str(cwd),
        "passed": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout_tail": stdout_lines[-20:],
        "stderr_tail": stderr_lines[-20:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan the non-training GPU run backlog and optionally execute ready entries."
    )
    parser.add_argument("--backlog", default=str(DEFAULT_NON_TRAINING_GPU_RUN_BACKLOG_PATH))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--execute-ready", action="store_true")
    parser.add_argument(
        "--include-manual",
        action="store_true",
        help="Also execute ready runs with auto_trigger=false.",
    )
    args = parser.parse_args()

    host = collect_host_capabilities()
    assessments = evaluate_non_training_gpu_run_backlog(
        backlog_path=Path(args.backlog),
        host_capabilities=host,
    )

    execution_results = []
    if args.execute_ready:
        for assessment in assessments:
            if not assessment.ready:
                continue
            if not assessment.item.auto_trigger and not args.include_manual:
                continue
            cwd = Path(assessment.item.cwd)
            if not cwd.is_absolute():
                cwd = ROOT / cwd
            execution_results.append(
                {
                    "loop_run_id": assessment.item.loop_run_id,
                    "title": assessment.item.title,
                    "execution": _execute(assessment.item.command, cwd),
                }
            )

    summary = {
        "backlog_path": str(Path(args.backlog).resolve()),
        "host_capabilities": host,
        "ready_count": sum(1 for assessment in assessments if assessment.ready),
        "auto_trigger_ready_count": sum(
            1 for assessment in assessments if assessment.ready and assessment.item.auto_trigger
        ),
        "assessments": [assessment.to_dict() for assessment in assessments],
        "execution_results": execution_results,
    }

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

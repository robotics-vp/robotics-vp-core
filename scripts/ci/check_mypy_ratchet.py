#!/usr/bin/env python3
"""Fail when mypy errors exceed the stored legacy-debt baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "config" / "quality_ratchet.json"
TARGET_PYTHON_VERSION = "3.9"


def _load_baseline() -> int:
    payload = json.loads(BASELINE_PATH.read_text())
    return int(payload["mypy"]["baseline_errors"])


def _run_mypy() -> tuple[int, list[str]]:
    command = [
        sys.executable,
        "-m",
        "mypy",
        "src/",
        "--python-version",
        TARGET_PYTHON_VERSION,
        "--hide-error-context",
        "--no-color-output",
        "--show-error-codes",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode not in (0, 1):
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    lines = [line for line in completed.stdout.splitlines() if ": error:" in line]
    return len(lines), lines


def main() -> None:
    baseline = _load_baseline()
    current, errors = _run_mypy()
    delta = current - baseline
    print(f"mypy ratchet: baseline={baseline} current={current} delta={delta:+d}")
    if current <= baseline:
        return

    print("First 10 mypy errors:")
    for line in errors[:10]:
        print(f"- {line}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

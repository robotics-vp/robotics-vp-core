#!/usr/bin/env python3
"""Fail CI if mypy error count regresses above baseline."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "config" / "quality_ratchet.json"
ERROR_RE = re.compile(r"Found\s+(\d+)\s+errors?")


def load_baseline() -> int:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return int(payload["mypy"]["baseline_errors"])


def run_mypy() -> tuple[int, str]:
    cmd = [sys.executable, "-m", "mypy", "src/", "--no-color-output", "--hide-error-context"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

    combined = "\n".join(part for part in [proc.stdout, proc.stderr] if part).strip()
    if proc.returncode == 0:
        return 0, ""

    match = ERROR_RE.search(combined)
    if match:
        return int(match.group(1)), ""

    return -1, combined


def main() -> int:
    baseline = load_baseline()
    count, err = run_mypy()
    if count < 0:
        print("mypy invocation failed or error count could not be parsed")
        print(err[-4000:].strip())
        return 2

    delta = count - baseline
    print(f"mypy baseline: {baseline}")
    print(f"mypy current:  {count}")
    print(f"mypy delta:    {delta:+d}")

    if count > baseline:
        print("mypy ratchet failed: error count increased")
        return 1

    if count < baseline:
        print("mypy ratchet passed: errors decreased; update config/quality_ratchet.json")
    else:
        print("mypy ratchet passed: no regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

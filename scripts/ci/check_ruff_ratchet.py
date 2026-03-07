#!/usr/bin/env python3
"""Fail CI if ruff violations regress above baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "config" / "quality_ratchet.json"


def load_baseline() -> int:
    payload = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return int(payload["ruff"]["baseline_errors"])


def run_ruff() -> tuple[int, str]:
    cmd = [sys.executable, "-m", "ruff", "check", ".", "--output-format", "json"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return -1, proc.stderr or proc.stdout
    output = (proc.stdout or "").strip()
    if not output:
        return 0, ""
    diagnostics = json.loads(output)
    return len(diagnostics), ""


def main() -> int:
    baseline = load_baseline()
    count, err = run_ruff()
    if count < 0:
        print("ruff invocation failed")
        print(err.strip())
        return 2

    delta = count - baseline
    print(f"ruff baseline: {baseline}")
    print(f"ruff current:  {count}")
    print(f"ruff delta:    {delta:+d}")

    if count > baseline:
        print("ruff ratchet failed: violations increased")
        return 1

    if count < baseline:
        print("ruff ratchet passed: violations decreased; update config/quality_ratchet.json")
    else:
        print("ruff ratchet passed: no regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

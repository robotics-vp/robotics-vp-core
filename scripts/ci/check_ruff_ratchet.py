#!/usr/bin/env python3
"""Fail when Ruff findings exceed the stored legacy-debt baseline."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "config" / "quality_ratchet.json"


def _load_baseline() -> int:
    payload = json.loads(BASELINE_PATH.read_text())
    return int(payload["ruff"]["baseline_errors"])


def _run_ruff() -> tuple[int, list[dict[str, object]]]:
    command = [
        sys.executable,
        "-m",
        "ruff",
        "check",
        ".",
        "--output-format",
        "json",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode not in (0, 1):
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    try:
        diagnostics = json.loads(completed.stdout or "[]")
    except json.JSONDecodeError as exc:
        sys.stderr.write(completed.stdout)
        raise SystemExit(f"Unable to parse Ruff JSON output: {exc}") from exc
    return len(diagnostics), diagnostics


def main() -> None:
    baseline = _load_baseline()
    current, diagnostics = _run_ruff()
    delta = current - baseline
    print(f"ruff ratchet: baseline={baseline} current={current} delta={delta:+d}")
    if current <= baseline:
        return

    print("First 10 Ruff diagnostics:")
    for diagnostic in diagnostics[:10]:
        filename = diagnostic.get("filename", "<unknown>")
        location = diagnostic.get("location", {})
        row = location.get("row", "?") if isinstance(location, dict) else "?"
        code = diagnostic.get("code", "<unknown>")
        message = diagnostic.get("message", "")
        print(f"- {filename}:{row} {code} {message}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
